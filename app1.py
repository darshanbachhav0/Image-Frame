import os
import re
import zipfile
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageChops, ImageFile, ImageOps, UnidentifiedImageError


# Helps PIL accept slightly imperfect mobile images instead of failing late.
ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = Path(__file__).resolve().parent
FRAMES_DIR = BASE_DIR / "frames"
SUPPORTED_UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp"]

# Detection is done on resized copies only. Final output is still created at frame size.
DETECTION_MAX_SIZE = 850
TRIM_DETECTION_MAX_SIZE = 900
PREVIEW_MAX_SIZE = (420, 420)
DEFAULT_PREVIEW_LIMIT = 24
_THREAD_LOCAL = threading.local()

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:  # Pillow < 9
    RESAMPLE_LANCZOS = Image.LANCZOS
    RESAMPLE_NEAREST = Image.NEAREST


# -----------------------------
# Small utilities
# -----------------------------

def sanitize_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^\w.\- ]+", "_", name, flags=re.UNICODE).strip()
    name = re.sub(r"\s+", " ", name)
    return name or "image"


def load_frames(frames_dir: Path = FRAMES_DIR) -> Tuple[List[Path], Optional[str]]:
    if not frames_dir.exists():
        return [], "frames/ folder missing. Add PNG frames inside the frames folder."
    if not frames_dir.is_dir():
        return [], "frames exists, but it is not a folder."

    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        return [], "No PNG frames found in frames/."
    return frames, None


def correct_exif_orientation(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    # compress_level=1 is much faster than the default and does not change image quality.
    image.save(buffer, format="PNG", compress_level=1, optimize=False)
    return buffer.getvalue()


def png_bytes_to_image(data: bytes) -> Image.Image:
    img = Image.open(BytesIO(data))
    img.load()
    return img.convert("RGBA")


def get_thread_cached_frame(frame_bytes: bytes) -> Image.Image:
    """Decode the selected frame once per worker thread.

    This avoids repeatedly opening the same PNG frame for every uploaded photo.
    The returned image is used read-only by the worker.
    """
    frame_key = id(frame_bytes)
    if getattr(_THREAD_LOCAL, "frame_key", None) != frame_key:
        _THREAD_LOCAL.frame_key = frame_key
        _THREAD_LOCAL.frame_image = png_bytes_to_image(frame_bytes)
    return _THREAD_LOCAL.frame_image


def resize_for_detection(image: Image.Image, max_size: int = DETECTION_MAX_SIZE) -> Tuple[Image.Image, float, float]:
    width, height = image.size
    largest = max(width, height)

    if largest <= max_size:
        return image.copy(), 1.0, 1.0

    scale = max_size / largest
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    resized = image.resize((new_w, new_h), RESAMPLE_LANCZOS)

    return resized, width / new_w, height / new_h


def scale_bbox_to_original(
    bbox: Tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
    original_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width, height = original_size

    left = max(0, int(left * scale_x))
    top = max(0, int(top * scale_y))
    right = min(width, int(right * scale_x))
    bottom = min(height, int(bottom * scale_y))

    return left, top, right, bottom


def shrink_bbox(
    bbox: Tuple[int, int, int, int],
    shrink_percent: float,
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    box_w = right - left
    box_h = bottom - top

    dx = int(box_w * shrink_percent)
    dy = int(box_h * shrink_percent)

    width, height = image_size
    left = max(0, left + dx)
    top = max(0, top + dy)
    right = min(width, right - dx)
    bottom = min(height, bottom - dy)

    if right <= left or bottom <= top:
        return bbox
    return left, top, right, bottom


# -----------------------------
# Frame opening detection
# -----------------------------

def create_transparent_opening_mask(frame: Image.Image) -> Image.Image:
    alpha = frame.convert("RGBA").getchannel("A")
    return alpha.point(lambda a: 255 if a <= 35 else 0)


def create_dark_opening_mask(frame: Image.Image) -> Image.Image:
    rgba = frame.convert("RGBA")
    alpha = rgba.getchannel("A")
    lum = rgba.convert("L")

    dark = lum.point(lambda p: 255 if p <= 65 else 0)
    visible = alpha.point(lambda a: 255 if a > 20 else 0)
    return ImageChops.multiply(dark, visible)


def create_light_opening_mask(frame: Image.Image) -> Image.Image:
    rgba = frame.convert("RGBA")
    alpha = rgba.getchannel("A")
    hsv = rgba.convert("RGB").convert("HSV")
    _, saturation, value = hsv.split()

    low_saturation = saturation.point(lambda p: 255 if p <= 55 else 0)
    bright = value.point(lambda p: 255 if p >= 145 else 0)
    visible = alpha.point(lambda a: 255 if a > 20 else 0)
    return ImageChops.multiply(ImageChops.multiply(low_saturation, bright), visible)


def flood_region_from_center_seeds(
    mask: Image.Image,
) -> Tuple[Optional[Image.Image], Optional[Tuple[int, int, int, int]], int]:
    mask = mask.convert("L")
    width, height = mask.size
    pixels = mask.load()

    seed_points = [
        (width // 2, height // 2),
        (width // 2, int(height * 0.42)),
        (width // 2, int(height * 0.58)),
        (int(width * 0.42), height // 2),
        (int(width * 0.58), height // 2),
        (int(width * 0.35), int(height * 0.50)),
        (int(width * 0.65), int(height * 0.50)),
    ]

    queue = deque()
    visited = bytearray(width * height)

    for sx, sy in seed_points:
        if 0 <= sx < width and 0 <= sy < height and pixels[sx, sy] > 0:
            idx = sy * width + sx
            visited[idx] = 1
            queue.append((sx, sy))

    if not queue:
        return None, None, 0

    region = Image.new("L", (width, height), 0)
    region_pixels = region.load()

    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    count = 0

    while queue:
        x, y = queue.popleft()
        if pixels[x, y] == 0:
            continue

        region_pixels[x, y] = 255
        count += 1

        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y

        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < width and 0 <= ny < height:
                idx = ny * width + nx
                if not visited[idx]:
                    visited[idx] = 1
                    if pixels[nx, ny] > 0:
                        queue.append((nx, ny))

    if count == 0:
        return None, None, 0

    return region, (min_x, min_y, max_x + 1, max_y + 1), count


def find_best_opening_region(
    mask: Image.Image,
    min_area_ratio: float = 0.015,
) -> Tuple[Optional[Image.Image], Optional[Tuple[int, int, int, int]]]:
    mask = mask.convert("L")
    width, height = mask.size

    region, bbox, count = flood_region_from_center_seeds(mask)
    if region is not None and bbox is not None and count >= width * height * min_area_ratio:
        return region, bbox

    return None, None


def fallback_photo_area(frame_size: Tuple[int, int]) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    width, height = frame_size
    left = int(width * 0.12)
    top = int(height * 0.16)
    right = int(width * 0.88)
    bottom = int(height * 0.84)

    mask = Image.new("L", frame_size, 0)
    mask.paste(255, (left, top, right, bottom))
    return mask, (left, top, right, bottom)


def apply_opening_mask_to_frame(frame: Image.Image, opening_mask: Image.Image) -> Image.Image:
    frame = frame.convert("RGBA")
    opening_mask = opening_mask.convert("L").resize(frame.size, RESAMPLE_NEAREST)

    alpha = frame.getchannel("A")
    new_alpha = ImageChops.subtract(alpha, opening_mask)
    frame.putalpha(new_alpha)
    return frame


@st.cache_data(show_spinner=False)
def prepare_frame_cached(
    frame_path_str: str,
    frame_mtime: float,
    frame_size_bytes: int,
) -> Tuple[Optional[bytes], Optional[Tuple[int, int, int, int]], Optional[str], bool]:
    # frame_mtime and frame_size_bytes are cache-busting arguments.
    _ = (frame_mtime, frame_size_bytes)

    try:
        with Image.open(frame_path_str) as raw_frame:
            raw_frame.load()
            frame = raw_frame.convert("RGBA")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return None, None, f"Frame error: {exc}", False

    original_size = frame.size
    detection_frame, scale_x, scale_y = resize_for_detection(frame, DETECTION_MAX_SIZE)

    masks = [
        create_transparent_opening_mask(detection_frame),
        create_dark_opening_mask(detection_frame),
        create_light_opening_mask(detection_frame),
    ]

    opening_mask_small = None
    bbox_small = None
    was_fixed = False

    for idx, mask in enumerate(masks):
        opening_mask_small, bbox_small = find_best_opening_region(mask)
        if opening_mask_small is not None and bbox_small is not None:
            was_fixed = idx != 0
            break

    if opening_mask_small is None or bbox_small is None:
        opening_mask, bbox = fallback_photo_area(original_size)
    else:
        opening_mask = opening_mask_small.resize(original_size, RESAMPLE_NEAREST)
        bbox = scale_bbox_to_original(bbox_small, scale_x, scale_y, original_size)

    bbox = shrink_bbox(bbox, 0.006, original_size)
    frame = apply_opening_mask_to_frame(frame, opening_mask)

    return image_to_png_bytes(frame), bbox, None, was_fixed


@st.cache_data(show_spinner=False)
def make_frame_preview_cached(frame_bytes: bytes) -> bytes:
    frame = png_bytes_to_image(frame_bytes)
    return make_preview_bytes(frame, max_size=PREVIEW_MAX_SIZE, quality=82)


# -----------------------------
# Student image fitting
# -----------------------------

def required_cover_scale(source_size: Tuple[int, int], target_size: Tuple[int, int]) -> float:
    source_w, source_h = source_size
    target_w, target_h = target_size
    return max(target_w / source_w, target_h / source_h)


def required_contain_scale(source_size: Tuple[int, int], target_size: Tuple[int, int]) -> float:
    source_w, source_h = source_size
    target_w, target_h = target_size
    return min(target_w / source_w, target_h / source_h)


def estimate_photo_background_color(image: Image.Image) -> Tuple[int, int, int]:
    rgb = image.convert("RGB")
    width, height = rgb.size

    sample_points = [
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
        (width // 2, 0),
        (width // 2, height - 1),
    ]

    colors = [rgb.getpixel(point) for point in sample_points]
    r = int(sum(c[0] for c in colors) / len(colors))
    g = int(sum(c[1] for c in colors) / len(colors))
    b = int(sum(c[2] for c in colors) / len(colors))

    if max(r, g, b) >= 210 and max(r, g, b) - min(r, g, b) <= 35:
        return 255, 255, 255

    return r, g, b


def crop_with_padding(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    pad_x_ratio: float,
    pad_top_ratio: float,
    pad_bottom_ratio: float,
) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = bbox

    box_w = right - left
    box_h = bottom - top

    pad_x = int(box_w * pad_x_ratio)
    pad_top = int(box_h * pad_top_ratio)
    pad_bottom = int(box_h * pad_bottom_ratio)

    left = max(0, left - pad_x)
    right = min(width, right + pad_x)
    top = max(0, top - pad_top)
    bottom = min(height, bottom + pad_bottom)

    return image.crop((left, top, right, bottom))


def scale_bbox_between_sizes(
    bbox: Tuple[int, int, int, int],
    from_size: Tuple[int, int],
    to_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    from_w, from_h = from_size
    to_w, to_h = to_size
    sx = to_w / from_w
    sy = to_h / from_h

    left, top, right, bottom = bbox
    return (
        max(0, int(left * sx)),
        max(0, int(top * sy)),
        min(to_w, int(right * sx)),
        min(to_h, int(bottom * sy)),
    )


def detect_trim_bbox_fast(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Find content bounds on a small copy, then scale the bbox to the original.

    This keeps the same final output quality because only detection is downscaled;
    cropping/resizing is still done from the original pixels.
    """
    img = image.convert("RGBA")
    original_size = img.size
    detection_img, _, _ = resize_for_detection(img, TRIM_DETECTION_MAX_SIZE)
    detection_size = detection_img.size

    alpha = detection_img.getchannel("A")
    alpha_bbox = alpha.point(lambda a: 255 if a > 15 else 0).getbbox()

    if alpha_bbox:
        det_w, det_h = detection_size
        alpha_area = (alpha_bbox[2] - alpha_bbox[0]) * (alpha_bbox[3] - alpha_bbox[1])
        if alpha_area < det_w * det_h * 0.95:
            return scale_bbox_between_sizes(alpha_bbox, detection_size, original_size)

    rgb = detection_img.convert("RGB")
    bg_color = estimate_photo_background_color(rgb)
    bg = Image.new("RGB", rgb.size, bg_color)
    diff = ImageChops.difference(rgb, bg)
    gray = diff.convert("L")

    mask = gray.point(lambda p: 255 if p > 22 else 0)
    bbox = mask.getbbox()

    if bbox is None:
        return None

    left, top, right, bottom = bbox
    crop_w = right - left
    crop_h = bottom - top
    det_w, det_h = detection_size

    if crop_w < det_w * 0.25 or crop_h < det_h * 0.25:
        return None
    if crop_w > det_w * 0.96 and crop_h > det_h * 0.96:
        return None

    return scale_bbox_between_sizes(bbox, detection_size, original_size)


def auto_trim_photo_margins(image: Image.Image) -> Image.Image:
    img = image.convert("RGBA")
    width, height = img.size

    # Preserve the original high-quality alpha trimming behavior for transparent PNG/WebP files.
    alpha = img.getchannel("A")
    alpha_bbox = alpha.point(lambda a: 255 if a > 15 else 0).getbbox()
    if alpha_bbox:
        alpha_area = (alpha_bbox[2] - alpha_bbox[0]) * (alpha_bbox[3] - alpha_bbox[1])
        if alpha_area < width * height * 0.95:
            return crop_with_padding(img, alpha_bbox, 0.10, 0.10, 0.08)

    # For normal photos, detect margins on a small copy, then crop full-resolution pixels.
    bbox = detect_trim_bbox_fast(img)
    if bbox is None:
        return img

    left, top, right, bottom = bbox
    crop_w = right - left
    crop_h = bottom - top

    if crop_w < width * 0.25 or crop_h < height * 0.25:
        return img

    return crop_with_padding(img, bbox, 0.18, 0.18, 0.12)


def cover_crop_to_area(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    target_w, target_h = target_size
    source_w, source_h = image.size

    scale = required_cover_scale((source_w, source_h), target_size)
    new_w = max(target_w, int(round(source_w * scale)))
    new_h = max(target_h, int(round(source_h * scale)))

    resized = image.resize((new_w, new_h), RESAMPLE_LANCZOS)

    left = (new_w - target_w) // 2
    extra_h = new_h - target_h
    top = int(extra_h * 0.42) if extra_h > 0 else 0

    return resized.crop((left, top, left + target_w, top + target_h))


def contain_fit_to_area(
    image: Image.Image,
    target_size: Tuple[int, int],
    background_color: Tuple[int, int, int],
) -> Image.Image:
    target_w, target_h = target_size
    source_w, source_h = image.size

    scale = required_contain_scale((source_w, source_h), target_size) * 0.985
    new_w = max(1, int(round(source_w * scale)))
    new_h = max(1, int(round(source_h * scale)))

    resized = image.resize((new_w, new_h), RESAMPLE_LANCZOS)
    area_canvas = Image.new("RGBA", target_size, (*background_color, 255))

    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    area_canvas.alpha_composite(resized.convert("RGBA"), (paste_x, paste_y))

    return area_canvas


def smart_fit_photo_to_area(image: Image.Image, area_size: Tuple[int, int]) -> Image.Image:
    # Estimate background before trim, as in the original behavior.
    detection_for_bg, _, _ = resize_for_detection(image, TRIM_DETECTION_MAX_SIZE)
    original_bg = estimate_photo_background_color(detection_for_bg)

    trimmed = auto_trim_photo_margins(image)
    source_w, source_h = trimmed.size
    area_w, area_h = area_size

    source_aspect = source_w / source_h
    area_aspect = area_w / area_h
    aspect_ratio_gap = max(source_aspect / area_aspect, area_aspect / source_aspect)

    if aspect_ratio_gap >= 1.28:
        return contain_fit_to_area(trimmed, area_size, original_bg)

    return cover_crop_to_area(trimmed, area_size)


def place_student_inside_frame_area(
    student_image: Image.Image,
    frame_size: Tuple[int, int],
    photo_area_bbox: Tuple[int, int, int, int],
) -> Image.Image:
    left, top, right, bottom = photo_area_bbox
    area_w = right - left
    area_h = bottom - top

    fitted_area = smart_fit_photo_to_area(student_image, (area_w, area_h))

    canvas = Image.new("RGBA", frame_size, (255, 255, 255, 255))
    canvas.alpha_composite(fitted_area, (left, top))
    return canvas


def apply_frame(student_background: Image.Image, frame: Image.Image) -> Image.Image:
    if student_background.mode != "RGBA":
        student_background = student_background.convert("RGBA")
    if frame.mode != "RGBA":
        frame = frame.convert("RGBA")
    return Image.alpha_composite(student_background, frame)


def rgba_to_rgb(
    image: Image.Image,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGB")

    background = Image.new("RGB", image.size, background_color)
    background.paste(image, mask=image.getchannel("A"))
    return background


# -----------------------------
# Saving, previews, ZIP
# -----------------------------

def save_output_image_to_bytes(
    image: Image.Image,
    output_format: str,
    jpg_quality: int,
    icc_profile: Optional[bytes],
) -> bytes:
    buffer = BytesIO()
    save_kwargs: Dict[str, object] = {}

    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile

    if output_format.upper() == "PNG":
        image.save(
            buffer,
            format="PNG",
            compress_level=1,
            optimize=False,
            **save_kwargs,
        )
    else:
        rgb_image = rgba_to_rgb(image)
        # optimize=True is CPU-heavy on Render. Disabling it keeps visual quality the same.
        rgb_image.save(
            buffer,
            format="JPEG",
            quality=int(jpg_quality),
            subsampling=0,
            optimize=False,
            progressive=False,
            **save_kwargs,
        )

    return buffer.getvalue()


def make_preview_bytes(
    image: Image.Image,
    max_size: Tuple[int, int] = PREVIEW_MAX_SIZE,
    quality: int = 82,
) -> bytes:
    preview = image.copy()
    preview.thumbnail(max_size, RESAMPLE_LANCZOS)

    if preview.mode == "RGBA":
        background = Image.new("RGB", preview.size, (255, 255, 255))
        background.paste(preview, mask=preview.getchannel("A"))
        preview = background
    else:
        preview = preview.convert("RGB")

    buffer = BytesIO()
    preview.save(buffer, format="JPEG", quality=quality, optimize=False)
    return buffer.getvalue()


def create_zip_from_results(results: List[Dict[str, object]]) -> bytes:
    zip_buffer = BytesIO()

    # PNG/JPG are already compressed. ZIP_DEFLATED wastes CPU and slows Render a lot.
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_STORED) as zip_file:
        for result in results:
            zip_file.writestr(str(result["filename"]), result["file_bytes"])

    return zip_buffer.getvalue()


def get_mime_type(output_format: str) -> str:
    return "image/jpeg" if output_format.upper() == "JPG" else "image/png"


def make_unique_output_name(
    original_filename: str,
    frame_filename: str,
    output_format: str,
    used_names: set,
) -> str:
    original_stem = sanitize_filename(Path(original_filename).stem)
    frame_stem = sanitize_filename(Path(frame_filename).stem)

    extension = "jpg" if output_format.upper() == "JPG" else "png"
    base_name = f"{original_stem}_{frame_stem}"

    candidate = f"{base_name}.{extension}"
    counter = 1

    while candidate.lower() in used_names:
        candidate = f"{base_name}_{counter}.{extension}"
        counter += 1

    used_names.add(candidate.lower())
    return candidate


# -----------------------------
# Per-image processing
# -----------------------------

def open_student_image_from_bytes(
    filename: str,
    image_bytes: bytes,
) -> Tuple[Optional[Image.Image], Optional[bytes], Optional[str]]:
    try:
        with Image.open(BytesIO(image_bytes)) as raw_image:
            raw_image.load()
            icc_profile = raw_image.info.get("icc_profile")
            image = correct_exif_orientation(raw_image).convert("RGBA")
            return image, icc_profile, None
    except (UnidentifiedImageError, OSError, ValueError):
        return None, None, f"{filename} skipped. Invalid or corrupt image."


def process_single_image(
    index: int,
    filename: str,
    image_bytes: bytes,
    output_name: str,
    frame_size: Tuple[int, int],
    frame_bytes: bytes,
    photo_area_bbox: Tuple[int, int, int, int],
    output_format: str,
    jpg_quality: int,
    mime_type: str,
    make_preview: bool,
) -> Tuple[int, Optional[Dict[str, object]], List[str]]:
    messages: List[str] = []

    image, icc_profile, error = open_student_image_from_bytes(filename, image_bytes)
    if error:
        return index, None, [error]
    if image is None:
        return index, None, [f"{filename} skipped. Could not open image."]

    area_w = photo_area_bbox[2] - photo_area_bbox[0]
    area_h = photo_area_bbox[3] - photo_area_bbox[1]

    contain_scale = required_contain_scale(image.size, (area_w, area_h))
    if contain_scale > 1.0:
        messages.append(
            f"{filename}: small image enlarged from "
            f"{image.width}x{image.height}px to fit opening {area_w}x{area_h}px."
        )

    # Decode the selected frame only once per worker thread.
    frame = get_thread_cached_frame(frame_bytes)

    student_background = place_student_inside_frame_area(
        student_image=image,
        frame_size=frame_size,
        photo_area_bbox=photo_area_bbox,
    )
    final_image = apply_frame(student_background, frame)

    file_bytes = save_output_image_to_bytes(
        image=final_image,
        output_format=output_format,
        jpg_quality=jpg_quality,
        icc_profile=icc_profile,
    )

    preview_bytes = make_preview_bytes(final_image) if make_preview else None

    result = {
        "filename": output_name,
        "preview_bytes": preview_bytes,
        "file_bytes": file_bytes,
        "mime": mime_type,
        "size": final_image.size,
    }

    return index, result, messages


def process_uploaded_files_fast(
    uploaded_files,
    frame_bytes: bytes,
    photo_area_bbox: Tuple[int, int, int, int],
    selected_frame_path: Path,
    output_format: str,
    jpg_quality: int,
    preview_limit: int,
) -> Tuple[List[Dict[str, object]], List[str], bytes]:
    used_names = set()
    mime_type = get_mime_type(output_format)
    frame_size = png_bytes_to_image(frame_bytes).size

    payloads = []
    for index, uploaded_file in enumerate(uploaded_files):
        file_bytes = uploaded_file.getvalue()
        output_name = make_unique_output_name(
            original_filename=uploaded_file.name,
            frame_filename=selected_frame_path.name,
            output_format=output_format,
            used_names=used_names,
        )
        payloads.append(
            {
                "index": index,
                "filename": uploaded_file.name,
                "image_bytes": file_bytes,
                "output_name": output_name,
                "make_preview": index < preview_limit,
            }
        )

    if not payloads:
        return [], [], b""

    results_by_index: Dict[int, Dict[str, object]] = {}
    all_messages: List[str] = []

    progress = st.progress(0)
    total = len(payloads)

    # Render often has limited CPU/RAM. A small thread count is faster and safer than
    # launching too many workers for large PIL images.
    cpu_count = os.cpu_count() or 2
    max_workers = min(total, max(2, cpu_count), 6)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                process_single_image,
                payload["index"],
                payload["filename"],
                payload["image_bytes"],
                payload["output_name"],
                frame_size,
                frame_bytes,
                photo_area_bbox,
                output_format,
                jpg_quality,
                mime_type,
                payload["make_preview"],
            ): payload["index"]
            for payload in payloads
        }

        completed = 0
        for future in as_completed(future_map):
            completed += 1
            try:
                index, result, messages = future.result()
            except Exception as exc:
                index = future_map[future]
                result = None
                messages = [f"Image {index + 1} skipped because of an error: {exc}"]

            if result is not None:
                results_by_index[index] = result
            all_messages.extend(messages)
            progress.progress(completed / total)

    progress.empty()

    results = [results_by_index[i] for i in sorted(results_by_index.keys())]
    zip_bytes = create_zip_from_results(results)
    return results, all_messages, zip_bytes


# -----------------------------
# Streamlit UI
# -----------------------------

def show_gallery(results: List[Dict[str, object]], preview_limit: int) -> None:
    if not results or preview_limit <= 0:
        return

    preview_results = [r for r in results[:preview_limit] if r.get("preview_bytes")]
    if not preview_results:
        return

    st.subheader("Preview")
    if len(results) > len(preview_results):
        st.caption(
            f"Showing first {len(preview_results)} preview(s) for speed. "
            "The ZIP contains all generated images."
        )

    columns_per_row = 4
    for start in range(0, len(preview_results), columns_per_row):
        cols = st.columns(columns_per_row)
        for idx, (col, result) in enumerate(zip(cols, preview_results[start:start + columns_per_row])):
            global_index = start + idx
            with col:
                st.image(
                    result["preview_bytes"],
                    caption=result["filename"],
                    use_container_width=True,
                )
                st.download_button(
                    label="Download",
                    data=result["file_bytes"],
                    file_name=result["filename"],
                    mime=result["mime"],
                    key=f"download_single_{global_index}_{result['filename']}",
                    use_container_width=True,
                )


def main() -> None:
    st.set_page_config(
        page_title="Graduation Frame App",
        page_icon="🎓",
        layout="wide",
    )

    st.title("🎓 Graduation Frame App")

    with st.sidebar:
        output_format = st.radio("Format", ["PNG", "JPG"], index=0)

        jpg_quality = 95
        if output_format == "JPG":
            jpg_quality = st.slider("JPG Quality", 90, 100, 95)
            st.caption("JPG is usually much faster and smaller for photo outputs.")

        preview_limit = st.slider(
            "Preview limit",
            min_value=0,
            max_value=100,
            value=DEFAULT_PREVIEW_LIMIT,
            help="Lower preview count makes large batches much faster on Render. ZIP still includes every image.",
        )

    frames, frame_error = load_frames(FRAMES_DIR)
    if frame_error:
        st.error(frame_error)
        st.stop()

    frame_names = [frame.name for frame in frames]
    selected_frame_name = st.selectbox("Frame", frame_names)
    selected_frame_path = FRAMES_DIR / selected_frame_name

    stat = selected_frame_path.stat()
    frame_bytes, photo_area_bbox, frame_error, frame_was_fixed = prepare_frame_cached(
        str(selected_frame_path),
        stat.st_mtime,
        stat.st_size,
    )

    if frame_bytes is None or photo_area_bbox is None:
        st.error(frame_error or "Could not load frame.")
        st.stop()

    col1, col2 = st.columns([1, 2])

    with col1:
        frame_preview_bytes = make_frame_preview_cached(frame_bytes)
        st.image(frame_preview_bytes, caption=selected_frame_name, use_container_width=True)
        if frame_was_fixed:
            st.success("Frame opening detected")

    with col2:
        uploaded_files = st.file_uploader(
            "Student photos",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.write(f"{len(uploaded_files)} image(s) selected")

        generate = st.button(
            "Generate",
            type="primary",
            disabled=not uploaded_files,
            use_container_width=True,
        )

    if generate:
        with st.spinner("Generating images..."):
            results, messages, zip_bytes = process_uploaded_files_fast(
                uploaded_files=uploaded_files,
                frame_bytes=frame_bytes,
                photo_area_bbox=photo_area_bbox,
                selected_frame_path=selected_frame_path,
                output_format=output_format,
                jpg_quality=jpg_quality,
                preview_limit=preview_limit,
            )

        st.session_state["results"] = results
        st.session_state["messages"] = messages
        st.session_state["zip_bytes"] = zip_bytes
        st.session_state["zip_filename"] = "graduation_frames.zip"
        st.session_state["preview_limit"] = preview_limit

        if results:
            st.success(f"Generated {len(results)} image(s)")
        else:
            st.error("No images generated")

    if st.session_state.get("messages"):
        with st.expander("Warnings"):
            for message in st.session_state["messages"]:
                st.warning(message)

    if st.session_state.get("results"):
        st.download_button(
            "Download ZIP",
            data=st.session_state["zip_bytes"],
            file_name=st.session_state["zip_filename"],
            mime="application/zip",
            use_container_width=True,
        )
        show_gallery(st.session_state["results"], st.session_state.get("preview_limit", DEFAULT_PREVIEW_LIMIT))


if __name__ == "__main__":
    main()
