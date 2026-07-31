



























import gc
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
DEFAULT_PREVIEW_LIMIT = 12
MAX_PREVIEW_LIMIT = 24
MAX_UPLOAD_FILES = 100
DEFAULT_IMAGES_PER_ZIP = 10
MAX_IMAGES_PER_ZIP = 25
MAX_SOURCE_PIXELS = max(20_000_000, int(os.getenv("MAX_SOURCE_PIXELS", "80000000")))
SOURCE_OVERSAMPLE = max(1.0, float(os.getenv("SOURCE_OVERSAMPLE", "2.0")))
BATCH_RETENTION_SECONDS = max(3600, int(os.getenv("BATCH_RETENTION_SECONDS", "21600")))
UPLOAD_COPY_CHUNK_SIZE = 1024 * 1024
TEMP_ROOT = Path(os.getenv("APP_TEMP_DIR", tempfile.gettempdir())) / "graduation_frame_app"
PROCESSING_LOCK = threading.Lock()

# Pillow will still warn for very large images, while the app performs its own
# explicit size validation before decoding them.
Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS

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
            # Keep the frame at its exact original resolution. The final image
            # dimensions are therefore identical to the selected frame dimensions.
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
    img = image if image.mode == "RGBA" else image.convert("RGBA")
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

    cropped = resized.crop((left, top, left + target_w, top + target_h))
    resized.close()
    return cropped


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
    resized_rgba = resized if resized.mode == "RGBA" else resized.convert("RGBA")
    area_canvas.alpha_composite(resized_rgba, (paste_x, paste_y))
    if resized_rgba is not resized:
        resized_rgba.close()
    resized.close()

    return area_canvas


def smart_fit_photo_to_area(image: Image.Image, area_size: Tuple[int, int]) -> Image.Image:
    # Estimate background before trim, as in the original behavior.
    detection_for_bg, _, _ = resize_for_detection(image, TRIM_DETECTION_MAX_SIZE)
    try:
        original_bg = estimate_photo_background_color(detection_for_bg)
    finally:
        detection_for_bg.close()

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
        converted = student_background.convert("RGBA")
        student_background.close()
        student_background = converted
    if frame.mode != "RGBA":
        frame = frame.convert("RGBA")

    # Composite in-place to avoid allocating another full-frame RGBA image.
    student_background.alpha_composite(frame)
    return student_background


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
# Saving, previews, temporary batches, ZIP parts
# -----------------------------

def ensure_temp_root() -> Path:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return TEMP_ROOT


def is_safe_batch_path(path: Path) -> bool:
    try:
        root = ensure_temp_root().resolve()
        candidate = path.resolve()
    except OSError:
        return False
    return candidate != root and root in candidate.parents


def remove_batch_directory(path_value: object) -> None:
    if not path_value:
        return
    try:
        path = Path(str(path_value))
        if is_safe_batch_path(path) and path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except (OSError, ValueError):
        pass


def cleanup_old_batches() -> None:
    root = ensure_temp_root()
    cutoff = time.time() - BATCH_RETENTION_SECONDS
    try:
        children = list(root.iterdir())
    except OSError:
        return

    for child in children:
        if not child.is_dir():
            continue
        try:
            if child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


def clear_completed_batch() -> None:
    completed = st.session_state.pop("completed_batch", None)
    if isinstance(completed, dict):
        remove_batch_directory(completed.get("batch_dir"))

    for key in (
        "batch_error",
        "download_part_index",
        "generation_notice",
    ):
        st.session_state.pop(key, None)
    gc.collect()


def clear_pending_batch() -> None:
    pending = st.session_state.pop("pending_batch", None)
    if isinstance(pending, dict):
        remove_batch_directory(pending.get("batch_dir"))
    gc.collect()


def safe_staged_filename(index: int, original_name: str, used_names: set) -> str:
    cleaned = sanitize_filename(Path(original_name).name)
    candidate = f"{index + 1:03d}_{cleaned}"
    stem = Path(candidate).stem
    suffix = Path(candidate).suffix
    counter = 1

    while candidate.lower() in used_names:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1

    used_names.add(candidate.lower())
    return candidate


def stage_uploaded_files(uploaded_files) -> Tuple[Path, List[Dict[str, object]], int]:
    """Copy uploaded files to temporary disk, then let Streamlit clear the uploader."""
    root = ensure_temp_root()
    batch_id = f"batch_{int(time.time())}_{uuid.uuid4().hex[:10]}"
    batch_dir = root / batch_id
    input_dir = batch_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=False)

    manifest: List[Dict[str, object]] = []
    total_bytes = 0
    used_names = set()

    try:
        for index, uploaded_file in enumerate(uploaded_files):
            original_name = Path(uploaded_file.name).name
            stored_name = safe_staged_filename(index, original_name, used_names)
            destination = input_dir / stored_name

            uploaded_file.seek(0)
            with destination.open("wb") as output_file:
                while True:
                    chunk = uploaded_file.read(UPLOAD_COPY_CHUNK_SIZE)
                    if not chunk:
                        break
                    output_file.write(chunk)
                    total_bytes += len(chunk)

            manifest.append(
                {
                    "original_name": original_name,
                    "path": str(destination),
                    "size": destination.stat().st_size,
                }
            )
    except Exception:
        shutil.rmtree(batch_dir, ignore_errors=True)
        raise

    return batch_dir, manifest, total_bytes


def rgba_to_rgb_for_saving(
    image: Image.Image,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGB")

    background = Image.new("RGB", image.size, background_color)
    background.paste(image, mask=image.getchannel("A"))
    return background


def save_output_image_to_path(
    image: Image.Image,
    output_path: Path,
    output_format: str,
    jpg_quality: int,
    icc_profile: Optional[bytes],
) -> None:
    """Save at full frame resolution; PNG is lossless and JPG uses 4:4:4 chroma."""

    def save_once(include_icc: bool) -> None:
        save_kwargs: Dict[str, object] = {}
        if include_icc and icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        if output_format.upper() == "PNG":
            image.save(
                output_path,
                format="PNG",
                compress_level=1,
                optimize=False,
                **save_kwargs,
            )
            return

        rgb_image = rgba_to_rgb_for_saving(image)
        try:
            rgb_image.save(
                output_path,
                format="JPEG",
                quality=max(90, min(int(jpg_quality), 100)),
                subsampling=0,
                optimize=False,
                progressive=False,
                **save_kwargs,
            )
        finally:
            rgb_image.close()

    try:
        save_once(include_icc=True)
    except (OSError, ValueError):
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass
        save_once(include_icc=False)


def make_preview_bytes(
    image: Image.Image,
    max_size: Tuple[int, int] = PREVIEW_MAX_SIZE,
    quality: int = 84,
) -> bytes:
    preview = image.copy()
    try:
        preview.thumbnail(max_size, RESAMPLE_LANCZOS)

        if preview.mode == "RGBA":
            background = Image.new("RGB", preview.size, (255, 255, 255))
            background.paste(preview, mask=preview.getchannel("A"))
            preview.close()
            preview = background
        else:
            converted = preview.convert("RGB")
            preview.close()
            preview = converted

        buffer = BytesIO()
        preview.save(buffer, format="JPEG", quality=quality, optimize=False)
        return buffer.getvalue()
    finally:
        preview.close()


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


def useful_decode_size(frame_size: Tuple[int, int]) -> Tuple[int, int]:
    """Keep extra source detail before the final high-quality Lanczos resize."""
    return (
        max(1, int(round(frame_size[0] * SOURCE_OVERSAMPLE))),
        max(1, int(round(frame_size[1] * SOURCE_OVERSAMPLE))),
    )


def open_student_image_from_path(
    filename: str,
    image_path: Path,
    frame_size: Tuple[int, int],
) -> Tuple[Optional[Image.Image], Optional[bytes], Optional[str]]:
    try:
        with Image.open(image_path) as raw_image:
            width, height = raw_image.size
            if width <= 0 or height <= 0:
                return None, None, f"{filename} skipped. Invalid image dimensions."

            pixel_count = width * height
            if pixel_count > MAX_SOURCE_PIXELS:
                return (
                    None,
                    None,
                    f"{filename} skipped. Image is {width}x{height}px "
                    f"({pixel_count:,} pixels); maximum is {MAX_SOURCE_PIXELS:,} pixels.",
                )

            icc_profile = raw_image.info.get("icc_profile")
            decode_size = useful_decode_size(frame_size)

            # JPEG draft decoding is used only when the source is much larger than
            # the final frame. At least SOURCE_OVERSAMPLE times the output dimensions
            # are retained before the final Lanczos resize.
            if (
                (raw_image.format or "").upper() in {"JPEG", "JPG"}
                and width > decode_size[0] * 2
                and height > decode_size[1] * 2
            ):
                try:
                    raw_image.draft("RGB", decode_size)
                except (OSError, ValueError):
                    pass

            oriented = correct_exif_orientation(raw_image)
            try:
                oriented.load()
                oriented.thumbnail(decode_size, RESAMPLE_LANCZOS)
                image = oriented.convert("RGBA")
            finally:
                if oriented is not raw_image:
                    oriented.close()

            return image, icc_profile, None
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ):
        return None, None, f"{filename} skipped. Invalid, corrupt, or oversized image."


def process_single_image_to_file(
    filename: str,
    image_path: Path,
    output_path: Path,
    output_name: str,
    frame_size: Tuple[int, int],
    frame: Image.Image,
    photo_area_bbox: Tuple[int, int, int, int],
    output_format: str,
    jpg_quality: int,
    make_preview: bool,
) -> Tuple[Optional[Dict[str, object]], List[str]]:
    messages: List[str] = []
    image: Optional[Image.Image] = None
    final_image: Optional[Image.Image] = None

    try:
        image, icc_profile, error = open_student_image_from_path(
            filename=filename,
            image_path=image_path,
            frame_size=frame_size,
        )
        if error:
            return None, [error]
        if image is None:
            return None, [f"{filename} skipped. Could not open image."]

        area_w = photo_area_bbox[2] - photo_area_bbox[0]
        area_h = photo_area_bbox[3] - photo_area_bbox[1]
        contain_scale = required_contain_scale(image.size, (area_w, area_h))
        if contain_scale > 1.0:
            messages.append(
                f"{filename}: small image enlarged from "
                f"{image.width}x{image.height}px to fit opening {area_w}x{area_h}px."
            )

        final_image = place_student_inside_frame_area(
            student_image=image,
            frame_size=frame_size,
            photo_area_bbox=photo_area_bbox,
        )
        final_image = apply_frame(final_image, frame)

        save_output_image_to_path(
            image=final_image,
            output_path=output_path,
            output_format=output_format,
            jpg_quality=jpg_quality,
            icc_profile=icc_profile,
        )

        preview_bytes = make_preview_bytes(final_image) if make_preview else None
        result = {
            "filename": output_name,
            "preview_bytes": preview_bytes,
            "size": final_image.size,
        }
        return result, messages
    finally:
        if final_image is not None:
            final_image.close()
        if image is not None:
            image.close()


def process_staged_batch(pending: Dict[str, Any]) -> Dict[str, object]:
    batch_dir = Path(str(pending["batch_dir"]))
    files = list(pending["files"])
    selected_frame_path = Path(str(pending["frame_path"]))
    output_format = str(pending["output_format"])
    jpg_quality = int(pending["jpg_quality"])
    preview_limit = int(pending["preview_limit"])
    images_per_zip = max(1, int(pending["images_per_zip"]))

    if not is_safe_batch_path(batch_dir) or not batch_dir.exists():
        raise RuntimeError("The temporary batch directory is missing.")

    stat = selected_frame_path.stat()
    frame_bytes, photo_area_bbox, frame_error, _ = prepare_frame_cached(
        str(selected_frame_path),
        stat.st_mtime,
        stat.st_size,
    )
    if frame_bytes is None or photo_area_bbox is None:
        raise RuntimeError(frame_error or "Could not load the selected frame.")

    frame = png_bytes_to_image(frame_bytes)
    frame_size = frame.size
    downloads_dir = batch_dir / "downloads"
    output_dir = batch_dir / "working"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    used_names = set()
    previews: List[Dict[str, object]] = []
    messages: List[str] = []
    zip_parts: List[Dict[str, object]] = []
    generated_count = 0
    total = len(files)
    progress = st.progress(0)
    status_text = st.empty()

    current_zip: Optional[zipfile.ZipFile] = None
    current_zip_path: Optional[Path] = None
    current_part_number = 0
    current_part_count = 0

    def close_current_zip() -> None:
        nonlocal current_zip, current_zip_path, current_part_count
        if current_zip is None or current_zip_path is None:
            return
        current_zip.close()
        zip_parts.append(
            {
                "part": current_part_number,
                "path": str(current_zip_path),
                "filename": current_zip_path.name,
                "count": current_part_count,
                "size": current_zip_path.stat().st_size,
            }
        )
        current_zip = None
        current_zip_path = None
        current_part_count = 0

    try:
        for index, item in enumerate(files):
            original_name = str(item["original_name"])
            source_path = Path(str(item["path"]))
            status_text.write(f"Processing {index + 1} of {total}: {original_name}")

            output_name = make_unique_output_name(
                original_filename=original_name,
                frame_filename=selected_frame_path.name,
                output_format=output_format,
                used_names=used_names,
            )
            temporary_output = output_dir / output_name

            try:
                result, image_messages = process_single_image_to_file(
                    filename=original_name,
                    image_path=source_path,
                    output_path=temporary_output,
                    output_name=output_name,
                    frame_size=frame_size,
                    frame=frame,
                    photo_area_bbox=photo_area_bbox,
                    output_format=output_format,
                    jpg_quality=jpg_quality,
                    make_preview=generated_count < preview_limit,
                )
            except Exception as exc:
                result = None
                image_messages = [f"{original_name} skipped because of an error: {exc}"]

            messages.extend(image_messages)

            if result is not None and temporary_output.exists():
                if current_zip is None or current_part_count >= images_per_zip:
                    close_current_zip()
                    current_part_number += 1
                    current_zip_path = downloads_dir / (
                        f"graduation_frames_part_{current_part_number:02d}.zip"
                    )
                    current_zip = zipfile.ZipFile(
                        current_zip_path,
                        mode="w",
                        compression=zipfile.ZIP_STORED,
                        allowZip64=True,
                    )

                current_zip.write(temporary_output, arcname=output_name)
                current_part_count += 1
                generated_count += 1

                if result.get("preview_bytes"):
                    previews.append(result)

            try:
                temporary_output.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                source_path.unlink(missing_ok=True)
            except OSError:
                pass

            progress.progress((index + 1) / total)
            if index % 2 == 1:
                gc.collect()

        close_current_zip()
    finally:
        if current_zip is not None:
            current_zip.close()
        frame.close()
        progress.empty()
        status_text.empty()
        shutil.rmtree(output_dir, ignore_errors=True)
        gc.collect()

    return {
        "batch_dir": str(batch_dir),
        "batch_id": batch_dir.name,
        "generated_count": generated_count,
        "requested_count": total,
        "messages": messages,
        "previews": previews,
        "preview_limit": preview_limit,
        "zip_parts": zip_parts,
        "output_format": output_format,
        "frame_size": frame_size,
    }


# -----------------------------
# Streamlit UI
# -----------------------------

def human_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def show_gallery(results: List[Dict[str, object]]) -> None:
    if not results:
        return

    st.subheader("Preview")
    columns_per_row = 4
    for start in range(0, len(results), columns_per_row):
        cols = st.columns(columns_per_row)
        for col, result in zip(cols, results[start:start + columns_per_row]):
            with col:
                st.image(
                    result["preview_bytes"],
                    caption=str(result["filename"]),
                    use_container_width=True,
                )


def run_pending_batch() -> None:
    pending = st.session_state.get("pending_batch")
    if not isinstance(pending, dict):
        return

    st.subheader("Generating images")
    st.caption(
        "The upload widget has been cleared to release memory. "
        "Photos are now processed one at a time at full frame resolution."
    )

    try:
        wait_notice = st.empty()
        if PROCESSING_LOCK.locked():
            wait_notice.info("Another batch is being processed. This batch will start next.")
        with PROCESSING_LOCK:
            wait_notice.empty()
            completed = process_staged_batch(pending)
    except Exception as exc:
        st.session_state["batch_error"] = f"Generation failed: {exc}"
        remove_batch_directory(pending.get("batch_dir"))
    else:
        st.session_state["completed_batch"] = completed
        st.session_state["generation_notice"] = (
            f"Generated {completed['generated_count']} of "
            f"{completed['requested_count']} image(s)."
        )
    finally:
        st.session_state.pop("pending_batch", None)
        gc.collect()

    st.rerun()


def show_completed_batch() -> None:
    completed = st.session_state.get("completed_batch")
    if not isinstance(completed, dict):
        return

    generated_count = int(completed.get("generated_count", 0))
    requested_count = int(completed.get("requested_count", 0))
    frame_size = completed.get("frame_size")

    if generated_count:
        st.success(f"Generated {generated_count} of {requested_count} image(s).")
        if isinstance(frame_size, (tuple, list)) and len(frame_size) == 2:
            st.caption(
                f"Every output is {frame_size[0]} x {frame_size[1]} pixels, "
                "the exact selected-frame resolution."
            )
    else:
        st.error("No images were generated.")

    messages = completed.get("messages") or []
    if messages:
        with st.expander(f"Warnings ({len(messages)})"):
            for message in messages:
                st.warning(str(message))

    zip_parts = completed.get("zip_parts") or []
    valid_parts = [part for part in zip_parts if Path(str(part.get("path", ""))).exists()]

    if valid_parts:
        st.subheader("Download")
        labels = [
            f"Part {part['part']} - {part['count']} image(s) - {human_size(int(part['size']))}"
            for part in valid_parts
        ]
        selected_label = st.selectbox(
            "Select a ZIP part",
            labels,
            key="download_part_index",
        )
        selected_index = labels.index(selected_label)
        selected_part = valid_parts[selected_index]
        selected_path = Path(str(selected_part["path"]))

        # Only the selected ZIP part is registered with Streamlit, preventing all
        # 100 outputs from being duplicated in server memory at once.
        with selected_path.open("rb") as zip_file:
            st.download_button(
                label=f"Download {selected_part['filename']}",
                data=zip_file,
                file_name=str(selected_part["filename"]),
                mime="application/zip",
                key=f"download_{completed.get('batch_id')}_{selected_part['part']}",
                use_container_width=True,
            )

        st.caption(
            "Select and download each ZIP part. Splitting changes only the packaging; "
            "it does not recompress or reduce the image quality."
        )
    elif generated_count:
        st.error("The generated ZIP files are no longer available. Generate the batch again.")

    show_gallery(list(completed.get("previews") or []))

    if st.button("Clear generated batch", use_container_width=True):
        clear_completed_batch()
        st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Graduation Frame App",
        page_icon="🎓",
        layout="wide",
    )

    cleanup_old_batches()
    st.title("🎓 Graduation Frame App")

    if st.session_state.get("pending_batch"):
        run_pending_batch()
        return

    if st.session_state.get("batch_error"):
        st.error(st.session_state.pop("batch_error"))

    if st.session_state.get("generation_notice"):
        st.info(st.session_state.pop("generation_notice"))

    with st.sidebar:
        output_format = st.radio("Format", ["PNG", "JPG"], index=1)

        jpg_quality = 95
        if output_format == "JPG":
            jpg_quality = st.slider("JPG Quality", 90, 100, 95)
            st.caption(
                "JPG uses full frame resolution and 4:4:4 chroma. "
                "Quality 100 creates much larger files."
            )
        else:
            st.caption("PNG output is lossless.")

        preview_limit = st.slider(
            "Preview limit",
            min_value=0,
            max_value=MAX_PREVIEW_LIMIT,
            value=DEFAULT_PREVIEW_LIMIT,
            help="This affects only the on-screen previews, never the generated files.",
        )

        images_per_zip = st.slider(
            "Images per ZIP part",
            min_value=5,
            max_value=MAX_IMAGES_PER_ZIP,
            value=DEFAULT_IMAGES_PER_ZIP,
            step=5,
            help=(
                "Smaller ZIP parts are safer on low-memory Render services. "
                "ZIP splitting does not change image quality."
            ),
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
        with Image.open(selected_frame_path) as raw_frame:
            st.caption(f"Output resolution: {raw_frame.width} x {raw_frame.height} pixels")
        if frame_was_fixed:
            st.success("Frame opening detected")

    with col2:
        uploader_nonce = int(st.session_state.get("uploader_nonce", 0))
        uploaded_files = st.file_uploader(
            "Student photos (maximum 100)",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"student_photos_{uploader_nonce}",
        )

        upload_error: Optional[str] = None
        if uploaded_files:
            total_upload_bytes = sum(
                int(getattr(uploaded_file, "size", 0) or 0)
                for uploaded_file in uploaded_files
            )
            st.write(
                f"{len(uploaded_files)} image(s) selected "
                f"({human_size(total_upload_bytes)} total)"
            )
            if len(uploaded_files) > MAX_UPLOAD_FILES:
                upload_error = f"Select at most {MAX_UPLOAD_FILES} photos in one batch."

        if upload_error:
            st.error(upload_error)

        generate = st.button(
            "Generate",
            type="primary",
            disabled=not uploaded_files or upload_error is not None,
            use_container_width=True,
        )

    if generate:
        clear_pending_batch()
        clear_completed_batch()

        try:
            with st.spinner("Copying uploads to temporary disk..."):
                batch_dir, manifest, total_bytes = stage_uploaded_files(uploaded_files)
        except Exception as exc:
            st.error(f"Could not stage the uploaded files: {exc}")
        else:
            st.session_state["pending_batch"] = {
                "batch_dir": str(batch_dir),
                "files": manifest,
                "total_upload_bytes": total_bytes,
                "frame_path": str(selected_frame_path),
                "output_format": output_format,
                "jpg_quality": jpg_quality,
                "preview_limit": preview_limit,
                "images_per_zip": images_per_zip,
            }

            # Changing the uploader key clears all UploadedFile objects before the
            # expensive image processing begins on the next Streamlit run.
            st.session_state["uploader_nonce"] = uploader_nonce + 1
            del uploaded_files
            gc.collect()
            st.rerun()

    show_completed_batch()


if __name__ == "__main__":
    main()