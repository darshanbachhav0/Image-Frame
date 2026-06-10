import re
import zipfile
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageChops, ImageOps, ImageStat, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent
FRAMES_DIR = BASE_DIR / "frames"
SUPPORTED_UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp"]

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS
    RESAMPLE_BICUBIC = Image.BICUBIC


def sanitize_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^\w.\- ]+", "_", name, flags=re.UNICODE).strip()
    name = re.sub(r"\s+", " ", name)
    return name or "image"


def load_frames(frames_dir: Path = FRAMES_DIR) -> Tuple[List[Path], Optional[str]]:
    if not frames_dir.exists():
        return [], "frames/ folder missing. Add your PNG frames inside the frames folder."

    if not frames_dir.is_dir():
        return [], "frames exists, but it is not a folder."

    frames = sorted(frames_dir.glob("*.png"))

    if not frames:
        return [], "No PNG frames found in frames/."

    return frames, None


def correct_exif_orientation(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image)


def image_has_transparency(image: Image.Image) -> bool:
    img = image.convert("RGBA")
    alpha_min, alpha_max = img.getchannel("A").getextrema()
    return alpha_min < 255 and alpha_max > 0


def is_light_neutral_pixel(r: int, g: int, b: int, tolerance: int) -> bool:
    max_channel = max(r, g, b)
    min_channel = min(r, g, b)
    return max_channel >= 145 and max_channel - min_channel <= tolerance


def is_close_color(
    r: int,
    g: int,
    b: int,
    target: Tuple[int, int, int],
    tolerance: int,
) -> bool:
    tr, tg, tb = target
    return (
        abs(r - tr) <= tolerance
        and abs(g - tg) <= tolerance
        and abs(b - tb) <= tolerance
    )


def detect_fake_center_background(frame: Image.Image) -> bool:
    img = frame.convert("RGBA")
    width, height = img.size

    crop_w = max(20, width // 5)
    crop_h = max(20, height // 5)

    left = (width - crop_w) // 2
    top = (height - crop_h) // 2

    center_crop = img.crop((left, top, left + crop_w, top + crop_h))

    alpha = center_crop.getchannel("A")
    alpha_min, _ = alpha.getextrema()

    if alpha_min < 80:
        return False

    rgb_crop = center_crop.convert("RGB")
    stat = ImageStat.Stat(rgb_crop)

    mean_r, mean_g, mean_b = stat.mean
    std_r, std_g, std_b = stat.stddev

    mean_max = max(mean_r, mean_g, mean_b)
    mean_min = min(mean_r, mean_g, mean_b)

    is_light = mean_max >= 150
    is_neutral = mean_max - mean_min <= 50
    has_variation = max(std_r, std_g, std_b) >= 3

    return is_light and is_neutral and has_variation


def auto_make_checkerboard_transparent(frame: Image.Image) -> Tuple[Image.Image, bool]:
    img = frame.convert("RGBA")

    if image_has_transparency(img) and not detect_fake_center_background(img):
        return img, False

    width, height = img.size
    pixels = img.load()
    tolerance = 55

    seed_points = [
        (width // 2, height // 2),
        (width // 2, height // 3),
        (width // 2, (height * 2) // 3),
        (width // 3, height // 2),
        ((width * 2) // 3, height // 2),
    ]

    seed_colors: List[Tuple[int, int, int]] = []

    for sx, sy in seed_points:
        r, g, b, a = pixels[sx, sy]
        if a > 0 and is_light_neutral_pixel(r, g, b, tolerance):
            seed_colors.append((r, g, b))

    if not seed_colors:
        return img, False

    def is_background_pixel(x: int, y: int) -> bool:
        r, g, b, a = pixels[x, y]

        if a == 0:
            return True

        if not is_light_neutral_pixel(r, g, b, tolerance):
            return False

        return any(is_close_color(r, g, b, color, tolerance) for color in seed_colors)

    visited = bytearray(width * height)
    queue = deque()

    for sx, sy in seed_points:
        idx = sy * width + sx
        visited[idx] = 1
        queue.append((sx, sy))

    changed_pixels = 0

    while queue:
        x, y = queue.popleft()

        if not is_background_pixel(x, y):
            continue

        r, g, b, _ = pixels[x, y]
        pixels[x, y] = (r, g, b, 0)
        changed_pixels += 1

        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < width and 0 <= ny < height:
                idx = ny * width + nx
                if not visited[idx]:
                    visited[idx] = 1
                    queue.append((nx, ny))

    return img, (changed_pixels / float(width * height)) > 0.005


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
            if not visited[idx]:
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

        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

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
    min_area_ratio: float = 0.02,
) -> Tuple[Optional[Image.Image], Optional[Tuple[int, int, int, int]]]:
    mask = mask.convert("L")
    width, height = mask.size

    region, bbox, count = flood_region_from_center_seeds(mask)

    if region is not None and bbox is not None:
        if count >= width * height * min_area_ratio:
            return region, bbox

    pixels = mask.load()
    visited = bytearray(width * height)

    best_score = 0.0
    best_bbox = None
    best_region = None

    center_x = width / 2
    center_y = height / 2

    for y in range(height):
        for x in range(width):
            idx = y * width + x

            if visited[idx]:
                continue

            visited[idx] = 1

            if pixels[x, y] == 0:
                continue

            queue = deque([(x, y)])
            points: List[int] = []

            min_x = max_x = x
            min_y = max_y = y
            touches_border = False

            while queue:
                cx, cy = queue.popleft()

                if pixels[cx, cy] == 0:
                    continue

                pidx = cy * width + cx
                points.append(pidx)

                if cx == 0 or cy == 0 or cx == width - 1 or cy == height - 1:
                    touches_border = True

                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height:
                        nidx = ny * width + nx
                        if not visited[nidx]:
                            visited[nidx] = 1
                            if pixels[nx, ny] > 0:
                                queue.append((nx, ny))

            count = len(points)

            if count < width * height * min_area_ratio:
                continue

            bbox = (min_x, min_y, max_x + 1, max_y + 1)
            box_w = bbox[2] - bbox[0]
            box_h = bbox[3] - bbox[1]

            if box_w < width * 0.10 or box_h < height * 0.10:
                continue

            box_center_x = (bbox[0] + bbox[2]) / 2
            box_center_y = (bbox[1] + bbox[3]) / 2
            center_distance = abs(box_center_x - center_x) + abs(box_center_y - center_y)

            score = float(count)

            if touches_border:
                score *= 0.15

            score *= max(0.25, 1.0 - (center_distance / (width + height)))

            if score > best_score:
                component = Image.new("L", (width, height), 0)
                component_pixels = component.load()

                for pidx in points:
                    py, px = divmod(pidx, width)
                    component_pixels[px, py] = 255

                best_score = score
                best_bbox = bbox
                best_region = component

    return best_region, best_bbox


def create_transparent_opening_mask(frame: Image.Image) -> Image.Image:
    alpha = frame.convert("RGBA").getchannel("A")
    return alpha.point(lambda a: 255 if a <= 35 else 0)


def create_dark_opening_mask(frame: Image.Image) -> Image.Image:
    img = frame.convert("RGBA")
    width, height = img.size
    pixels = img.load()

    mask = Image.new("L", (width, height), 0)
    mask_pixels = mask.load()

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]

            if a < 20:
                continue

            lum = int(0.299 * r + 0.587 * g + 0.114 * b)

            if lum <= 60 and max(r, g, b) - min(r, g, b) <= 45:
                mask_pixels[x, y] = 255

    return mask


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


def apply_opening_mask_to_frame(
    frame: Image.Image,
    opening_mask: Image.Image,
) -> Image.Image:
    img = frame.convert("RGBA")
    alpha = img.getchannel("A")
    mask = opening_mask.convert("L")

    new_alpha = ImageChops.subtract(alpha, mask)
    img.putalpha(new_alpha)

    return img


def fallback_photo_area(frame_size: Tuple[int, int]) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    width, height = frame_size

    left = int(width * 0.12)
    top = int(height * 0.16)
    right = int(width * 0.88)
    bottom = int(height * 0.84)

    mask = Image.new("L", (width, height), 0)
    mask.paste(255, (left, top, right, bottom))

    return mask, (left, top, right, bottom)


def load_and_prepare_frame(
    frame_path: Path,
) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[Tuple[int, int, int, int]], Optional[str], bool]:
    try:
        with Image.open(frame_path) as raw_frame:
            raw_frame.load()
            frame = raw_frame.convert("RGBA")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return None, None, None, f"Frame error: {exc}", False

    frame, checker_fixed = auto_make_checkerboard_transparent(frame)

    transparent_mask = create_transparent_opening_mask(frame)
    opening_mask, bbox = find_best_opening_region(transparent_mask)

    used_dark_placeholder = False

    if opening_mask is None or bbox is None:
        dark_mask = create_dark_opening_mask(frame)
        opening_mask, bbox = find_best_opening_region(dark_mask)
        used_dark_placeholder = opening_mask is not None and bbox is not None

    if opening_mask is None or bbox is None:
        opening_mask, bbox = fallback_photo_area(frame.size)

    bbox = shrink_bbox(bbox, 0.006, frame.size)
    frame = apply_opening_mask_to_frame(frame, opening_mask)

    was_fixed = checker_fixed or used_dark_placeholder

    return frame, opening_mask, bbox, None, was_fixed


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


def auto_trim_photo_margins(image: Image.Image) -> Image.Image:
    img = image.convert("RGBA")
    width, height = img.size

    alpha = img.getchannel("A")
    alpha_bbox = alpha.point(lambda a: 255 if a > 15 else 0).getbbox()

    if alpha_bbox:
        alpha_area = (alpha_bbox[2] - alpha_bbox[0]) * (alpha_bbox[3] - alpha_bbox[1])
        if alpha_area < width * height * 0.95:
            return crop_with_padding(img, alpha_bbox, 0.10, 0.10, 0.08)

    rgb = img.convert("RGB")
    bg_color = estimate_photo_background_color(rgb)

    bg = Image.new("RGB", rgb.size, bg_color)
    diff = ImageChops.difference(rgb, bg)
    gray = diff.convert("L")

    mask = gray.point(lambda p: 255 if p > 22 else 0)
    bbox = mask.getbbox()

    if bbox is None:
        return img

    left, top, right, bottom = bbox
    crop_w = right - left
    crop_h = bottom - top

    if crop_w < width * 0.25 or crop_h < height * 0.25:
        return img

    if crop_w > width * 0.96 and crop_h > height * 0.96:
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

    if extra_h > 0:
        top = int(extra_h * 0.42)
    else:
        top = 0

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


def smart_fit_photo_to_area(
    image: Image.Image,
    area_size: Tuple[int, int],
) -> Image.Image:
    original_bg = estimate_photo_background_color(image)
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

    if student_background.size != frame.size:
        frame = frame.resize(student_background.size, RESAMPLE_LANCZOS)

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
        rgb_image.save(
            buffer,
            format="JPEG",
            quality=int(jpg_quality),
            subsampling=0,
            optimize=True,
            **save_kwargs,
        )

    buffer.seek(0)
    return buffer.getvalue()


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


def open_student_image(uploaded_file) -> Tuple[Optional[Image.Image], Optional[bytes], Optional[str]]:
    try:
        uploaded_file.seek(0)

        with Image.open(uploaded_file) as raw_image:
            raw_image.load()
            icc_profile = raw_image.info.get("icc_profile")
            image = correct_exif_orientation(raw_image).convert("RGBA")
            return image, icc_profile, None

    except (UnidentifiedImageError, OSError, ValueError):
        return None, None, f"{uploaded_file.name} skipped. Invalid or corrupt image."


def make_preview_bytes(image: Image.Image, max_size: Tuple[int, int] = (650, 650)) -> bytes:
    preview = image.copy()
    preview.thumbnail(max_size, RESAMPLE_LANCZOS)

    buffer = BytesIO()
    preview.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)

    return buffer.getvalue()


def create_zip_from_results(results: List[Dict[str, object]]) -> bytes:
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for result in results:
            zip_file.writestr(result["filename"], result["file_bytes"])

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def get_mime_type(output_format: str) -> str:
    if output_format.upper() == "JPG":
        return "image/jpeg"
    return "image/png"


def process_uploaded_files(
    uploaded_files,
    frame: Image.Image,
    photo_area_bbox: Tuple[int, int, int, int],
    selected_frame_path: Path,
    output_format: str,
    jpg_quality: int,
) -> Tuple[List[Dict[str, object]], List[str], bytes]:
    results: List[Dict[str, object]] = []
    messages: List[str] = []

    frame_size = frame.size
    used_names = set()
    mime_type = get_mime_type(output_format)

    progress = st.progress(0)

    area_w = photo_area_bbox[2] - photo_area_bbox[0]
    area_h = photo_area_bbox[3] - photo_area_bbox[1]

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        image, icc_profile, error = open_student_image(uploaded_file)

        if error:
            messages.append(error)
            progress.progress(index / len(uploaded_files))
            continue

        if image is None:
            progress.progress(index / len(uploaded_files))
            continue

        contain_scale = required_contain_scale(image.size, (area_w, area_h))

        if contain_scale > 1.0:
            messages.append(
                f"{uploaded_file.name}: small image enlarged from "
                f"{image.width}x{image.height}px to fit opening {area_w}x{area_h}px."
            )

        student_background = place_student_inside_frame_area(
            student_image=image,
            frame_size=frame_size,
            photo_area_bbox=photo_area_bbox,
        )

        final_image = apply_frame(student_background, frame)

        output_name = make_unique_output_name(
            original_filename=uploaded_file.name,
            frame_filename=selected_frame_path.name,
            output_format=output_format,
            used_names=used_names,
        )

        file_bytes = save_output_image_to_bytes(
            image=final_image,
            output_format=output_format,
            jpg_quality=jpg_quality,
            icc_profile=icc_profile,
        )

        results.append(
            {
                "filename": output_name,
                "preview_bytes": make_preview_bytes(final_image),
                "file_bytes": file_bytes,
                "mime": mime_type,
                "size": final_image.size,
            }
        )

        progress.progress(index / len(uploaded_files))

    progress.empty()

    zip_bytes = create_zip_from_results(results)

    return results, messages, zip_bytes


def show_gallery(results: List[Dict[str, object]]) -> None:
    if not results:
        return

    st.subheader("Preview")

    columns_per_row = 4

    for start in range(0, len(results), columns_per_row):
        cols = st.columns(columns_per_row)

        for idx, (col, result) in enumerate(zip(cols, results[start:start + columns_per_row])):
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

    frames, frame_error = load_frames(FRAMES_DIR)

    if frame_error:
        st.error(frame_error)
        st.stop()

    frame_names = [frame.name for frame in frames]

    selected_frame_name = st.selectbox("Frame", frame_names)
    selected_frame_path = FRAMES_DIR / selected_frame_name

    frame_image, opening_mask, photo_area_bbox, frame_error, frame_was_fixed = load_and_prepare_frame(
        selected_frame_path
    )

    if frame_image is None or opening_mask is None or photo_area_bbox is None:
        st.error(frame_error or "Could not load frame.")
        st.stop()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(frame_image, caption=selected_frame_name, use_container_width=True)

    with col2:
        uploaded_files = st.file_uploader(
            "Sube tu fotos",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.write(f"{len(uploaded_files)} image(s) selected")

        if frame_was_fixed:
            st.success("Frame opening detected and fixed")

        generate = st.button(
            "Generate",
            type="primary",
            disabled=not uploaded_files,
            use_container_width=True,
        )

    if generate:
        results, messages, zip_bytes = process_uploaded_files(
            uploaded_files=uploaded_files,
            frame=frame_image,
            photo_area_bbox=photo_area_bbox,
            selected_frame_path=selected_frame_path,
            output_format=output_format,
            jpg_quality=jpg_quality,
        )

        st.session_state["results"] = results
        st.session_state["messages"] = messages
        st.session_state["zip_bytes"] = zip_bytes
        st.session_state["zip_filename"] = "graduation_frames.zip"

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

        show_gallery(st.session_state["results"])


if __name__ == "__main__":
    main()