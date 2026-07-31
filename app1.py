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


# Allow Pillow to read slightly incomplete mobile images.
ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = Path(__file__).resolve().parent
FRAMES_DIR = BASE_DIR / "frames"
SUPPORTED_UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp"]

# Detection uses smaller copies. Final images retain the frame's full resolution.
DETECTION_MAX_SIZE = 850
TRIM_DETECTION_MAX_SIZE = 900
PREVIEW_MAX_SIZE = (420, 420)

DEFAULT_PREVIEW_LIMIT = 12
MAX_PREVIEW_LIMIT = 24
MAX_UPLOAD_FILES = 100

# Maximum source photo pixel count.
MAX_SOURCE_PIXELS = max(
    20_000_000,
    int(os.getenv("MAX_SOURCE_PIXELS", "80000000")),
)

# Retain enough source detail before the final high-quality resize.
SOURCE_OVERSAMPLE = max(
    1.0,
    float(os.getenv("SOURCE_OVERSAMPLE", "2.0")),
)

# Remove temporary batches after six hours by default.
BATCH_RETENTION_SECONDS = max(
    3600,
    int(os.getenv("BATCH_RETENTION_SECONDS", "21600")),
)

UPLOAD_COPY_CHUNK_SIZE = 1024 * 1024

TEMP_ROOT = (
    Path(os.getenv("APP_TEMP_DIR", tempfile.gettempdir()))
    / "graduation_frame_app"
)

# Prevent multiple large image batches from running simultaneously.
PROCESSING_LOCK = threading.Lock()

# The app also performs its own size validation before decoding.
Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    # Compatibility with older Pillow versions.
    RESAMPLE_LANCZOS = Image.LANCZOS
    RESAMPLE_NEAREST = Image.NEAREST


# =========================================================
# Small utilities
# =========================================================

def sanitize_filename(name: str) -> str:
    """Create a safe filename while preserving normal Unicode characters."""
    name = Path(name).name
    name = re.sub(r"[^\w.\- ]+", "_", name, flags=re.UNICODE).strip()
    name = re.sub(r"\s+", " ", name)
    return name or "image"


def load_frames(
    frames_dir: Path = FRAMES_DIR,
) -> Tuple[List[Path], Optional[str]]:
    """Return all PNG frames from the frames directory."""
    if not frames_dir.exists():
        return [], (
            "frames/ folder missing. "
            "Create a frames folder beside app.py and add PNG frames."
        )

    if not frames_dir.is_dir():
        return [], "frames exists, but it is not a folder."

    frames = sorted(frames_dir.glob("*.png"))

    if not frames:
        return [], "No PNG frames were found inside the frames folder."

    return frames, None


def correct_exif_orientation(image: Image.Image) -> Image.Image:
    """Rotate an image according to its EXIF orientation."""
    return ImageOps.exif_transpose(image)


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Convert a Pillow image to PNG bytes."""
    buffer = BytesIO()

    # Low compression level is faster and remains lossless.
    image.save(
        buffer,
        format="PNG",
        compress_level=1,
        optimize=False,
    )

    return buffer.getvalue()


def png_bytes_to_image(data: bytes) -> Image.Image:
    """Convert PNG bytes to a fully loaded RGBA image."""
    with Image.open(BytesIO(data)) as image:
        image.load()
        return image.convert("RGBA")


def resize_for_detection(
    image: Image.Image,
    max_size: int = DETECTION_MAX_SIZE,
) -> Tuple[Image.Image, float, float]:
    """
    Resize an image only for detection work.

    Returns:
        resized_image,
        scale_x_to_original,
        scale_y_to_original
    """
    width, height = image.size
    largest = max(width, height)

    if largest <= max_size:
        return image.copy(), 1.0, 1.0

    scale = max_size / largest

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = image.resize(
        (new_width, new_height),
        RESAMPLE_LANCZOS,
    )

    return (
        resized,
        width / new_width,
        height / new_height,
    )


def scale_bbox_to_original(
    bbox: Tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
    original_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Scale a bounding box from a detection image to the original size."""
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
    """Slightly shrink a detected opening to avoid visible frame-edge gaps."""
    left, top, right, bottom = bbox

    box_width = right - left
    box_height = bottom - top

    dx = int(box_width * shrink_percent)
    dy = int(box_height * shrink_percent)

    width, height = image_size

    left = max(0, left + dx)
    top = max(0, top + dy)
    right = min(width, right - dx)
    bottom = min(height, bottom - dy)

    if right <= left or bottom <= top:
        return bbox

    return left, top, right, bottom


# =========================================================
# Frame opening detection
# =========================================================

def create_transparent_opening_mask(frame: Image.Image) -> Image.Image:
    """Detect a transparent opening in a frame."""
    alpha = frame.convert("RGBA").getchannel("A")
    return alpha.point(lambda value: 255 if value <= 35 else 0)


def create_dark_opening_mask(frame: Image.Image) -> Image.Image:
    """Detect a dark/black opening in a frame."""
    rgba = frame.convert("RGBA")
    alpha = rgba.getchannel("A")
    luminance = rgba.convert("L")

    dark = luminance.point(
        lambda value: 255 if value <= 65 else 0
    )

    visible = alpha.point(
        lambda value: 255 if value > 20 else 0
    )

    return ImageChops.multiply(dark, visible)


def create_light_opening_mask(frame: Image.Image) -> Image.Image:
    """Detect a light or low-saturation opening in a frame."""
    rgba = frame.convert("RGBA")
    alpha = rgba.getchannel("A")

    hsv = rgba.convert("RGB").convert("HSV")
    _, saturation, value = hsv.split()

    low_saturation = saturation.point(
        lambda pixel: 255 if pixel <= 55 else 0
    )

    bright = value.point(
        lambda pixel: 255 if pixel >= 145 else 0
    )

    visible = alpha.point(
        lambda pixel: 255 if pixel > 20 else 0
    )

    return ImageChops.multiply(
        ImageChops.multiply(low_saturation, bright),
        visible,
    )


def flood_region_from_center_seeds(
    mask: Image.Image,
) -> Tuple[
    Optional[Image.Image],
    Optional[Tuple[int, int, int, int]],
    int,
]:
    """Find a connected opening region starting from central seed points."""
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

    for seed_x, seed_y in seed_points:
        if (
            0 <= seed_x < width
            and 0 <= seed_y < height
            and pixels[seed_x, seed_y] > 0
        ):
            index = seed_y * width + seed_x
            visited[index] = 1
            queue.append((seed_x, seed_y))

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

        neighbours = (
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        )

        for neighbour_x, neighbour_y in neighbours:
            if (
                0 <= neighbour_x < width
                and 0 <= neighbour_y < height
            ):
                index = neighbour_y * width + neighbour_x

                if not visited[index]:
                    visited[index] = 1

                    if pixels[neighbour_x, neighbour_y] > 0:
                        queue.append((neighbour_x, neighbour_y))

    if count == 0:
        return None, None, 0

    bbox = (
        min_x,
        min_y,
        max_x + 1,
        max_y + 1,
    )

    return region, bbox, count


def find_best_opening_region(
    mask: Image.Image,
    min_area_ratio: float = 0.015,
) -> Tuple[
    Optional[Image.Image],
    Optional[Tuple[int, int, int, int]],
]:
    """Return the detected center-connected frame opening."""
    mask = mask.convert("L")
    width, height = mask.size

    region, bbox, count = flood_region_from_center_seeds(mask)

    minimum_area = width * height * min_area_ratio

    if (
        region is not None
        and bbox is not None
        and count >= minimum_area
    ):
        return region, bbox

    return None, None


def fallback_photo_area(
    frame_size: Tuple[int, int],
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Fallback opening used when automatic detection cannot find one."""
    width, height = frame_size

    left = int(width * 0.12)
    top = int(height * 0.16)
    right = int(width * 0.88)
    bottom = int(height * 0.84)

    mask = Image.new("L", frame_size, 0)
    mask.paste(255, (left, top, right, bottom))

    return mask, (left, top, right, bottom)


def apply_opening_mask_to_frame(
    frame: Image.Image,
    opening_mask: Image.Image,
) -> Image.Image:
    """Make the detected opening transparent in the frame."""
    frame = frame.convert("RGBA")

    opening_mask = opening_mask.convert("L").resize(
        frame.size,
        RESAMPLE_NEAREST,
    )

    alpha = frame.getchannel("A")
    new_alpha = ImageChops.subtract(alpha, opening_mask)
    frame.putalpha(new_alpha)

    return frame


@st.cache_data(show_spinner=False)
def prepare_frame_cached(
    frame_path_string: str,
    frame_modified_time: float,
    frame_size_bytes: int,
) -> Tuple[
    Optional[bytes],
    Optional[Tuple[int, int, int, int]],
    Optional[str],
    bool,
]:
    """
    Load and prepare a frame.

    The modified time and file size are included to invalidate the cache
    when the frame changes.
    """
    _ = (frame_modified_time, frame_size_bytes)

    try:
        with Image.open(frame_path_string) as raw_frame:
            raw_frame.load()
            frame = raw_frame.convert("RGBA")
    except (UnidentifiedImageError, OSError, ValueError) as exception:
        return (
            None,
            None,
            f"Frame error: {exception}",
            False,
        )

    original_size = frame.size

    detection_frame, scale_x, scale_y = resize_for_detection(
        frame,
        DETECTION_MAX_SIZE,
    )

    masks = [
        create_transparent_opening_mask(detection_frame),
        create_dark_opening_mask(detection_frame),
        create_light_opening_mask(detection_frame),
    ]

    opening_mask_small = None
    bbox_small = None
    frame_was_fixed = False

    for index, mask in enumerate(masks):
        opening_mask_small, bbox_small = find_best_opening_region(mask)

        if (
            opening_mask_small is not None
            and bbox_small is not None
        ):
            frame_was_fixed = index != 0
            break

    detection_frame.close()

    if opening_mask_small is None or bbox_small is None:
        opening_mask, bbox = fallback_photo_area(original_size)
    else:
        opening_mask = opening_mask_small.resize(
            original_size,
            RESAMPLE_NEAREST,
        )

        bbox = scale_bbox_to_original(
            bbox_small,
            scale_x,
            scale_y,
            original_size,
        )

        opening_mask_small.close()

    bbox = shrink_bbox(
        bbox,
        0.006,
        original_size,
    )

    prepared_frame = apply_opening_mask_to_frame(
        frame,
        opening_mask,
    )

    opening_mask.close()
    frame.close()

    prepared_bytes = image_to_png_bytes(prepared_frame)
    prepared_frame.close()

    return (
        prepared_bytes,
        bbox,
        None,
        frame_was_fixed,
    )


@st.cache_data(show_spinner=False)
def make_frame_preview_cached(frame_bytes: bytes) -> bytes:
    """Create a small preview of the selected frame."""
    frame = png_bytes_to_image(frame_bytes)

    try:
        return make_preview_bytes(
            frame,
            max_size=PREVIEW_MAX_SIZE,
            quality=82,
        )
    finally:
        frame.close()


# =========================================================
# Student image fitting
# =========================================================

def required_cover_scale(
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> float:
    """Return the scale required to completely cover a target area."""
    source_width, source_height = source_size
    target_width, target_height = target_size

    return max(
        target_width / source_width,
        target_height / source_height,
    )


def required_contain_scale(
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> float:
    """Return the scale required to contain an image inside a target area."""
    source_width, source_height = source_size
    target_width, target_height = target_size

    return min(
        target_width / source_width,
        target_height / source_height,
    )


def estimate_photo_background_color(
    image: Image.Image,
) -> Tuple[int, int, int]:
    """Estimate the photo background using edge and corner pixels."""
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

    colors = [
        rgb.getpixel(point)
        for point in sample_points
    ]

    red = int(
        sum(color[0] for color in colors)
        / len(colors)
    )

    green = int(
        sum(color[1] for color in colors)
        / len(colors)
    )

    blue = int(
        sum(color[2] for color in colors)
        / len(colors)
    )

    rgb.close()

    if (
        max(red, green, blue) >= 210
        and max(red, green, blue) - min(red, green, blue) <= 35
    ):
        return 255, 255, 255

    return red, green, blue


def crop_with_padding(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    horizontal_padding_ratio: float,
    top_padding_ratio: float,
    bottom_padding_ratio: float,
) -> Image.Image:
    """Crop an image around a bounding box with proportional padding."""
    width, height = image.size

    left, top, right, bottom = bbox

    box_width = right - left
    box_height = bottom - top

    horizontal_padding = int(
        box_width * horizontal_padding_ratio
    )

    top_padding = int(
        box_height * top_padding_ratio
    )

    bottom_padding = int(
        box_height * bottom_padding_ratio
    )

    left = max(0, left - horizontal_padding)
    right = min(width, right + horizontal_padding)
    top = max(0, top - top_padding)
    bottom = min(height, bottom + bottom_padding)

    return image.crop((left, top, right, bottom))


def scale_bbox_between_sizes(
    bbox: Tuple[int, int, int, int],
    from_size: Tuple[int, int],
    to_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Scale a bounding box from one image size to another."""
    from_width, from_height = from_size
    to_width, to_height = to_size

    scale_x = to_width / from_width
    scale_y = to_height / from_height

    left, top, right, bottom = bbox

    return (
        max(0, int(left * scale_x)),
        max(0, int(top * scale_y)),
        min(to_width, int(right * scale_x)),
        min(to_height, int(bottom * scale_y)),
    )


def detect_trim_bbox_fast(
    image: Image.Image,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect photo content bounds using a smaller copy.

    Cropping and final resizing still use higher-quality source pixels.
    """
    image_rgba = image.convert("RGBA")
    original_size = image_rgba.size

    detection_image, _, _ = resize_for_detection(
        image_rgba,
        TRIM_DETECTION_MAX_SIZE,
    )

    detection_size = detection_image.size

    alpha = detection_image.getchannel("A")
    alpha_bbox = alpha.point(
        lambda value: 255 if value > 15 else 0
    ).getbbox()

    if alpha_bbox:
        detection_width, detection_height = detection_size

        alpha_area = (
            alpha_bbox[2] - alpha_bbox[0]
        ) * (
            alpha_bbox[3] - alpha_bbox[1]
        )

        if (
            alpha_area
            < detection_width * detection_height * 0.95
        ):
            result = scale_bbox_between_sizes(
                alpha_bbox,
                detection_size,
                original_size,
            )

            detection_image.close()
            image_rgba.close()
            return result

    rgb = detection_image.convert("RGB")

    background_color = estimate_photo_background_color(rgb)
    background = Image.new("RGB", rgb.size, background_color)

    difference = ImageChops.difference(rgb, background)
    gray = difference.convert("L")

    mask = gray.point(
        lambda value: 255 if value > 22 else 0
    )

    bbox = mask.getbbox()

    rgb.close()
    background.close()
    difference.close()
    gray.close()
    mask.close()

    if bbox is None:
        detection_image.close()
        image_rgba.close()
        return None

    left, top, right, bottom = bbox

    crop_width = right - left
    crop_height = bottom - top

    detection_width, detection_height = detection_size

    if (
        crop_width < detection_width * 0.25
        or crop_height < detection_height * 0.25
    ):
        detection_image.close()
        image_rgba.close()
        return None

    if (
        crop_width > detection_width * 0.96
        and crop_height > detection_height * 0.96
    ):
        detection_image.close()
        image_rgba.close()
        return None

    result = scale_bbox_between_sizes(
        bbox,
        detection_size,
        original_size,
    )

    detection_image.close()
    image_rgba.close()

    return result


def auto_trim_photo_margins(
    image: Image.Image,
) -> Image.Image:
    """Remove unnecessary transparent or plain margins around a photo."""
    if image.mode == "RGBA":
        image_rgba = image
    else:
        image_rgba = image.convert("RGBA")

    width, height = image_rgba.size

    alpha = image_rgba.getchannel("A")
    alpha_bbox = alpha.point(
        lambda value: 255 if value > 15 else 0
    ).getbbox()

    if alpha_bbox:
        alpha_area = (
            alpha_bbox[2] - alpha_bbox[0]
        ) * (
            alpha_bbox[3] - alpha_bbox[1]
        )

        if alpha_area < width * height * 0.95:
            return crop_with_padding(
                image_rgba,
                alpha_bbox,
                0.10,
                0.10,
                0.08,
            )

    bbox = detect_trim_bbox_fast(image_rgba)

    if bbox is None:
        return image_rgba

    left, top, right, bottom = bbox

    crop_width = right - left
    crop_height = bottom - top

    if (
        crop_width < width * 0.25
        or crop_height < height * 0.25
    ):
        return image_rgba

    return crop_with_padding(
        image_rgba,
        bbox,
        0.18,
        0.18,
        0.12,
    )


def cover_crop_to_area(
    image: Image.Image,
    target_size: Tuple[int, int],
) -> Image.Image:
    """Resize and crop an image so it completely covers the target."""
    target_width, target_height = target_size
    source_width, source_height = image.size

    scale = required_cover_scale(
        (source_width, source_height),
        target_size,
    )

    new_width = max(
        target_width,
        int(round(source_width * scale)),
    )

    new_height = max(
        target_height,
        int(round(source_height * scale)),
    )

    resized = image.resize(
        (new_width, new_height),
        RESAMPLE_LANCZOS,
    )

    left = (new_width - target_width) // 2

    extra_height = new_height - target_height

    # Slight upward bias generally keeps faces centered.
    top = int(extra_height * 0.42) if extra_height > 0 else 0

    cropped = resized.crop(
        (
            left,
            top,
            left + target_width,
            top + target_height,
        )
    )

    resized.close()

    return cropped


def contain_fit_to_area(
    image: Image.Image,
    target_size: Tuple[int, int],
    background_color: Tuple[int, int, int],
) -> Image.Image:
    """Fit the whole photo inside the target without cropping."""
    target_width, target_height = target_size
    source_width, source_height = image.size

    scale = required_contain_scale(
        (source_width, source_height),
        target_size,
    ) * 0.985

    new_width = max(
        1,
        int(round(source_width * scale)),
    )

    new_height = max(
        1,
        int(round(source_height * scale)),
    )

    resized = image.resize(
        (new_width, new_height),
        RESAMPLE_LANCZOS,
    )

    area_canvas = Image.new(
        "RGBA",
        target_size,
        (*background_color, 255),
    )

    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    if resized.mode == "RGBA":
        resized_rgba = resized
    else:
        resized_rgba = resized.convert("RGBA")

    area_canvas.alpha_composite(
        resized_rgba,
        (paste_x, paste_y),
    )

    if resized_rgba is not resized:
        resized_rgba.close()

    resized.close()

    return area_canvas


def smart_fit_photo_to_area(
    image: Image.Image,
    area_size: Tuple[int, int],
) -> Image.Image:
    """Choose cover or contain fitting based on aspect-ratio difference."""
    detection_image, _, _ = resize_for_detection(
        image,
        TRIM_DETECTION_MAX_SIZE,
    )

    try:
        original_background = estimate_photo_background_color(
            detection_image
        )
    finally:
        detection_image.close()

    trimmed = auto_trim_photo_margins(image)

    source_width, source_height = trimmed.size
    area_width, area_height = area_size

    source_aspect = source_width / source_height
    area_aspect = area_width / area_height

    aspect_ratio_gap = max(
        source_aspect / area_aspect,
        area_aspect / source_aspect,
    )

    if aspect_ratio_gap >= 1.28:
        result = contain_fit_to_area(
            trimmed,
            area_size,
            original_background,
        )
    else:
        result = cover_crop_to_area(
            trimmed,
            area_size,
        )

    if trimmed is not image:
        trimmed.close()

    return result


def place_student_inside_frame_area(
    student_image: Image.Image,
    frame_size: Tuple[int, int],
    photo_area_bbox: Tuple[int, int, int, int],
) -> Image.Image:
    """Place a fitted student photo onto a frame-sized white canvas."""
    left, top, right, bottom = photo_area_bbox

    area_width = right - left
    area_height = bottom - top

    fitted_area = smart_fit_photo_to_area(
        student_image,
        (area_width, area_height),
    )

    canvas = Image.new(
        "RGBA",
        frame_size,
        (255, 255, 255, 255),
    )

    canvas.alpha_composite(
        fitted_area,
        (left, top),
    )

    fitted_area.close()

    return canvas


def apply_frame(
    student_background: Image.Image,
    frame: Image.Image,
) -> Image.Image:
    """Composite the frame over the prepared student image."""
    if student_background.mode != "RGBA":
        converted_background = student_background.convert("RGBA")
        student_background.close()
        student_background = converted_background

    if frame.mode != "RGBA":
        converted_frame = frame.convert("RGBA")
    else:
        converted_frame = frame

    # Composite in place to avoid another full-frame allocation.
    student_background.alpha_composite(converted_frame)

    if converted_frame is not frame:
        converted_frame.close()

    return student_background


def rgba_to_rgb_for_saving(
    image: Image.Image,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Flatten RGBA onto a solid background for JPEG saving."""
    if image.mode != "RGBA":
        return image.convert("RGB")

    background = Image.new(
        "RGB",
        image.size,
        background_color,
    )

    background.paste(
        image,
        mask=image.getchannel("A"),
    )

    return background


# =========================================================
# Saving, temporary files, and ZIP creation
# =========================================================

def ensure_temp_root() -> Path:
    """Create and return the app's temporary root folder."""
    TEMP_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    return TEMP_ROOT


def is_safe_batch_path(path: Path) -> bool:
    """Ensure a path belongs to this app's temporary directory."""
    try:
        root = ensure_temp_root().resolve()
        candidate = path.resolve()
    except OSError:
        return False

    return (
        candidate != root
        and root in candidate.parents
    )


def remove_batch_directory(path_value: object) -> None:
    """Safely delete a temporary batch directory."""
    if not path_value:
        return

    try:
        path = Path(str(path_value))

        if is_safe_batch_path(path) and path.exists():
            shutil.rmtree(
                path,
                ignore_errors=True,
            )
    except (OSError, ValueError):
        pass


def cleanup_old_batches() -> None:
    """Remove abandoned temporary batches."""
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
                shutil.rmtree(
                    child,
                    ignore_errors=True,
                )
        except OSError:
            continue


def clear_completed_batch() -> None:
    """Remove a completed batch from session state and disk."""
    completed = st.session_state.pop(
        "completed_batch",
        None,
    )

    if isinstance(completed, dict):
        remove_batch_directory(
            completed.get("batch_dir")
        )

    for key in (
        "batch_error",
        "generation_notice",
    ):
        st.session_state.pop(key, None)

    gc.collect()


def clear_pending_batch() -> None:
    """Remove an unfinished pending batch."""
    pending = st.session_state.pop(
        "pending_batch",
        None,
    )

    if isinstance(pending, dict):
        remove_batch_directory(
            pending.get("batch_dir")
        )

    gc.collect()


def safe_staged_filename(
    index: int,
    original_name: str,
    used_names: set,
) -> str:
    """Create a unique temporary filename for an upload."""
    cleaned = sanitize_filename(
        Path(original_name).name
    )

    candidate = f"{index + 1:03d}_{cleaned}"

    stem = Path(candidate).stem
    suffix = Path(candidate).suffix

    counter = 1

    while candidate.lower() in used_names:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1

    used_names.add(candidate.lower())

    return candidate


def stage_uploaded_files(
    uploaded_files,
) -> Tuple[Path, List[Dict[str, object]], int]:
    """
    Copy all UploadedFile objects to temporary disk.

    This allows the Streamlit uploader to be cleared before image processing.
    """
    root = ensure_temp_root()

    batch_id = (
        f"batch_{int(time.time())}_"
        f"{uuid.uuid4().hex[:10]}"
    )

    batch_dir = root / batch_id
    input_dir = batch_dir / "inputs"

    input_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    manifest: List[Dict[str, object]] = []
    total_bytes = 0
    used_names = set()

    try:
        for index, uploaded_file in enumerate(uploaded_files):
            original_name = Path(
                uploaded_file.name
            ).name

            stored_name = safe_staged_filename(
                index,
                original_name,
                used_names,
            )

            destination = input_dir / stored_name

            uploaded_file.seek(0)

            with destination.open("wb") as output_file:
                while True:
                    chunk = uploaded_file.read(
                        UPLOAD_COPY_CHUNK_SIZE
                    )

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
        shutil.rmtree(
            batch_dir,
            ignore_errors=True,
        )
        raise

    return batch_dir, manifest, total_bytes


def save_output_image_to_path(
    image: Image.Image,
    output_path: Path,
    output_format: str,
    jpg_quality: int,
    icc_profile: Optional[bytes],
) -> None:
    """
    Save an output image directly to disk.

    PNG remains lossless.
    JPG uses full-resolution 4:4:4 chroma.
    """

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
                quality=max(
                    90,
                    min(int(jpg_quality), 100),
                ),
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
        # Some uploaded images contain invalid ICC profiles.
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
    """Create a small JPEG preview for the browser."""
    preview = image.copy()

    try:
        preview.thumbnail(
            max_size,
            RESAMPLE_LANCZOS,
        )

        if preview.mode == "RGBA":
            background = Image.new(
                "RGB",
                preview.size,
                (255, 255, 255),
            )

            background.paste(
                preview,
                mask=preview.getchannel("A"),
            )

            preview.close()
            preview = background

        else:
            converted = preview.convert("RGB")
            preview.close()
            preview = converted

        buffer = BytesIO()

        preview.save(
            buffer,
            format="JPEG",
            quality=quality,
            optimize=False,
        )

        return buffer.getvalue()

    finally:
        preview.close()


def make_unique_output_name(
    original_filename: str,
    frame_filename: str,
    output_format: str,
    used_names: set,
) -> str:
    """Create a unique filename for a generated photo."""
    original_stem = sanitize_filename(
        Path(original_filename).stem
    )

    frame_stem = sanitize_filename(
        Path(frame_filename).stem
    )

    extension = (
        "jpg"
        if output_format.upper() == "JPG"
        else "png"
    )

    base_name = (
        f"{original_stem}_{frame_stem}"
    )

    candidate = f"{base_name}.{extension}"
    counter = 1

    while candidate.lower() in used_names:
        candidate = (
            f"{base_name}_{counter}.{extension}"
        )
        counter += 1

    used_names.add(candidate.lower())

    return candidate


def useful_decode_size(
    frame_size: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Determine how much source detail should be retained.

    The final output still uses the exact original frame dimensions.
    """
    return (
        max(
            1,
            int(round(frame_size[0] * SOURCE_OVERSAMPLE)),
        ),
        max(
            1,
            int(round(frame_size[1] * SOURCE_OVERSAMPLE)),
        ),
    )


def open_student_image_from_path(
    filename: str,
    image_path: Path,
    frame_size: Tuple[int, int],
) -> Tuple[
    Optional[Image.Image],
    Optional[bytes],
    Optional[str],
]:
    """Open and prepare one uploaded image from temporary disk."""
    try:
        with Image.open(image_path) as raw_image:
            width, height = raw_image.size

            if width <= 0 or height <= 0:
                return (
                    None,
                    None,
                    f"{filename} skipped. Invalid image dimensions.",
                )

            pixel_count = width * height

            if pixel_count > MAX_SOURCE_PIXELS:
                return (
                    None,
                    None,
                    (
                        f"{filename} skipped. Image is "
                        f"{width}x{height}px "
                        f"({pixel_count:,} pixels); "
                        f"maximum is "
                        f"{MAX_SOURCE_PIXELS:,} pixels."
                    ),
                )

            icc_profile = raw_image.info.get(
                "icc_profile"
            )

            decode_size = useful_decode_size(
                frame_size
            )

            # JPEG draft decoding reduces memory use for huge source photos.
            # At least SOURCE_OVERSAMPLE times the final frame size is retained.
            if (
                (raw_image.format or "").upper()
                in {"JPEG", "JPG"}
                and width > decode_size[0] * 2
                and height > decode_size[1] * 2
            ):
                try:
                    raw_image.draft(
                        "RGB",
                        decode_size,
                    )
                except (OSError, ValueError):
                    pass

            oriented = correct_exif_orientation(
                raw_image
            )

            try:
                oriented.load()

                oriented.thumbnail(
                    decode_size,
                    RESAMPLE_LANCZOS,
                )

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
        return (
            None,
            None,
            (
                f"{filename} skipped. "
                "Invalid, corrupt, or oversized image."
            ),
        )


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
) -> Tuple[
    Optional[Dict[str, object]],
    List[str],
]:
    """Generate one framed photo and save it directly to disk."""
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
            return (
                None,
                [f"{filename} skipped. Could not open image."],
            )

        area_width = (
            photo_area_bbox[2]
            - photo_area_bbox[0]
        )

        area_height = (
            photo_area_bbox[3]
            - photo_area_bbox[1]
        )

        contain_scale = required_contain_scale(
            image.size,
            (area_width, area_height),
        )

        if contain_scale > 1.0:
            messages.append(
                (
                    f"{filename}: small image enlarged from "
                    f"{image.width}x{image.height}px "
                    f"to fit opening "
                    f"{area_width}x{area_height}px."
                )
            )

        final_image = place_student_inside_frame_area(
            student_image=image,
            frame_size=frame_size,
            photo_area_bbox=photo_area_bbox,
        )

        final_image = apply_frame(
            final_image,
            frame,
        )

        save_output_image_to_path(
            image=final_image,
            output_path=output_path,
            output_format=output_format,
            jpg_quality=jpg_quality,
            icc_profile=icc_profile,
        )

        preview_bytes = (
            make_preview_bytes(final_image)
            if make_preview
            else None
        )

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


def process_staged_batch(
    pending: Dict[str, Any],
) -> Dict[str, object]:
    """
    Process up to 100 staged photos.

    Each output is immediately added to one disk-backed ZIP file.
    """
    batch_dir = Path(
        str(pending["batch_dir"])
    )

    files = list(
        pending["files"]
    )

    selected_frame_path = Path(
        str(pending["frame_path"])
    )

    output_format = str(
        pending["output_format"]
    )

    jpg_quality = int(
        pending["jpg_quality"]
    )

    preview_limit = int(
        pending["preview_limit"]
    )

    if (
        not is_safe_batch_path(batch_dir)
        or not batch_dir.exists()
    ):
        raise RuntimeError(
            "The temporary batch directory is missing."
        )

    frame_stat = selected_frame_path.stat()

    (
        frame_bytes,
        photo_area_bbox,
        frame_error,
        _,
    ) = prepare_frame_cached(
        str(selected_frame_path),
        frame_stat.st_mtime,
        frame_stat.st_size,
    )

    if (
        frame_bytes is None
        or photo_area_bbox is None
    ):
        raise RuntimeError(
            frame_error
            or "Could not load the selected frame."
        )

    frame = png_bytes_to_image(frame_bytes)
    frame_size = frame.size

    downloads_dir = batch_dir / "downloads"
    working_dir = batch_dir / "working"

    downloads_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    working_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    zip_path = (
        downloads_dir
        / "graduation_frames_all.zip"
    )

    used_names = set()
    previews: List[Dict[str, object]] = []
    messages: List[str] = []

    generated_count = 0
    total = len(files)

    progress = st.progress(0)
    status_text = st.empty()

    try:
        # JPEG and PNG are already compressed.
        # ZIP_STORED packages them without altering quality.
        with zipfile.ZipFile(
            zip_path,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as zip_file:

            for index, item in enumerate(files):
                original_name = str(
                    item["original_name"]
                )

                source_path = Path(
                    str(item["path"])
                )

                status_text.write(
                    (
                        f"Processing {index + 1} "
                        f"of {total}: {original_name}"
                    )
                )

                output_name = make_unique_output_name(
                    original_filename=original_name,
                    frame_filename=selected_frame_path.name,
                    output_format=output_format,
                    used_names=used_names,
                )

                temporary_output = (
                    working_dir / output_name
                )

                try:
                    result, image_messages = (
                        process_single_image_to_file(
                            filename=original_name,
                            image_path=source_path,
                            output_path=temporary_output,
                            output_name=output_name,
                            frame_size=frame_size,
                            frame=frame,
                            photo_area_bbox=photo_area_bbox,
                            output_format=output_format,
                            jpg_quality=jpg_quality,
                            make_preview=(
                                generated_count
                                < preview_limit
                            ),
                        )
                    )

                except Exception as exception:
                    result = None

                    image_messages = [
                        (
                            f"{original_name} skipped "
                            f"because of an error: "
                            f"{exception}"
                        )
                    ]

                messages.extend(image_messages)

                if (
                    result is not None
                    and temporary_output.exists()
                ):
                    zip_file.write(
                        temporary_output,
                        arcname=output_name,
                    )

                    generated_count += 1

                    if result.get("preview_bytes"):
                        previews.append(result)

                try:
                    temporary_output.unlink(
                        missing_ok=True
                    )
                except OSError:
                    pass

                try:
                    source_path.unlink(
                        missing_ok=True
                    )
                except OSError:
                    pass

                progress.progress(
                    (index + 1) / total
                )

                if index % 2 == 1:
                    gc.collect()

    finally:
        frame.close()
        progress.empty()
        status_text.empty()

        shutil.rmtree(
            working_dir,
            ignore_errors=True,
        )

        gc.collect()

    zip_information: Optional[
        Dict[str, object]
    ] = None

    if (
        generated_count > 0
        and zip_path.exists()
    ):
        zip_information = {
            "path": str(zip_path),
            "filename": zip_path.name,
            "count": generated_count,
            "size": zip_path.stat().st_size,
        }

    else:
        try:
            zip_path.unlink(missing_ok=True)
        except OSError:
            pass

    return {
        "batch_dir": str(batch_dir),
        "batch_id": batch_dir.name,
        "generated_count": generated_count,
        "requested_count": total,
        "messages": messages,
        "previews": previews,
        "preview_limit": preview_limit,
        "zip_file": zip_information,
        "output_format": output_format,
        "frame_size": frame_size,
    }


# =========================================================
# Streamlit interface
# =========================================================

def human_size(size_bytes: int) -> str:
    """Format a byte count for display."""
    size = float(size_bytes)

    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"

        size /= 1024

    return f"{size_bytes} B"


def show_gallery(
    results: List[Dict[str, object]],
) -> None:
    """Display small previews without storing full output images in memory."""
    if not results:
        return

    st.subheader("Preview")

    columns_per_row = 4

    for start in range(
        0,
        len(results),
        columns_per_row,
    ):
        columns = st.columns(
            columns_per_row
        )

        current_results = results[
            start:start + columns_per_row
        ]

        for column, result in zip(
            columns,
            current_results,
        ):
            with column:
                st.image(
                    result["preview_bytes"],
                    caption=str(
                        result["filename"]
                    ),
                    use_container_width=True,
                )


def run_pending_batch() -> None:
    """Run a staged batch after the uploader has been cleared."""
    pending = st.session_state.get(
        "pending_batch"
    )

    if not isinstance(pending, dict):
        return

    st.subheader("Generating images")

    st.caption(
        (
            "The upload widget has been cleared to release memory. "
            "Photos are processed one at a time at full frame "
            "resolution, and every successful result is added "
            "to one ZIP file."
        )
    )

    try:
        wait_notice = st.empty()

        if PROCESSING_LOCK.locked():
            wait_notice.info(
                (
                    "Another batch is currently being processed. "
                    "This batch will start when it finishes."
                )
            )

        with PROCESSING_LOCK:
            wait_notice.empty()

            completed = process_staged_batch(
                pending
            )

    except Exception as exception:
        st.session_state["batch_error"] = (
            f"Generation failed: {exception}"
        )

        remove_batch_directory(
            pending.get("batch_dir")
        )

    else:
        st.session_state[
            "completed_batch"
        ] = completed

        st.session_state[
            "generation_notice"
        ] = (
            f"Generated "
            f"{completed['generated_count']} of "
            f"{completed['requested_count']} image(s)."
        )

    finally:
        st.session_state.pop(
            "pending_batch",
            None,
        )

        gc.collect()

    st.rerun()


def show_completed_batch() -> None:
    """Show warnings, previews, and the final all-images ZIP download."""
    completed = st.session_state.get(
        "completed_batch"
    )

    if not isinstance(completed, dict):
        return

    generated_count = int(
        completed.get(
            "generated_count",
            0,
        )
    )

    requested_count = int(
        completed.get(
            "requested_count",
            0,
        )
    )

    frame_size = completed.get(
        "frame_size"
    )

    if generated_count:
        st.success(
            (
                f"Generated {generated_count} "
                f"of {requested_count} image(s)."
            )
        )

        if (
            isinstance(frame_size, (tuple, list))
            and len(frame_size) == 2
        ):
            st.caption(
                (
                    f"Every output is "
                    f"{frame_size[0]} x "
                    f"{frame_size[1]} pixels, "
                    "the exact selected-frame resolution."
                )
            )

    else:
        st.error(
            "No images were generated."
        )

    messages = (
        completed.get("messages")
        or []
    )

    if messages:
        with st.expander(
            f"Warnings ({len(messages)})"
        ):
            for message in messages:
                st.warning(str(message))

    zip_information = completed.get(
        "zip_file"
    )

    zip_path: Optional[Path] = None

    if isinstance(zip_information, dict):
        candidate = Path(
            str(
                zip_information.get(
                    "path",
                    "",
                )
            )
        )

        if candidate.exists() and candidate.is_file():
            zip_path = candidate

    if (
        zip_path is not None
        and generated_count
    ):
        st.subheader(
            "Download all photos"
        )

        zip_size = int(
            zip_information.get(
                "size",
                zip_path.stat().st_size,
            )
        )

        zip_name = str(
            zip_information.get(
                "filename",
                zip_path.name,
            )
        )

        st.write(
            (
                f"One ZIP file contains all "
                f"{generated_count} generated photo(s): "
                f"**{human_size(zip_size)}**"
            )
        )

        # Streamlit 1.60+ runs this only when the user clicks.
        # The complete ZIP is therefore not loaded into memory on every rerun.
        def open_complete_zip():
            return zip_path.open("rb")

        st.download_button(
            label=(
                f"Download all "
                f"{generated_count} photos as ZIP"
            ),
            data=open_complete_zip,
            file_name=zip_name,
            mime="application/zip",
            key=(
                f"download_all_"
                f"{completed.get('batch_id')}"
            ),
            on_click="ignore",
            type="primary",
            icon=":material/download:",
            width="stretch",
        )

        st.caption(
            (
               
            )
        )

    elif generated_count:
        st.error(
            (
                "The generated ZIP file is no longer available. "
                "Generate the batch again."
            )
        )

    show_gallery(
        list(
            completed.get("previews")
            or []
        )
    )

    if st.button(
        "Clear generated batch",
        width="stretch",
    ):
        clear_completed_batch()
        st.rerun()


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Graduation Frame App",
        page_icon="🎓",
        layout="wide",
    )

    cleanup_old_batches()

    st.title(
        "🎓 Graduation Frame App"
    )

    # A staged batch is processed on a new Streamlit run after uploads
    # have been copied to disk and removed from the upload widget.
    if st.session_state.get(
        "pending_batch"
    ):
        run_pending_batch()
        return

    if st.session_state.get(
        "batch_error"
    ):
        st.error(
            st.session_state.pop(
                "batch_error"
            )
        )

    if st.session_state.get(
        "generation_notice"
    ):
        st.info(
            st.session_state.pop(
                "generation_notice"
            )
        )

    with st.sidebar:
        output_format = st.radio(
            "Output format",
            ["PNG", "JPG"],
            index=1,
        )

        jpg_quality = 100

        if output_format == "JPG":
            jpg_quality = st.slider(
                "JPG quality",
                min_value=90,
                max_value=100,
                value=100,
            )

            st.caption(
                (
                    
                )
            )

        else:
            st.caption(
                "PNG output is lossless."
            )

        preview_limit = st.slider(
            "Preview limit",
            min_value=0,
            max_value=MAX_PREVIEW_LIMIT,
            value=DEFAULT_PREVIEW_LIMIT,
            help=(
                "This affects only browser previews. "
                "It does not affect generated files or ZIP contents."
            ),
        )

    frames, frame_error = load_frames(
        FRAMES_DIR
    )

    if frame_error:
        st.error(frame_error)
        st.stop()

    frame_names = [
        frame.name
        for frame in frames
    ]

    selected_frame_name = st.selectbox(
        "Frame",
        frame_names,
    )

    selected_frame_path = (
        FRAMES_DIR
        / selected_frame_name
    )

    frame_stat = selected_frame_path.stat()

    (
        frame_bytes,
        photo_area_bbox,
        frame_error,
        frame_was_fixed,
    ) = prepare_frame_cached(
        str(selected_frame_path),
        frame_stat.st_mtime,
        frame_stat.st_size,
    )

    if (
        frame_bytes is None
        or photo_area_bbox is None
    ):
        st.error(
            frame_error
            or "Could not load the selected frame."
        )

        st.stop()

    left_column, right_column = st.columns(
        [1, 2]
    )

    with left_column:
        frame_preview_bytes = (
            make_frame_preview_cached(
                frame_bytes
            )
        )

        st.image(
            frame_preview_bytes,
            caption=selected_frame_name,
            use_container_width=True,
        )

        with Image.open(
            selected_frame_path
        ) as raw_frame:
            st.caption(
                (
                    f"Output resolution: "
                    f"{raw_frame.width} x "
                    f"{raw_frame.height} pixels"
                )
            )

        if frame_was_fixed:
            st.success(
                "Frame opening detected"
            )

    with right_column:
        uploader_nonce = int(
            st.session_state.get(
                "uploader_nonce",
                0,
            )
        )

        uploaded_files = st.file_uploader(
            "Student photos (maximum 100)",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
            key=(
                f"student_photos_"
                f"{uploader_nonce}"
            ),
            max_upload_size=100,
        )

        upload_error: Optional[str] = None

        if uploaded_files:
            total_upload_bytes = sum(
                int(
                    getattr(
                        uploaded_file,
                        "size",
                        0,
                    )
                    or 0
                )
                for uploaded_file in uploaded_files
            )

            st.write(
                (
                    f"{len(uploaded_files)} image(s) selected "
                    f"({human_size(total_upload_bytes)} total)"
                )
            )

            if (
                len(uploaded_files)
                > MAX_UPLOAD_FILES
            ):
                upload_error = (
                    f"Select at most "
                    f"{MAX_UPLOAD_FILES} photos "
                    "in one batch."
                )

        if upload_error:
            st.error(upload_error)

        generate = st.button(
            "Generate",
            type="primary",
            disabled=(
                not uploaded_files
                or upload_error is not None
            ),
            use_container_width=True,
        )

    if generate:
        clear_pending_batch()
        clear_completed_batch()

        try:
            with st.spinner(
                "Copying uploads to temporary disk..."
            ):
                (
                    batch_dir,
                    manifest,
                    total_bytes,
                ) = stage_uploaded_files(
                    uploaded_files
                )

        except Exception as exception:
            st.error(
                (
                    "Could not stage the uploaded files: "
                    f"{exception}"
                )
            )

        else:
            st.session_state[
                "pending_batch"
            ] = {
                "batch_dir": str(batch_dir),
                "files": manifest,
                "total_upload_bytes": total_bytes,
                "frame_path": str(
                    selected_frame_path
                ),
                "output_format": output_format,
                "jpg_quality": jpg_quality,
                "preview_limit": preview_limit,
            }

            # A new uploader key clears UploadedFile objects before
            # the expensive processing begins.
            st.session_state[
                "uploader_nonce"
            ] = uploader_nonce + 1

            del uploaded_files
            gc.collect()

            st.rerun()

    show_completed_batch()


if __name__ == "__main__":
    main()