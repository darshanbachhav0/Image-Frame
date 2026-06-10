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
DEFAULT_OUTPUT_DIR_NAME = "output"
SUPPORTED_UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp"]

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS
    RESAMPLE_BICUBIC = Image.BICUBIC


# ------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------
def sanitize_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^\w.\- ]+", "_", name, flags=re.UNICODE).strip()
    name = re.sub(r"\s+", " ", name)
    return name or "image"


def sanitize_folder_name(folder_name: str) -> str:
    folder_name = folder_name.strip().replace("\\", "/").split("/")[-1]
    folder_name = folder_name.strip(". ")
    folder_name = re.sub(r"[^\w.\- ]+", "_", folder_name, flags=re.UNICODE).strip()
    return folder_name or DEFAULT_OUTPUT_DIR_NAME


def ensure_output_folder(folder_name: str) -> Tuple[Optional[Path], str, Optional[str]]:
    safe_name = sanitize_folder_name(folder_name)
    output_dir = BASE_DIR / safe_name

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir, safe_name, None
    except OSError as exc:
        return None, safe_name, str(exc)


def load_frames(frames_dir: Path = FRAMES_DIR) -> Tuple[List[Path], Optional[str]]:
    if not frames_dir.exists():
        return [], "frames/ folder missing."

    if not frames_dir.is_dir():
        return [], "frames is not a folder."

    frames = sorted(frames_dir.glob("*.png"))

    if not frames:
        return [], "No PNG frames found in frames/."

    return frames, None


def correct_exif_orientation(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image)


def image_has_transparency(image: Image.Image) -> bool:
    image = image.convert("RGBA")
    alpha_min, alpha_max = image.getchannel("A").getextrema()
    return alpha_min < 255 and alpha_max > 0


# ------------------------------------------------------------
# Automatic frame transparency fixing
# ------------------------------------------------------------
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


def auto_make_center_transparent(frame: Image.Image) -> Tuple[Image.Image, bool]:
    """
    Automatically removes fake checkerboard / light center background.

    This is useful when the frame is saved with a visible checkerboard pattern
    instead of real transparency.
    """
    img = frame.convert("RGBA")

    if image_has_transparency(img):
        center_is_fake = detect_fake_center_background(img)

        if not center_is_fake:
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

        for color in seed_colors:
            if is_close_color(r, g, b, color, tolerance):
                return True

        return False

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

    changed_ratio = changed_pixels / float(width * height)

    return img, changed_ratio > 0.005


# ------------------------------------------------------------
# Frame photo area detection
# ------------------------------------------------------------
def find_largest_transparent_region_bbox(
    frame: Image.Image,
    alpha_threshold: int = 30,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Finds the largest connected transparent area in the frame.
    This is usually the inner photo area.
    """
    img = frame.convert("RGBA")
    width, height = img.size
    alpha = img.getchannel("A")
    alpha_pixels = alpha.load()

    visited = bytearray(width * height)

    best_bbox = None
    best_area = 0

    center_x = width // 2
    center_y = height // 2

    for y in range(height):
        for x in range(width):
            idx = y * width + x

            if visited[idx]:
                continue

            visited[idx] = 1

            if alpha_pixels[x, y] > alpha_threshold:
                continue

            queue = deque([(x, y)])

            min_x = max_x = x
            min_y = max_y = y
            count = 0

            touches_border = False

            while queue:
                cx, cy = queue.popleft()
                count += 1

                if cx == 0 or cy == 0 or cx == width - 1 or cy == height - 1:
                    touches_border = True

                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height:
                        nidx = ny * width + nx

                        if visited[nidx]:
                            continue

                        visited[nidx] = 1

                        if alpha_pixels[nx, ny] <= alpha_threshold:
                            queue.append((nx, ny))

            if count < 100:
                continue

            bbox = (min_x, min_y, max_x + 1, max_y + 1)
            box_w = bbox[2] - bbox[0]
            box_h = bbox[3] - bbox[1]

            if box_w < width * 0.10 or box_h < height * 0.10:
                continue

            box_center_x = (bbox[0] + bbox[2]) / 2
            box_center_y = (bbox[1] + bbox[3]) / 2

            center_distance = abs(box_center_x - center_x) + abs(box_center_y - center_y)

            score = count

            if touches_border:
                score *= 0.30

            if center_distance > (width + height) * 0.35:
                score *= 0.50

            if score > best_area:
                best_area = score
                best_bbox = bbox

    return best_bbox


def bbox_from_alpha(frame: Image.Image, alpha_threshold: int = 30) -> Optional[Tuple[int, int, int, int]]:
    """
    Fallback method: get bounding box of transparent pixels.
    """
    img = frame.convert("RGBA")
    alpha = img.getchannel("A")

    mask = alpha.point(lambda a: 255 if a <= alpha_threshold else 0)
    bbox = mask.getbbox()

    if bbox is None:
        return None

    width, height = img.size
    box_w = bbox[2] - bbox[0]
    box_h = bbox[3] - bbox[1]

    if box_w < width * 0.10 or box_h < height * 0.10:
        return None

    return bbox


def shrink_bbox(
    bbox: Tuple[int, int, int, int],
    shrink_percent: float,
) -> Tuple[int, int, int, int]:
    """
    Slightly shrink detected opening so the photo does not leak under inner border.
    """
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top

    dx = int(width * shrink_percent)
    dy = int(height * shrink_percent)

    return left + dx, top + dy, right - dx, bottom - dy


def detect_photo_area(frame: Image.Image) -> Tuple[int, int, int, int]:
    """
    Detects the real inner photo opening of the frame.
    """
    bbox = find_largest_transparent_region_bbox(frame)

    if bbox is None:
        bbox = bbox_from_alpha(frame)

    if bbox is None:
        width, height = frame.size
        margin_x = int(width * 0.15)
        margin_y = int(height * 0.15)
        bbox = (margin_x, margin_y, width - margin_x, height - margin_y)

    bbox = shrink_bbox(bbox, 0.015)

    left, top, right, bottom = bbox
    width, height = frame.size

    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)

    return left, top, right, bottom


def load_and_prepare_frame(
    frame_path: Path,
) -> Tuple[Optional[Image.Image], Optional[Tuple[int, int, int, int]], Optional[str], bool]:
    """
    Loads frame, automatically fixes fake checkerboard center,
    and detects the inner photo area.
    """
    try:
        with Image.open(frame_path) as raw_frame:
            raw_frame.load()
            frame = raw_frame.convert("RGBA")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return None, None, f"Frame error: {exc}", False

    fixed_frame, was_fixed = auto_make_center_transparent(frame)

    if not image_has_transparency(fixed_frame):
        photo_area = detect_photo_area(fixed_frame)
        return fixed_frame, photo_area, "Frame has no transparent area.", was_fixed

    photo_area = detect_photo_area(fixed_frame)

    return fixed_frame, photo_area, None, was_fixed


# ------------------------------------------------------------
# Student photo automatic adjustment
# ------------------------------------------------------------
def required_cover_scale(source_size: Tuple[int, int], target_size: Tuple[int, int]) -> float:
    source_w, source_h = source_size
    target_w, target_h = target_size

    return max(target_w / source_w, target_h / source_h)


def auto_trim_photo_margins(image: Image.Image) -> Image.Image:
    """
    Automatically trims plain white/light margins from student ID photos.
    This helps the face/body sit better inside the frame.
    """
    rgb = image.convert("RGB")

    background = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
    diff = ImageChops.difference(rgb, background)

    gray = diff.convert("L")
    mask = gray.point(lambda p: 255 if p > 12 else 0)

    bbox = mask.getbbox()

    if bbox is None:
        return image

    img_w, img_h = image.size
    left, top, right, bottom = bbox

    crop_w = right - left
    crop_h = bottom - top

    if crop_w < img_w * 0.35 or crop_h < img_h * 0.35:
        return image

    pad_x = int(crop_w * 0.08)
    pad_y_top = int(crop_h * 0.12)
    pad_y_bottom = int(crop_h * 0.08)

    left = max(0, left - pad_x)
    right = min(img_w, right + pad_x)
    top = max(0, top - pad_y_top)
    bottom = min(img_h, bottom + pad_y_bottom)

    return image.crop((left, top, right, bottom))


def fit_photo_to_area(
    image: Image.Image,
    area_size: Tuple[int, int],
) -> Image.Image:
    """
    Automatically fits student photo into the detected frame opening.
    Uses cover crop so there is no empty space.
    """
    target_w, target_h = area_size
    source_w, source_h = image.size

    scale = required_cover_scale((source_w, source_h), area_size)

    new_w = max(target_w, int(round(source_w * scale)))
    new_h = max(target_h, int(round(source_h * scale)))

    resized = image.resize((new_w, new_h), RESAMPLE_LANCZOS)

    left = (new_w - target_w) // 2

    # Portrait-friendly crop:
    # Slightly move crop upward so faces are not cut from the forehead
    # and the shoulders remain inside the frame.
    extra_h = new_h - target_h

    if extra_h > 0:
        top = int(extra_h * 0.38)
    else:
        top = 0

    right = left + target_w
    bottom = top + target_h

    return resized.crop((left, top, right, bottom))


def place_student_inside_frame_area(
    student_image: Image.Image,
    frame_size: Tuple[int, int],
    photo_area: Tuple[int, int, int, int],
) -> Image.Image:
    """
    Creates a full-size canvas, then places the student photo only inside
    the detected transparent frame opening.
    """
    left, top, right, bottom = photo_area
    area_w = right - left
    area_h = bottom - top

    trimmed = auto_trim_photo_margins(student_image)
    fitted = fit_photo_to_area(trimmed, (area_w, area_h))

    canvas = Image.new("RGBA", frame_size, (255, 255, 255, 255))
    canvas.alpha_composite(fitted.convert("RGBA"), (left, top))

    return canvas


def apply_frame(student_background: Image.Image, frame: Image.Image) -> Image.Image:
    if student_background.mode != "RGBA":
        student_background = student_background.convert("RGBA")

    if frame.mode != "RGBA":
        frame = frame.convert("RGBA")

    if student_background.size != frame.size:
        frame = frame.resize(student_background.size, RESAMPLE_LANCZOS)

    return Image.alpha_composite(student_background, frame)


# ------------------------------------------------------------
# Save / ZIP / preview
# ------------------------------------------------------------
def rgba_to_rgb(
    image: Image.Image,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGB")

    background = Image.new("RGB", image.size, background_color)
    background.paste(image, mask=image.getchannel("A"))

    return background


def save_output_image(
    image: Image.Image,
    output_path: Path,
    output_format: str,
    jpg_quality: int,
    icc_profile: Optional[bytes],
) -> None:
    save_kwargs: Dict[str, object] = {}

    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile

    if output_format.upper() == "PNG":
        image.save(
            output_path,
            format="PNG",
            compress_level=1,
            optimize=False,
            **save_kwargs,
        )
    else:
        rgb_image = rgba_to_rgb(image)
        rgb_image.save(
            output_path,
            format="JPEG",
            quality=int(jpg_quality),
            subsampling=0,
            optimize=True,
            **save_kwargs,
        )


def make_unique_output_path(
    output_dir: Path,
    original_filename: str,
    frame_filename: str,
    output_format: str,
    used_names: set,
) -> Path:
    original_stem = sanitize_filename(Path(original_filename).stem)
    frame_stem = sanitize_filename(Path(frame_filename).stem)

    extension = "jpg" if output_format.upper() == "JPG" else "png"
    base_name = f"{original_stem}_{frame_stem}"

    candidate = f"{base_name}.{extension}"
    counter = 1

    while candidate.lower() in used_names or (output_dir / candidate).exists():
        candidate = f"{base_name}_{counter}.{extension}"
        counter += 1

    used_names.add(candidate.lower())

    return output_dir / candidate


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


def create_zip(file_paths: List[Path]) -> BytesIO:
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in file_paths:
            if file_path.exists():
                zip_file.write(file_path, arcname=file_path.name)

    zip_buffer.seek(0)
    return zip_buffer


def process_uploaded_files(
    uploaded_files,
    frame: Image.Image,
    photo_area: Tuple[int, int, int, int],
    selected_frame_path: Path,
    output_dir: Path,
    output_format: str,
    jpg_quality: int,
) -> Tuple[List[Dict[str, object]], List[str], bytes]:
    results: List[Dict[str, object]] = []
    messages: List[str] = []
    generated_paths: List[Path] = []

    frame_size = frame.size
    used_names = {p.name.lower() for p in output_dir.iterdir() if p.is_file()}

    progress = st.progress(0)

    area_w = photo_area[2] - photo_area[0]
    area_h = photo_area[3] - photo_area[1]

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        image, icc_profile, error = open_student_image(uploaded_file)

        if error:
            messages.append(error)
            progress.progress(index / len(uploaded_files))
            continue

        if image is None:
            progress.progress(index / len(uploaded_files))
            continue

        scale_needed = required_cover_scale(image.size, (area_w, area_h))

        if scale_needed > 1.0:
            messages.append(
                f"{uploaded_file.name}: small image enlarged from "
                f"{image.width}x{image.height}px to fit frame opening {area_w}x{area_h}px."
            )

        student_background = place_student_inside_frame_area(
            student_image=image,
            frame_size=frame_size,
            photo_area=photo_area,
        )

        final_image = apply_frame(student_background, frame)

        output_path = make_unique_output_path(
            output_dir=output_dir,
            original_filename=uploaded_file.name,
            frame_filename=selected_frame_path.name,
            output_format=output_format,
            used_names=used_names,
        )

        save_output_image(
            image=final_image,
            output_path=output_path,
            output_format=output_format,
            jpg_quality=jpg_quality,
            icc_profile=icc_profile,
        )

        generated_paths.append(output_path)

        results.append(
            {
                "filename": output_path.name,
                "path": output_path,
                "preview_bytes": make_preview_bytes(final_image),
                "size": final_image.size,
            }
        )

        progress.progress(index / len(uploaded_files))

    progress.empty()

    zip_bytes = create_zip(generated_paths).getvalue()

    return results, messages, zip_bytes


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
                    caption=result["filename"],
                    use_container_width=True,
                )


# ------------------------------------------------------------
# Streamlit app
# ------------------------------------------------------------
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

        output_folder_name = st.text_input("Output folder", DEFAULT_OUTPUT_DIR_NAME)

    output_dir, safe_folder_name, output_error = ensure_output_folder(output_folder_name)

    if output_error or output_dir is None:
        st.error(f"Output folder error: {output_error}")
        st.stop()

    frames, frame_error = load_frames(FRAMES_DIR)

    if frame_error:
        st.error(frame_error)
        st.stop()

    frame_names = [frame.name for frame in frames]

    selected_frame_name = st.selectbox("Frame", frame_names)
    selected_frame_path = FRAMES_DIR / selected_frame_name

    frame_image, photo_area, frame_error, frame_was_fixed = load_and_prepare_frame(selected_frame_path)

    if frame_image is None or photo_area is None:
        st.error(frame_error or "Could not load frame.")
        st.stop()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(frame_image, caption=selected_frame_name, use_container_width=True)

    with col2:
        uploaded_files = st.file_uploader(
            "Student photos",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.write(f"{len(uploaded_files)} image(s) selected")

        if frame_was_fixed:
            st.success("Frame auto-fixed")

        if frame_error:
            st.warning(frame_error)

        generate = st.button(
            "Generate ZIP",
            type="primary",
            disabled=not uploaded_files,
            use_container_width=True,
        )

    if generate:
        results, messages, zip_bytes = process_uploaded_files(
            uploaded_files=uploaded_files,
            frame=frame_image,
            photo_area=photo_area,
            selected_frame_path=selected_frame_path,
            output_dir=output_dir,
            output_format=output_format,
            jpg_quality=jpg_quality,
        )

        st.session_state["results"] = results
        st.session_state["messages"] = messages
        st.session_state["zip_bytes"] = zip_bytes
        st.session_state["zip_filename"] = f"{safe_folder_name}_graduation_frames.zip"

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