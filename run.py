from rembg import remove
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from io import BytesIO
import math

# === Parameters ===
MARGIN_RATIO = 0.076  # 7.6%
WHITE_POINT = 219     # Reduce White Point
BLACK_POINT = 0       # Black Point at 0
MIDPOINT = 1.0        # Gamma stays neutral
BRIGHTNESS_INCREASE = 1.05
CONTRAST_INCREASE = 1.14
MAX_SIZE = 1000       # Final image max width/height


def apply_levels(img: Image.Image, black_point=0, gamma=1.0, white_point=255):
    """Apply Levels adjustment similar to Photoshop/GIMP."""
    arr = np.array(img).astype(np.float32)

    arr = (arr - black_point) / (white_point - black_point)
    arr = np.clip(arr, 0, 1)

    arr = arr ** (1.0 / gamma)

    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def crop_nonwhite(img: Image.Image, tol=240):
    """
    Crops the image to the bounding box of all non-white pixels.
    tol = threshold (0=black, 255=white).
    """
    arr = np.asarray(img.convert("RGB"))
    mask = (arr < tol).any(axis=2)

    if not mask.any():
        return img  # if nothing found, return unchanged

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # slices are exclusive

    return img.crop((x0, y0, x1, y1))


def process_image(input_path, output_path):
    # 0. Open & auto-rotate according to EXIF
    with Image.open(input_path) as im:
        im = ImageOps.exif_transpose(im)
        im.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="PNG")
        buf.seek(0)

    try:
        # 1. Try background removal WITH alpha matting
        result = remove(
            buf.read(),
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_structure_size=10,
        )
    except Exception as e:
        print(f"[WARN] Alpha matting failed ({e}), falling back to plain remove.")
        buf.seek(0)
        result = remove(buf.read())

    obj = Image.open(BytesIO(result)).convert("RGBA")

    # 2. Crop non-white pixels
    obj_cropped = crop_nonwhite(obj)

    # 3. White background
    white_bg = Image.new("RGBA", obj_cropped.size, (255, 255, 255, 255))
    obj_cropped = Image.alpha_composite(white_bg, obj_cropped).convert("RGB")

    # 4. Build square canvas w/ exact centered margins
    w, h = obj_cropped.size
    max_side = max(w, h)
    target_size = math.ceil(max_side / (1 - 2 * MARGIN_RATIO))

    new_img = Image.new("RGB", (target_size, target_size), "white")
    x_offset = (target_size - w) // 2
    y_offset = (target_size - h) // 2
    new_img.paste(obj_cropped, (x_offset, y_offset))

    # 5. Levels
    new_img = apply_levels(new_img, BLACK_POINT, MIDPOINT, WHITE_POINT)

    # 6. Brightness + Contrast
    new_img = ImageEnhance.Brightness(new_img).enhance(BRIGHTNESS_INCREASE)
    new_img = ImageEnhance.Contrast(new_img).enhance(CONTRAST_INCREASE)

    # 7. Resize to max 1000 px
    new_img.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)

    # 8. Save
    new_img.save(output_path, quality=95)


# Example usage:
if __name__ == "__main__":
    input_path = "unfiltered-photos/IMG_1915.jpeg"
    output_path = "output.png"

    process_image(input_path, output_path)
    print(f"Saved processed image to {output_path}")
