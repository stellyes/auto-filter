from rembg import remove
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import cv2
from io import BytesIO
import math
import os

# === Parameters ===
MARGIN_RATIO = 0.076         # 7.6% margin
WHITE_POINT = 219
BLACK_POINT = 0
MIDPOINT = 1.0
BRIGHTNESS_INCREASE = 1.05
CONTRAST_INCREASE = 1.14
MAX_SIZE = 1000              # final max dimension
RESIZE_FOR_REMOVAL = 1200    # size rembg receives

# === DETECTION PARAMETERS ===
COLOR_DIST_THRESH = 8        # Distance from white to consider non-background (lowered to catch subtle flower)
MIN_OBJECT_AREA = 50         # Minimum pixels for a valid object (very low to catch all fragments)
PAD_FRAC = 0.15              # Padding around detected objects
MIN_PAD_PX = 30              # Minimum padding

DEBUG = True

def apply_levels(img: Image.Image, black_point=0, gamma=1.0, white_point=255):
    arr = np.array(img).astype(np.float32)
    arr = (arr - black_point) / (white_point - black_point)
    arr = np.clip(arr, 0, 1)
    arr = arr ** (1.0 / gamma)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)

def ensure_dirs(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def debug_save(img, path):
    ensure_dirs(path)
    if isinstance(img, np.ndarray):
        if img.dtype == np.bool_:
            img = (img.astype(np.uint8) * 255)
        if img.ndim == 2:
            Image.fromarray(img).save(path)
            return
        elif img.ndim == 3:
            Image.fromarray(img).save(path)
            return
    img.save(path)

def detect_objects_in_original(orig_img: Image.Image, debug_dir=None):
    """
    Detect all non-background objects using edge detection
    Returns bounding box (x0, y0, x1, y1) that encompasses all objects
    """
    W, H = orig_img.size
    max_dim = max(W, H)
    
    # Convert to numpy
    rgb = np.asarray(orig_img.convert("RGB")).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # Use edge detection instead of color distance
    edges = cv2.Canny(gray, 30, 100)
    
    if debug_dir:
        debug_save(edges, os.path.join(debug_dir, "debug_original_mask_raw.png"))
    
    # Dilate edges to create object regions
    kernel = np.ones((11, 11), np.uint8)
    mask_clean = cv2.dilate(edges, kernel, iterations=3)
    
    if debug_dir:
        debug_save(mask_clean, os.path.join(debug_dir, "debug_original_mask_clean.png"))
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
    
    # Collect all valid objects and print sizes for debugging
    valid_mask = np.zeros_like(mask_clean, dtype=bool)
    valid_count = 0
    
    if debug_dir:
        print(f"\n=== Component Analysis ===")
        print(f"Total components found: {num_labels - 1}")
    
    for label in range(1, num_labels):  # Skip background (0)
        area = stats[label, cv2.CC_STAT_AREA]
        x, y, w, h = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        
        if debug_dir:
            print(f"Component {label}: area={area} pixels, bbox=({x},{y},{w}x{h})")
        
        if area > MIN_OBJECT_AREA:
            valid_mask |= (labels == label)
            valid_count += 1
            if debug_dir:
                print(f"  -> KEPT (area > {MIN_OBJECT_AREA})")
        elif debug_dir:
            print(f"  -> FILTERED OUT (area <= {MIN_OBJECT_AREA})")
    
    if debug_dir:
        debug_save(valid_mask.astype(np.uint8) * 255, os.path.join(debug_dir, "debug_original_objects.png"))
        print(f"\nKept {valid_count} valid objects")
        print(f"=========================\n")
    
    # Find bounding box
    ys, xs = np.where(valid_mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    
    # Add asymmetric padding
    pad_horizontal = max(MIN_PAD_PX, int(max_dim * PAD_FRAC))
    pad_bottom = max(MIN_PAD_PX, int(max_dim * PAD_FRAC))
    pad_top = max(MIN_PAD_PX // 2, int(max_dim * PAD_FRAC * 0.5))
    
    x0 = max(0, x0 - pad_horizontal)
    y0 = max(0, y0 - pad_top)
    x1 = min(W, x1 + pad_horizontal)
    y1 = min(H, y1 + pad_bottom)
    
    if debug_dir:
        debug_img = rgb.copy()
        cv2.rectangle(debug_img, (x0, y0), (x1, y1), (255, 0, 0), 3)
        Image.fromarray(debug_img).save(os.path.join(debug_dir, "debug_original_bbox.png"))
    
    return (x0, y0, x1, y1)

def process_image(input_path, output_path):
    debug_dir = os.path.splitext(output_path)[0] + "_debug"
    if DEBUG and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    # --- open & orient & resize original ---
    orig = Image.open(input_path)
    orig = ImageOps.exif_transpose(orig)
    orig.thumbnail((RESIZE_FOR_REMOVAL, RESIZE_FOR_REMOVAL), Image.Resampling.LANCZOS)

    if DEBUG:
        debug_save(orig.convert("RGB"), os.path.join(debug_dir, "debug_orig_before_rembg.png"))

    # --- Detect objects in original BEFORE rembg ---
    bbox = detect_objects_in_original(orig, debug_dir=debug_dir if DEBUG else None)

    # --- background removal ---
    buf = BytesIO()
    orig.save(buf, format="PNG")
    buf.seek(0)

    try:
        result = remove(
            buf.read(),
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_structure_size=6,
        )
    except Exception as e:
        print(f"[WARN] Alpha matting failed ({e}), falling back to plain remove.")
        buf.seek(0)
        result = remove(buf.read())

    rembg_obj = Image.open(BytesIO(result)).convert("RGBA")
    if DEBUG:
        debug_save(rembg_obj, os.path.join(debug_dir, "debug_rembg_post.png"))

    # --- Use the bbox from original detection to crop rembg output ---
    if bbox is None:
        print("[WARN] No objects detected, using full image")
        cropped_obj = rembg_obj
    else:
        x0, y0, x1, y1 = bbox
        cropped_obj = rembg_obj.crop((x0, y0, x1, y1))
        if DEBUG:
            debug_save(cropped_obj, os.path.join(debug_dir, "debug_cropped.png"))

    # --- create square canvas and center the cropped object ---
    w, h = cropped_obj.size
    max_side = max(w, h)
    target = math.ceil(max_side / (1 - 2 * MARGIN_RATIO))
    canvas = Image.new("RGBA", (target, target), (255, 255, 255, 0))

    # scale object to fit exactly within margins (preserve aspect ratio)
    max_obj = int(target * (1 - 2 * MARGIN_RATIO))
    scale = max_obj / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    obj_resized = cropped_obj.resize((new_w, new_h), Image.Resampling.LANCZOS)

    x_off = (target - new_w) // 2
    y_off = (target - new_h) // 2
    canvas.paste(obj_resized, (x_off, y_off), obj_resized)
    
    if DEBUG:
        debug_save(canvas.convert("RGBA"), os.path.join(debug_dir, "debug_centered.png"))

    # --- levels (RGB only) ---
    r, g, b, a = canvas.split()
    rgb = Image.merge("RGB", (r, g, b))
    rgb = apply_levels(rgb, BLACK_POINT, MIDPOINT, WHITE_POINT)
    canvas = Image.merge("RGBA", (*rgb.split(), a))

    # --- brightness + contrast ---
    r, g, b, a = canvas.split()
    rgb = Image.merge("RGB", (r, g, b))
    rgb = ImageEnhance.Brightness(rgb).enhance(BRIGHTNESS_INCREASE)
    rgb = ImageEnhance.Contrast(rgb).enhance(CONTRAST_INCREASE)
    canvas = Image.merge("RGBA", (*rgb.split(), a))

    # --- final resize if needed ---
    max_dim = max(canvas.size)
    if max_dim > MAX_SIZE:
        scale_f = MAX_SIZE / max_dim
        new_w = int(canvas.width * scale_f)
        new_h = int(canvas.height * scale_f)
        canvas = canvas.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # --- flatten on white and save ---
    final = Image.new("RGB", canvas.size, "white")
    final.paste(canvas, mask=canvas.split()[-1])
    ensure_dirs(output_path)
    final.save(output_path, quality=95)
    
    if DEBUG:
        debug_save(final, os.path.join(debug_dir, "debug_final.png"))
        print(f"Saved processed image to {output_path}")
        print(f"Debug files in folder: {debug_dir}")

if __name__ == "__main__":
    process_image("unfiltered-photos/IMG_1915.jpeg", "output.png")