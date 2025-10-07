import cv2
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path

# Text color values
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m" # Resets all formatting

# Styles
BOLD = "\033[1m"
UNDERLINE = "\033[4m"

def remove_background(image_path, output_path):
    """
    Remove white/grey background, crop to square with margins, and apply adjustments.
    Produces clean output with white background.
    
    Args:
        image_path: Path to input image
        output_path: Path to save processed image
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f">>> {RED}{BOLD}FILE READ ERROR:{RESET}\
                         \n>>> Could not read image from {UNDERLINE}{image_path}{RESET}")
    
    # Convert to RGB for PIL processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for mask creation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to preserve edges while removing noise
    gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Use multiple thresholding approaches and combine
    # Approach 1: Simple threshold for bright backgrounds
    _, mask1 = cv2.threshold(gray_filtered, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Approach 2: Detect near-white pixels more aggressively
    # This catches light gray background noise
    _, mask2 = cv2.threshold(gray_filtered, 170, 255, cv2.THRESH_BINARY_INV)
    
    # Combine masks - be conservative (only keep clearly non-background pixels)
    mask = cv2.bitwise_and(mask1, mask2)
    
    # Morphological operations focused on noise removal without eroding object edges
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    kernel_large = np.ones((9, 9), np.uint8)
    
    # Close small gaps in objects
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=3)
    
    # Remove small scattered noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_medium, iterations=2)
    
    # Close larger gaps to solidify objects
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
    # Light erosion-dilation to smooth edges without losing detail
    mask = cv2.erode(mask, kernel_small, iterations=1)
    mask = cv2.dilate(mask, kernel_small, iterations=1)
    
    # Find contours of objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError(">>> {RED}{BOLD}IMAGE PROCESSING ERROR:{RESET}\
                         \n>>> No objects detected in image")
    
    # Filter out very small contours (noise)
    min_contour_area = 1500  # Balanced threshold
    contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    
    if not contours:
        raise ValueError(">>> {RED}{BOLD}IMAGE PROCESSING ERROR:{RESET}\
                        \n>>> No significant objects detected in image")
    
    # Create a clean mask with only the largest valid contours
    # Use convex hull to preserve object edges and fill in any gaps
    clean_mask = np.zeros_like(mask)
    for contour in contours:
        # Get convex hull to preserve object shape
        hull = cv2.convexHull(contour)
        cv2.drawContours(clean_mask, [hull], -1, 255, -1)
    
    # Use the clean mask instead of the noisy one
    mask = clean_mask
    
    # Get bounding box that encompasses all objects
    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # Add small padding to ensure we capture full objects
    padding = 5
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)
    
    # Calculate square crop with 7.6% margin
    margin_ratio = 0.076
    object_ratio = 1 - 2 * margin_ratio  # 0.848
    
    # Determine the size needed for square crop
    max_dim = max(w, h)
    square_size = int(max_dim / object_ratio)
    
    # Calculate center of objects
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Calculate crop coordinates (centered)
    crop_x1 = center_x - square_size // 2
    crop_y1 = center_y - square_size // 2
    crop_x2 = crop_x1 + square_size
    crop_y2 = crop_y1 + square_size
    
    # Create white canvas for final image
    canvas = np.ones((square_size, square_size, 3), dtype=np.uint8) * 255
    
    # Calculate paste position
    paste_x = 0
    paste_y = 0
    src_x1 = crop_x1
    src_y1 = crop_y1
    src_x2 = crop_x2
    src_y2 = crop_y2
    
    # Adjust if crop extends beyond image boundaries
    if crop_x1 < 0:
        paste_x = -crop_x1
        src_x1 = 0
    if crop_y1 < 0:
        paste_y = -crop_y1
        src_y1 = 0
    if crop_x2 > img.shape[1]:
        src_x2 = img.shape[1]
    if crop_y2 > img.shape[0]:
        src_y2 = img.shape[0]
    
    # Calculate dimensions
    src_w = src_x2 - src_x1
    src_h = src_y2 - src_y1
    
    # Extract source region
    source_region = img_rgb[src_y1:src_y2, src_x1:src_x2]
    mask_region = mask[src_y1:src_y2, src_x1:src_x2]
    
    # Apply mask to source region
    mask_3channel = cv2.cvtColor(mask_region, cv2.COLOR_GRAY2RGB) / 255.0
    masked_source = (source_region * mask_3channel).astype(np.uint8)
    
    # Create white background for masked areas
    white_bg = np.ones_like(source_region) * 255
    final_region = (masked_source + white_bg * (1 - mask_3channel)).astype(np.uint8)
    
    # Additional cleanup: replace any remaining gray pixels with pure white
    # Convert to grayscale to detect gray areas
    final_gray = cv2.cvtColor(final_region, cv2.COLOR_RGB2GRAY)
    # Any pixel brighter than 200 gets set to white
    gray_pixels = final_gray > 200
    final_region[gray_pixels] = [255, 255, 255]
    
    # Paste onto canvas
    canvas[paste_y:paste_y+src_h, paste_x:paste_x+src_w] = final_region
    
    # Convert to PIL for adjustments
    pil_img = Image.fromarray(canvas)
    
    # Increase brightness by 7% (5% + 2%)
    brightness_enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = brightness_enhancer.enhance(1.07)
    
    # Increase contrast by 16% (14% + 2%)
    contrast_enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = contrast_enhancer.enhance(1.16)
    
    # Apply levels adjustment: white point = 219, midpoint = 1, black point unchanged
    img_array = np.array(pil_img).astype(np.float32)
    
    # Apply white point adjustment (map 219 to 255)
    # Formula: output = input * (255 / white_point)
    img_array = np.clip(img_array * (255.0 / 219.0), 0, 255)
    
    # Convert back to uint8
    result = img_array.astype(np.uint8)
    final_img = Image.fromarray(result)
    
    # Save result as JPEG with high quality
    if output_path.lower().endswith(('.jpg', '.jpeg')):
        final_img.save("filtered-photos/" + output_path, 'JPEG', quality=95)
    else:
        final_img.save("filtered-photos/" + output_path)

    print(f">>> {GREEN}{BOLD}{image_path}{RESET} successfully processed, image saved as {BOLD}{output_path}{RESET}")

if __name__ == "__main__":

    # Iterate through files in unfiltered-photos/ directory
    for filepath in Path('unfiltered-photos').iterdir():
        if filepath.is_file():
            remove_background("unfiltered-photos/" + filepath.name, filepath.name)
