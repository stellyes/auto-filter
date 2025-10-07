# Product Image Background Removal Script

A Python script designed to automatically remove white/grey backgrounds from product photography, crop images to a square format with precise margins, and apply professional color adjustments.

## Overview

This script processes product images by:
- Removing white and grey backgrounds while preserving object details
- Cropping to a square format with objects centered and 7.6% margins
- Applying brightness (+7%) and contrast (+16%) enhancements
- Adjusting levels (white point: 219, midpoint: 1, black point: unchanged)
- Outputting clean, professional product images with white backgrounds

## Features

- **Intelligent Background Removal**: Uses dual-threshold approach with bilateral filtering to detect and remove light backgrounds while preserving object edges
- **Edge Preservation**: Employs convex hull technique to maintain complete object shapes without cutting into edges
- **Noise Filtering**: Removes scattered gray noise and artifacts commonly found in photography backgrounds
- **Precise Framing**: Centers objects with exact 7.6% margins on all sides in a square format
- **Color Enhancement**: Applies professional-grade brightness, contrast, and levels adjustments
- **High Quality Output**: Saves images as JPEG with 95% quality or your preferred format

## Requirements

```bash
pip install opencv-python numpy pillow
```

### Dependencies
- Python 3.6+
- OpenCV (cv2) - Image processing and mask creation
- NumPy - Array operations
- Pillow (PIL) - Color adjustments and image saving

## Installation

1. Clone or download this repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Alternatively, install packages individually:
```bash
pip install opencv-python
pip install numpy
pip install pillow
```

## Usage

### Basic Usage

```bash
python run.py
```

### Command Line Arguments

- `input` (required): Path to the input image file
- `output` (required): Path to save the processed image

### Examples

Process a single image:
```bash
python script.py unfiltered-photos/product1.jpg filtered-photos/product1_filtered.jpg
```

Batch processing with a shell script:
```bash
for file in unfiltered-photos/*.jpg; do
    filename=$(basename "$file")
    python script.py "$file" "filtered-photos/${filename%.*}_filtered.jpg"
done
```

Batch processing with Python:
```python
import os
from script import remove_background

input_dir = "unfiltered-photos"
output_dir = "filtered-photos"

for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_filtered.jpg")
        remove_background(input_path, output_path)
```

## How It Works

### 1. Background Detection
- Converts image to grayscale and applies bilateral filtering to preserve edges
- Uses dual thresholding (200 and 170 brightness levels) to identify background pixels
- Combines thresholds conservatively to avoid false positives

### 2. Mask Refinement
- Applies morphological operations (closing, opening) to remove noise
- Filters out small contours below 1500 pixels to eliminate scattered artifacts
- Uses convex hull technique to reconstruct clean object boundaries

### 3. Object Framing
- Calculates bounding box encompassing all detected objects
- Determines square crop size based on 7.6% margin requirement
- Centers objects within the square frame

### 4. Background Replacement
- Replaces masked background areas with pure white (RGB: 255, 255, 255)
- Post-processes to convert any remaining gray pixels (brightness > 200) to white

### 5. Color Adjustments
- Increases brightness by 7%
- Increases contrast by 16%
- Applies levels adjustment with white point at 219

### 6. Output
- Saves as high-quality JPEG (95% quality) or specified format

## Technical Specifications

### Image Processing Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Margin Ratio | 7.6% | Space around objects in frame |
| Object Ratio | 84.8% | Percentage of frame occupied by objects |
| Threshold 1 | 200 | Primary background detection level |
| Threshold 2 | 170 | Secondary background detection level |
| Min Contour Area | 1500 px | Noise filtering threshold |
| Brightness Increase | +7% | Color enhancement |
| Contrast Increase | +16% | Color enhancement |
| White Point | 219 | Levels adjustment |
| JPEG Quality | 95% | Output quality setting |

### Morphological Operations

- **Bilateral Filter**: 9x9 kernel, sigma: 75
- **Close Operations**: 3x3 kernel (3 iterations), 9x9 kernel (2 iterations)
- **Open Operations**: 5x5 kernel (2 iterations)
- **Erosion/Dilation**: 3x3 kernel (1 iteration each)

## Troubleshooting

### Issue: Parts of the object are being removed

**Solution**: The threshold may be too aggressive for darker objects. Try adjusting:
- Increase `min_contour_area` if small parts are being filtered as noise
- Adjust threshold values (170 and 200) if object colors are too close to background

### Issue: Background noise remains in the output

**Solution**: Increase noise filtering:
- Raise `min_contour_area` value (currently 1500)
- Adjust the post-processing threshold from 200 to a lower value like 190

### Issue: Objects are too small or too large in frame

**Solution**: Adjust the margin ratio:
- Change `margin_ratio = 0.076` to a different value (e.g., 0.05 for 5% margins)

### Issue: Image quality is degraded

**Solution**: Increase JPEG quality:
- Modify the quality parameter: `final_img.save(output_path, 'JPEG', quality=98)`

### Issue: Script fails with "No objects detected"

**Possible causes**:
- Background is not white/grey enough
- Objects are too similar in color to the background
- Image has unusual lighting conditions

**Solutions**:
- Adjust threshold values to be more permissive
- Check that input image has sufficient contrast between objects and background

## Limitations

- Designed for white/grey backgrounds only (not suitable for colored backgrounds)
- Best results with good contrast between objects and background
- May struggle with translucent or reflective objects
- Requires objects to be clearly separated from background

## Output Specifications

- **Format**: JPEG (or PNG if specified in output path)
- **Color Space**: RGB
- **Quality**: 95% JPEG compression
- **Aspect Ratio**: 1:1 (square)
- **Background**: Pure white (RGB: 255, 255, 255)

## Performance Considerations

- Processing time depends on image size and complexity
- Typical processing time: 1-3 seconds per image on modern hardware
- Memory usage scales with input image dimensions
- For batch processing, consider implementing multiprocessing for faster throughput

## Best Practices

1. **Input Images**: Use high-resolution images with good lighting and clear object separation
2. **Consistent Backgrounds**: Works best when all product images have similar background tones
3. **File Organization**: Keep input and output directories organized for batch processing
4. **Quality Control**: Always review outputs to ensure edge preservation and clean backgrounds
5. **Backup Originals**: Keep original images before processing

## License

This script is provided as-is for product photography processing. Modify and distribute as needed for your use case.

## Support

For issues, questions, or feature requests, please refer to the troubleshooting section or modify the script parameters to suit your specific needs.

## Version History

- **v1.0**: Initial release with background removal and color adjustments
- **v1.1**: Added edge preservation with convex hull technique
- **v1.2**: Improved noise filtering and dual-threshold approach
- **v1.3**: Fine-tuned brightness/contrast adjustments (+7%/+16%)

## Credits

Built using OpenCV, NumPy, and Pillow libraries for robust image processing capabilities.