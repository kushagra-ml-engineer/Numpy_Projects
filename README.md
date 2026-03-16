# Numpy_Projects
# 🖼️ Image Processing with NumPy

A pure-Python image processing library built on **NumPy** and **Pillow** — no OpenCV required. Covers everything from basic pixel manipulation to convolution filters, morphological operations, and noise removal.

---

## ✨ Features

| Module | Operations |
|---|---|
| **Load / Save** | Read any image as NumPy array, create synthetic test images, save results |
| **Channel Ops** | Grayscale conversion (luminosity), channel split, RGB ↔ BGR swap |
| **Geometric Transforms** | Flip, rotate 90°, crop, nearest-neighbor resize, border padding |
| **Pixel Adjustments** | Brightness, contrast, normalization, gamma correction, threshold, invert |
| **Convolution & Filters** | Blur, Gaussian, sharpen, Sobel edge detection, emboss |
| **Morphological Ops** | Dilate, erode, opening, closing (stride-trick based, no OpenCV) |
| **Histogram** | Per-channel histograms + histogram equalization |
| **Noise** | Gaussian noise, salt-and-pepper noise, median filter denoising |
| **Blending** | Alpha blend two images, tinted binary mask overlay |

---

## 📦 Installation

**Clone the repo:**
```bash
git clone https://github.com/your-username/image-processing-numpy.git
cd image-processing-numpy
```

**Install dependencies:**
```bash
pip install numpy Pillow
```

> Python 3.8+ recommended.

---

## 🚀 Quick Start

```python
from image_processing_numpy import (
    load_image, save_image,
    to_grayscale, adjust_brightness,
    apply_filter, KERNEL_GAUSSIAN,
    edge_detection_sobel
)

# Load an image
img = load_image("photo.jpg")

# Convert to grayscale
gray = to_grayscale(img)

# Brighten and apply Gaussian blur
bright = adjust_brightness(img, delta=40)
blurred = apply_filter(img, KERNEL_GAUSSIAN)

# Detect edges
edges = edge_detection_sobel(gray)

# Save results
save_image(edges, "edges.png")
```

---

## 🧪 Run the Demo

Runs every operation on a generated test image and saves ~22 result PNGs to `output_images/`:

```bash
python image_processing_numpy.py
```

**Sample outputs generated:**

```
output_images/
├── 00_original.png
├── 01_grayscale.png
├── 02_flip_h.png
├── 03_flip_v.png
├── 04_rotate90.png
├── 05_resize_128.png
├── 06_bright+50.png
├── 07_contrast1.5.png
├── 08_gamma0.5.png
├── 09_invert.png
├── 10_blur.png
├── 11_gaussian.png
├── 12_sharpen.png
├── 13_emboss.png
├── 14_edges_sobel.png
├── 15_threshold.png
├── 16_histeq.png
├── 17_gaussian_noise.png
├── 18_salt_pepper.png
├── 19_median_denoised.png
├── 20_dilate.png
├── 21_erode.png
└── 22_normalized.png
```

---

## 📖 API Reference

### Load & Save

```python
load_image(path)               # → ndarray (H, W, C) uint8
save_image(array, path)        # Save array as image file
create_sample_image(h, w)      # Generate a colorful test image
```

### Channel Operations

```python
to_grayscale(img)              # RGB → grayscale (luminosity weights)
split_channels(img)            # → (R, G, B) tuple
swap_channels(img, order)      # Reorder channels, e.g. (2,1,0) for BGR
```

### Geometric Transforms

```python
flip_horizontal(img)
flip_vertical(img)
rotate_90(img, k=1)            # k=1,2,3 rotations
crop(img, y1, y2, x1, x2)
resize_nearest(img, new_h, new_w)
pad_image(img, pad, value=0)
```

### Pixel Adjustments

```python
adjust_brightness(img, delta)  # delta ∈ -255…255
adjust_contrast(img, factor)   # factor > 1 increases contrast
normalize(img)                 # Stretch pixel range to 0–255
gamma_correction(img, gamma)   # γ < 1 brightens, γ > 1 darkens
threshold(img, value=128)      # Binary threshold (grayscale)
invert(img)                    # 255 - img
```

### Convolution & Filters

```python
apply_filter(img, kernel)      # Apply any kernel to all channels
edge_detection_sobel(gray)     # Full Sobel edge magnitude map

# Ready-made kernels:
KERNEL_BLUR
KERNEL_GAUSSIAN
KERNEL_SHARPEN
KERNEL_EDGE_SOBEL_X
KERNEL_EDGE_SOBEL_Y
KERNEL_EMBOSS
```

### Morphological Operations

```python
dilate(binary, kernel_size=3)
erode(binary, kernel_size=3)
opening(binary, k=3)           # Erode → dilate (removes small bright spots)
closing(binary, k=3)           # Dilate → erode (fills small holes)
```

### Histogram

```python
histogram(img, bins=256)       # Per-channel histogram
equalize_histogram(gray)       # Histogram equalization
```

### Noise & Denoising

```python
add_gaussian_noise(img, mean=0, std=25)
add_salt_and_pepper(img, amount=0.02)
median_filter(img, kernel_size=3)   # Great for salt-and-pepper
```

### Blending & Compositing

```python
blend(img_a, img_b, alpha=0.5)                        # α·A + (1-α)·B
overlay_mask(img, mask, color=(255,0,0), alpha=0.4)   # Tinted mask overlay
```

---

## 🗂️ Project Structure

```
image-processing-numpy/
├── image_processing_numpy.py   # Main library (all functions)
├── README.md
├── requirements.txt
└── output_images/              # Auto-created by demo()
```

**requirements.txt:**
```
numpy
Pillow
```

---

## 🧠 How It Works

All operations are implemented with **pure NumPy array math** — no hidden OpenCV or scipy calls.

- **Convolution** uses nested loops over the kernel with `np.pad` reflection padding
- **Morphological ops** use `numpy.lib.stride_tricks.as_strided` for efficient sliding windows
- **Resize** maps output pixel coordinates back to input via index scaling
- **Histogram equalization** computes CDF of pixel values and applies it as a lookup table

---

## 📋 Requirements

- Python 3.8+
- numpy >= 1.21
- Pillow >= 9.0

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙌 Contributing

Pull requests are welcome! Please open an issue first to discuss any major changes.

1. Fork the repo
2. Create your branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request
