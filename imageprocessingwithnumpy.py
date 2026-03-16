"""
Image Processing with NumPy
============================
Covers: loading, transformations, filters, color ops, and more.
Dependencies: numpy, Pillow (pip install numpy Pillow)
"""

import numpy as np
from PIL import Image
import os


# ─────────────────────────────────────────────
# 1. LOAD & SAVE
# ─────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load an image as a NumPy array (H, W, C) uint8."""
    return np.array(Image.open(path))


def save_image(array: np.ndarray, path: str) -> None:
    """Save a NumPy array as an image file."""
    img = array.clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def create_sample_image(height: int = 256, width: int = 256) -> np.ndarray:
    """Create a colorful test image (gradient + shapes)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Horizontal red gradient
    img[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
    # Vertical green gradient
    img[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)
    # Blue rectangle
    img[60:120, 60:120, 2] = 200
    return img


# ─────────────────────────────────────────────
# 2. CHANNEL OPERATIONS
# ─────────────────────────────────────────────

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale using luminosity weights."""
    weights = np.array([0.2989, 0.5870, 0.1140])
    return (img[..., :3] @ weights).astype(np.uint8)


def split_channels(img: np.ndarray):
    """Split into R, G, B arrays."""
    return img[:, :, 0], img[:, :, 1], img[:, :, 2]


def swap_channels(img: np.ndarray, order=(2, 1, 0)) -> np.ndarray:
    """Reorder channels — e.g. RGB → BGR."""
    return img[:, :, list(order)]


# ─────────────────────────────────────────────
# 3. GEOMETRIC TRANSFORMATIONS
# ─────────────────────────────────────────────

def flip_horizontal(img: np.ndarray) -> np.ndarray:
    return img[:, ::-1]


def flip_vertical(img: np.ndarray) -> np.ndarray:
    return img[::-1, :]


def rotate_90(img: np.ndarray, k: int = 1) -> np.ndarray:
    """Rotate 90° counter-clockwise, k times (1–3)."""
    return np.rot90(img, k=k)


def crop(img: np.ndarray, y1: int, y2: int, x1: int, x2: int) -> np.ndarray:
    """Crop region [y1:y2, x1:x2]."""
    return img[y1:y2, x1:x2]


def resize_nearest(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Resize using nearest-neighbor interpolation."""
    h, w = img.shape[:2]
    row_idx = (np.arange(new_h) * h / new_h).astype(int)
    col_idx = (np.arange(new_w) * w / new_w).astype(int)
    return img[np.ix_(row_idx, col_idx)]


def pad_image(img: np.ndarray, pad: int, value: int = 0) -> np.ndarray:
    """Add border padding of constant value."""
    if img.ndim == 2:
        return np.pad(img, pad, constant_values=value)
    return np.pad(img, ((pad, pad), (pad, pad), (0, 0)), constant_values=value)


# ─────────────────────────────────────────────
# 4. PIXEL-LEVEL ADJUSTMENTS
# ─────────────────────────────────────────────

def adjust_brightness(img: np.ndarray, delta: int) -> np.ndarray:
    """Add/subtract brightness (delta ∈ -255…255)."""
    return np.clip(img.astype(int) + delta, 0, 255).astype(np.uint8)


def adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale contrast around mid-gray (factor > 1 increases contrast)."""
    return np.clip(127.5 + factor * (img.astype(float) - 127.5), 0, 255).astype(np.uint8)


def normalize(img: np.ndarray) -> np.ndarray:
    """Stretch pixel range to 0–255 (per-channel)."""
    out = np.zeros_like(img, dtype=np.uint8)
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(float)
        lo, hi = ch.min(), ch.max()
        if hi > lo:
            out[:, :, c] = ((ch - lo) / (hi - lo) * 255).astype(np.uint8)
    return out


def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    """Apply γ correction.  γ < 1 brightens; γ > 1 darkens."""
    table = (np.arange(256) / 255.0) ** gamma * 255
    return table[img].astype(np.uint8)


def threshold(img: np.ndarray, value: int = 128) -> np.ndarray:
    """Binary threshold on a grayscale image."""
    return (img >= value).astype(np.uint8) * 255


def invert(img: np.ndarray) -> np.ndarray:
    return 255 - img


# ─────────────────────────────────────────────
# 5. CONVOLUTION & FILTERS
# ─────────────────────────────────────────────

def convolve2d(img_channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Manual 2-D convolution on a single channel (no padding)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img_channel, ((ph, ph), (pw, pw)), mode='reflect')
    h, w = img_channel.shape
    out = np.zeros((h, w), dtype=float)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i+h, j:j+w]
    return out


def apply_filter(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a convolution kernel to all channels."""
    if img.ndim == 2:
        return np.clip(convolve2d(img.astype(float), kernel), 0, 255).astype(np.uint8)
    result = np.stack([
        np.clip(convolve2d(img[:, :, c].astype(float), kernel), 0, 255)
        for c in range(img.shape[2])
    ], axis=-1)
    return result.astype(np.uint8)


# Common kernels
KERNEL_BLUR = np.ones((3, 3)) / 9.0

KERNEL_GAUSSIAN = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=float) / 16.0

KERNEL_SHARPEN = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=float)

KERNEL_EDGE_SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=float)

KERNEL_EDGE_SOBEL_Y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=float)

KERNEL_EMBOSS = np.array([
    [-2, -1, 0],
    [-1,  1, 1],
    [ 0,  1, 2]
], dtype=float)


def edge_detection_sobel(gray: np.ndarray) -> np.ndarray:
    """Full Sobel edge map from a grayscale image."""
    gx = convolve2d(gray.astype(float), KERNEL_EDGE_SOBEL_X)
    gy = convolve2d(gray.astype(float), KERNEL_EDGE_SOBEL_Y)
    magnitude = np.sqrt(gx**2 + gy**2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# 6. MORPHOLOGICAL OPERATIONS
# ─────────────────────────────────────────────

def _sliding_window(img: np.ndarray, kernel_size: int):
    """Generate sliding windows view (uses stride tricks)."""
    from numpy.lib.stride_tricks import as_strided
    k = kernel_size
    h, w = img.shape
    shape = (h - k + 1, w - k + 1, k, k)
    strides = img.strides + img.strides
    return as_strided(img, shape=shape, strides=strides)


def dilate(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Morphological dilation on a binary/grayscale image."""
    padded = np.pad(binary, kernel_size // 2, mode='constant')
    windows = _sliding_window(padded, kernel_size)
    return windows.max(axis=(2, 3))


def erode(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Morphological erosion."""
    padded = np.pad(binary, kernel_size // 2, mode='constant', constant_values=255)
    windows = _sliding_window(padded, kernel_size)
    return windows.min(axis=(2, 3))


def opening(binary: np.ndarray, k: int = 3) -> np.ndarray:
    """Erosion then dilation (removes small bright spots)."""
    return dilate(erode(binary, k), k)


def closing(binary: np.ndarray, k: int = 3) -> np.ndarray:
    """Dilation then erosion (fills small holes)."""
    return erode(dilate(binary, k), k)


# ─────────────────────────────────────────────
# 7. HISTOGRAM & EQUALIZATION
# ─────────────────────────────────────────────

def histogram(img: np.ndarray, bins: int = 256):
    """Compute intensity histogram (grayscale or per-channel)."""
    if img.ndim == 2:
        return np.bincount(img.ravel(), minlength=bins)
    return [np.bincount(img[:, :, c].ravel(), minlength=bins) for c in range(img.shape[2])]


def equalize_histogram(gray: np.ndarray) -> np.ndarray:
    """Histogram equalization for a grayscale image."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min()
    n = gray.size
    lut = np.round((cdf - cdf_min) / (n - cdf_min) * 255).astype(np.uint8)
    return lut[gray]


# ─────────────────────────────────────────────
# 8. NOISE
# ─────────────────────────────────────────────

def add_gaussian_noise(img: np.ndarray, mean: float = 0, std: float = 25) -> np.ndarray:
    noise = np.random.normal(mean, std, img.shape)
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)


def add_salt_and_pepper(img: np.ndarray, amount: float = 0.02) -> np.ndarray:
    out = img.copy()
    n = int(amount * img.size // img.shape[-1] if img.ndim == 3 else amount * img.size)
    # Salt
    coords = tuple(np.random.randint(0, s, n) for s in img.shape[:2])
    out[coords] = 255
    # Pepper
    coords = tuple(np.random.randint(0, s, n) for s in img.shape[:2])
    out[coords] = 0
    return out


def median_filter(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Median filter — great for salt-and-pepper noise."""
    k = kernel_size
    padded = np.pad(img, ((k//2, k//2), (k//2, k//2), (0, 0)) if img.ndim == 3
                    else k//2, mode='reflect')
    h, w = img.shape[:2]
    if img.ndim == 2:
        out = np.zeros_like(img)
        for i in range(h):
            for j in range(w):
                out[i, j] = np.median(padded[i:i+k, j:j+k])
        return out.astype(np.uint8)
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        for i in range(h):
            for j in range(w):
                out[i, j, c] = np.median(padded[i:i+k, j:j+k, c])
    return out.astype(np.uint8)


# ─────────────────────────────────────────────
# 9. BLENDING & COMPOSITING
# ─────────────────────────────────────────────

def blend(img_a: np.ndarray, img_b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Linear blend: α·A + (1-α)·B"""
    return np.clip(alpha * img_a.astype(float) + (1 - alpha) * img_b.astype(float), 0, 255).astype(np.uint8)


def overlay_mask(img: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.4) -> np.ndarray:
    """Overlay a binary mask with a tint color."""
    out = img.copy().astype(float)
    color_layer = np.zeros_like(img, dtype=float)
    color_layer[mask > 0] = color
    out[mask > 0] = (1 - alpha) * out[mask > 0] + alpha * color_layer[mask > 0]
    return out.clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# 10. DEMO — runs all operations on a sample image
# ─────────────────────────────────────────────

def demo(output_dir: str = "output_images"):
    os.makedirs(output_dir, exist_ok=True)

    # Create sample image
    img = create_sample_image(256, 256)
    save_image(img, f"{output_dir}/00_original.png")
    print("✓ Original image saved.")

    # Grayscale
    gray = to_grayscale(img)
    save_image(gray, f"{output_dir}/01_grayscale.png")

    # Flip
    save_image(flip_horizontal(img), f"{output_dir}/02_flip_h.png")
    save_image(flip_vertical(img), f"{output_dir}/03_flip_v.png")

    # Rotate
    save_image(rotate_90(img, k=1), f"{output_dir}/04_rotate90.png")

    # Resize
    save_image(resize_nearest(img, 128, 128), f"{output_dir}/05_resize_128.png")

    # Brightness / contrast
    save_image(adjust_brightness(img, 50), f"{output_dir}/06_bright+50.png")
    save_image(adjust_contrast(img, 1.5), f"{output_dir}/07_contrast1.5.png")

    # Gamma
    save_image(gamma_correction(img, 0.5), f"{output_dir}/08_gamma0.5.png")

    # Invert
    save_image(invert(img), f"{output_dir}/09_invert.png")

    # Filters
    save_image(apply_filter(img, KERNEL_BLUR), f"{output_dir}/10_blur.png")
    save_image(apply_filter(img, KERNEL_GAUSSIAN), f"{output_dir}/11_gaussian.png")
    save_image(apply_filter(img, KERNEL_SHARPEN), f"{output_dir}/12_sharpen.png")
    save_image(apply_filter(img, KERNEL_EMBOSS), f"{output_dir}/13_emboss.png")

    # Edges
    edges = edge_detection_sobel(gray)
    save_image(edges, f"{output_dir}/14_edges_sobel.png")

    # Threshold
    save_image(threshold(gray, 128), f"{output_dir}/15_threshold.png")

    # Histogram equalization
    save_image(equalize_histogram(gray), f"{output_dir}/16_histeq.png")

    # Noise
    noisy = add_gaussian_noise(img, std=30)
    save_image(noisy, f"{output_dir}/17_gaussian_noise.png")
    sp = add_salt_and_pepper(img, 0.03)
    save_image(sp, f"{output_dir}/18_salt_pepper.png")
    save_image(median_filter(sp, 3), f"{output_dir}/19_median_denoised.png")

    # Morphology (on binary image)
    binary = threshold(gray, 128)
    save_image(dilate(binary, 3), f"{output_dir}/20_dilate.png")
    save_image(erode(binary, 3), f"{output_dir}/21_erode.png")

    # Normalize
    save_image(normalize(img), f"{output_dir}/22_normalized.png")

    print(f"✓ All processed images saved to '{output_dir}/'")


if __name__ == "__main__":
    demo()