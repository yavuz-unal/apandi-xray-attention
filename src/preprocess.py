"""
Preprocessing: unsharp masking for cropped abdominal radiographs.

Cropping was performed manually and is not included in this script.
This script applies unsharp masking (σ=1.0, λ=1.5) to already-cropped images.

Usage:
    python preprocess.py --input_dir /path/to/cropped --output_dir /path/to/output
"""

import os
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def unsharp_mask(image, sigma=1.0, strength=1.5):
    """
    Apply unsharp masking to a grayscale image.

    I_sharp = I + λ * (I - G_σ * I)
    """
    img_array = np.array(image, dtype=np.float32)
    if img_array.ndim == 3:
        # Process each channel separately
        sharpened = np.zeros_like(img_array)
        for c in range(img_array.shape[2]):
            blurred = gaussian_filter(img_array[:, :, c], sigma=sigma)
            sharpened[:, :, c] = img_array[:, :, c] + strength * (img_array[:, :, c] - blurred)
    else:
        blurred = gaussian_filter(img_array, sigma=sigma)
        sharpened = img_array + strength * (img_array - blurred)

    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return Image.fromarray(sharpened)


def process_directory(input_dir, output_dir, sigma=1.0, strength=1.5):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f'Processing {len(files)} images from {input_dir}')
    for i, fname in enumerate(files):
        img = Image.open(os.path.join(input_dir, fname))
        sharpened = unsharp_mask(img, sigma=sigma, strength=strength)
        sharpened.save(os.path.join(output_dir, fname))
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(files)} processed')

    print(f'Done. Output saved to {output_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing cropped images.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save unsharp-masked images.')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Gaussian blur standard deviation (default: 1.0).')
    parser.add_argument('--strength', type=float, default=1.5,
                        help='Sharpening strength (default: 1.5).')
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir,
                      sigma=args.sigma, strength=args.strength)


if __name__ == '__main__':
    main()
