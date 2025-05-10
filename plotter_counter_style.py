import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

def create_contour_overlay(original_img, heatmap, levels=6, alpha=0.5):
    """
    Creates a contour-style heatmap overlay (like in medical image papers).
    
    original_img: PIL.Image
    heatmap: 2D numpy array (same aspect ratio as image)
    Returns: Numpy array (RGB image with overlay)
    """
    # Convert PIL image to numpy
    original_np = np.array(original_img)
    h, w = original_np.shape[:2]

    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Create the figure with correct size
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # fill entire figure
    ax.imshow(original_np)
    norm = Normalize(vmin=0, vmax=1)

    # Create contour overlay
    ax.contourf(heatmap_resized, levels=levels, cmap='jet', alpha=alpha, norm=norm)
    ax.axis('off')

    # Draw and extract image from canvas
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)
    overlay_img = Image.open(buf).convert('RGB')
    plt.close(fig)

    return np.array(overlay_img)
