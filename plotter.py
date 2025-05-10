import matplotlib.pyplot as plt



def plot_gradcam_grid(images, heatmaps, overlays, save_path="gradcam_grid.png"):
    """
    images: List of original images (PIL or NumPy RGB)
    heatmaps: List of heatmap-only images (NumPy)
    overlays: List of overlayed images (NumPy)
    """
    num_images = len(images)
    fig, axs = plt.subplots(3, num_images, figsize=(3 * num_images, 9))

    for i in range(num_images):
        axs[0, i].imshow(images[i])
        axs[0, i].axis('off')
        if i == 0:
            axs[0, i].set_ylabel("Original Image", fontsize=14)

        axs[1, i].imshow(heatmaps[i])
        axs[1, i].axis('off')
        if i == 0:
            axs[1, i].set_ylabel("Heatmap Only", fontsize=14)

        axs[2, i].imshow(overlays[i])
        axs[2, i].axis('off')
        if i == 0:
            axs[2, i].set_ylabel("Overlaid", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
