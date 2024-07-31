import random
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np


def generate_wrong_label_images(model, dataloader, num_images=25):
    model.eval()
    images = []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            wrong_labels = torch.randint(0, 10, (x.size(0),))
            wrong_labels = (wrong_labels + y) % 10  # Ensure labels are different
            x_hat, _, _, _, _ = model((x, wrong_labels))
            images.extend(list(zip(x, x_hat)))
            if len(images) >= num_images:
                break

    original, reconstructed = zip(*images[:num_images])

    original = list(original)
    reconstructed = list(reconstructed)

    original_grid = make_grid(original, nrow=5)
    reconstructed_grid = make_grid(reconstructed, nrow=5)
    return torch.cat([original_grid, reconstructed_grid], dim=1)


def plt_result(label, result):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Check if result is a PyTorch tensor
    if isinstance(result, torch.Tensor):
        # Convert PyTorch tensor to NumPy array
        result = result.cpu().numpy()

    # Ensure the result is in the correct format for imshow
    if result.shape[0] == 3:  # If channels are in the first dimension
        result = np.transpose(result, (1, 2, 0))

    # Normalize the image if it's not in the range [0, 1]
    if result.max() > 1.0:
        result = result / 255.0

    ax.imshow(result)
    ax.set_title(f"{label}: Original (top) vs Reconstructed with Wrong Labels (bottom)")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{label}_vae_results.png")
    plt.close()
