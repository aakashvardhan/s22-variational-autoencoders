import random
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


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
    original_grid = make_grid(original, nrow=5)
    reconstructed_grid = make_grid(reconstructed, nrow=5)
    return torch.cat([original_grid, reconstructed_grid], dim=1)


def plt_result(label, result):
    fig, ax = plt.subplots(2, 1, figsize=(15, 30))

    ax.imshow(result.permute(1, 2, 0))
    ax.set_title(f"{label} - Original vs Reconstructed")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{label}.png")
