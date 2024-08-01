import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_wrong_label(correct_label: int, num_classes: int = 10) -> int:
    """Generate a random wrong label."""
    wrong_label = torch.randint(0, num_classes, (1,))[0].item()
    while wrong_label == correct_label:
        wrong_label = torch.randint(0, num_classes, (1,))[0].item()
    return wrong_label


def generate_prediction(
    model: torch.nn.Module, img: torch.Tensor, wrong_label: int
) -> torch.Tensor:
    """Generate a prediction using the model."""
    input_img = img.unsqueeze(0), torch.tensor([wrong_label]).to(img.device)
    with torch.no_grad():
        pred = model(input_img)[0]
    return pred.squeeze(0)  # Remove batch dimension


def plot_image(ax: plt.Axes, img: np.ndarray, title: str):
    """Plot a single image."""
    if img.ndim == 3 and img.shape[0] in [1, 3]:  # If image is (C, H, W)
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 2:  # If image is already (H, W)
        pass
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    if img.shape[-1] == 1:  # If it's a grayscale image
        img = img.squeeze()
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)

    ax.set_title(title)
    ax.axis("off")


def get_label_text(label: int, wrong_label: int, is_mnist: bool) -> str:
    """Get the label text for the plot title."""
    if is_mnist:
        return f"Label: {label} Pred: {wrong_label}"
    return f"Label: {CIFAR10_CLASSES[label]} Pred: {CIFAR10_CLASSES[wrong_label]}"


def generate_wrong_label_images(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    is_mnist: bool,
    num_images: int = 25,
) -> None:
    """Generate and save images with wrong labels."""
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to("cpu"), labels.to("cpu")

    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    axes = axes.flatten()

    for i in range(num_images):
        img = images[i]
        label = labels[i].item()
        wrong_label = get_wrong_label(label)

        pred_img = generate_prediction(model, img, wrong_label)
        # print(f"pred_img shape: {pred_img.shape}")  # Debug print
        title = get_label_text(label, wrong_label, is_mnist)

        plot_image(axes[i], pred_img.cpu().numpy(), title)

    plt.tight_layout()
    plt.savefig("wrong_label_images.png")
    print("Saved wrong label images to wrong_label_images.png")


# # Usage
# if __name__ == "__main__":
#     # Assuming you have your model and dataloader ready
#     # generate_wrong_label_images(model, dataloader, is_mnist=True)  # For MNIST
#     # generate_wrong_label_images(model, dataloader, is_mnist=False)  # For CIFAR10
