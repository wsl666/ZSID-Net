import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from networks import RJNet
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

def visualize_feature_maps(feature_maps):
    # Check if the input tensor has the expected shape
    assert feature_maps.shape == (1, 64, 256, 256), "Expected input shape (1, 64, 256, 256)"

    # Remove the batch dimension
    feature_maps = feature_maps.squeeze(0)

    # Set the number of feature maps to display per row and column
    num_feature_maps = feature_maps.shape[0]
    num_cols = 8
    num_rows = num_feature_maps // num_cols

    # Create a figure to hold the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

    for i in range(num_feature_maps):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        ax.imshow(feature_maps[i].detach().cpu().numpy(), cmap='viridis')
        ax.axis('off')  # Hide axes ticks

    plt.tight_layout()
    plt.show()


def visualize_feature_maps1(feature_maps):
    # Check if the input tensor has the expected shape
    assert feature_maps.shape == (1, 64, 256, 256), "Expected input shape (1, 64, 256, 256)"

    # Remove the batch dimension
    feature_maps = feature_maps.squeeze(0)

    # Normalize each feature map to the range [0, 1]
    for i in range(feature_maps.shape[0]):
        feature_map = feature_maps[i]
        feature_map_min = feature_map.min()
        feature_map_max = feature_map.max()
        feature_maps[i] = (feature_map - feature_map_min) / (feature_map_max - feature_map_min)

    # Set the number of feature maps to display per row and column
    num_feature_maps = feature_maps.shape[0]
    num_cols = 8
    num_rows = num_feature_maps // num_cols if num_feature_maps % num_cols == 0 else (num_feature_maps // num_cols) + 1

    # Create a figure to hold the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    for i in range(num_feature_maps):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]

        # Create a divider for existing axes instance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("left", size="5%", pad=0.05)

        # Plot the feature map
        img = ax.imshow(feature_maps[i].detach().cpu().numpy(), cmap='viridis')
        ax.axis('on')  # Show axes ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add a colorbar to the left side
        cbar = fig.colorbar(img, cax=cax, orientation='vertical')
        cax.yaxis.set_ticks_position('left')
        cbar.ax.tick_params(labelsize=6)

        # save_dir = "feature_maps"

        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # plt.savefig(os.path.join(save_dir, f'feature_map_{i}.png'))

    # Hide remaining subplots if any
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()



def visualize_feature_maps2(feature_maps):
    # Check if the input tensor has the expected shape
    assert feature_maps.shape == (1, 64, 256, 256), "Expected input shape (1, 64, 256, 256)"

    # Remove the batch dimension
    feature_maps = feature_maps.squeeze(0)

    # Normalize each feature map to the range [0, 1]
    normalized_feature_maps = []
    for i in range(feature_maps.shape[0]):
        feature_map = feature_maps[i]
        feature_map_min = feature_map.min()
        feature_map_max = feature_map.max()
        normalized_feature_map = (feature_map - feature_map_min) / (feature_map_max - feature_map_min)
        normalized_feature_maps.append(normalized_feature_map)

    # Display each feature map interactively
    for i, normalized_feature_map in enumerate(normalized_feature_maps):
        # Create a new figure for each feature map
        fig, ax = plt.subplots(figsize=(5, 5))

        # Create a divider for existing axes instance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("left", size="5%", pad=0.05)

        # Plot the feature map
        img = ax.imshow(normalized_feature_map.detach().cpu().numpy(), cmap='viridis')
        ax.axis('on')  # Show axes ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add a colorbar to the left side
        cbar = fig.colorbar(img, cax=cax, orientation='vertical')
        cax.yaxis.set_ticks_position('left')
        cbar.ax.tick_params(labelsize=6)

        plt.tight_layout()
        plt.show()

        # Wait for user input to proceed to the next image
        # input("Press Enter to show the next feature map...")


if __name__ == "__main__":
    JNet = RJNet.JNet().cuda()
    JNet.load_state_dict(torch.load("checkpoints/T_models/real_T.pth"))
    # 读取图像
    image_path = "datasets/GT_146.png"
    image = Image.open(image_path)

    # 使用transforms.ToTensor()将图像转换为张量，并进行归一化
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image)
    image_tensor = torch.unsqueeze(image_tensor, dim=0).cuda()

    _,feature_maps = JNet(image_tensor)

    # Example usage:
    # Create a random tensor with the shape (1, 64, 256, 256)
    # feature_maps = torch.randn(1, 64, 256, 256)

    # Visualize the feature maps
    visualize_feature_maps1(feature_maps)
    visualize_feature_maps2(feature_maps)