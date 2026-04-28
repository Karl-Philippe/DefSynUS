import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ==========================
# Warp label function
# ==========================
def warp_label(inputImage, original_label_image):
    resultWidth = 290
    resultHeight = 200
    centerX = resultWidth / 2
    centerY = -50
    maxAngle = np.pi * 35 / 180
    minAngle = -maxAngle
    minRadius = 61
    maxRadius = 250

    # Ensure input is 2D: [1, 1, H, W]
    inputImage = inputImage.squeeze()  # Remove unnecessary dimensions
    if inputImage.ndim == 2:
        inputImage = inputImage.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    h, w = inputImage.shape[-2:]

    x = torch.linspace(0, resultWidth - 1, resultWidth) - centerX
    y = torch.linspace(0, resultHeight - 1, resultHeight) - centerY
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    angle = torch.atan2(xx, yy)
    radius = torch.sqrt(xx ** 2 + yy ** 2)

    angle_mask = (angle > minAngle) & (angle < maxAngle)
    radius_mask = (radius > minRadius) & (radius < maxRadius)

    origCol = (angle - minAngle) / (maxAngle - minAngle) * (w - 1)
    origRow = (radius - minRadius) / (maxRadius - minRadius) * (h - 1)

    origCol = 2 * (origCol / (w - 1)) - 1
    origRow = 2 * (origRow / (h - 1)) - 1

    grid = torch.stack([origCol, origRow], dim=-1).unsqueeze(0)

    resultImage = F.grid_sample(inputImage.float(), grid, mode='nearest', align_corners=True)
    mask = (angle_mask & radius_mask).float().unsqueeze(0).unsqueeze(0)
    resultImage = resultImage * mask

    # Resize to (256,256)
    resultImage_resized = F.interpolate(resultImage, size=(256, 256), mode='nearest')

    unique_labels = torch.unique(original_label_image)
    unique_labels_with_zero = torch.cat((torch.tensor([0]), unique_labels))

    resultImage_np = resultImage_resized.squeeze().numpy()
    remapped_result_image_np = np.zeros_like(resultImage_np, dtype=np.int64)

    for i in range(resultImage_np.shape[0]):
        for j in range(resultImage_np.shape[1]):
            pixel_value = resultImage_np[i, j]
            if pixel_value not in unique_labels_with_zero.numpy():
                distances = np.abs(unique_labels_with_zero.numpy() - pixel_value)
                nearest_label = unique_labels_with_zero[torch.argmin(torch.tensor(distances))].item()
                remapped_result_image_np[i, j] = nearest_label
            else:
                remapped_result_image_np[i, j] = pixel_value

    return torch.tensor(remapped_result_image_np, dtype=torch.int64)

# ==========================
# Save label as image
# ==========================
def save_label(label_tensor, save_path):
    labelmap = label_tensor.numpy().astype(np.uint8)
    image = Image.fromarray(labelmap)
    image.save(save_path, format='PNG')

# ==========================
# Main loop
# ==========================
def warp_labels_from_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    label_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.tif', '.jpg'))]

    for f in label_files:
        # Load label map
        path = os.path.join(input_folder, f)
        label_image = Image.open(path)
        label_tensor = torch.tensor(np.array(label_image), dtype=torch.int64)

        # Warp the label
        warped_label = warp_label(label_tensor, label_tensor)

        # Save warped label
        save_path = os.path.join(output_folder, f"warped_{f}")
        save_label(warped_label, save_path)
        print(f"Saved warped label: {save_path}")

# ==========================
# Run
# ==========================
def run_warp(dataset="phantom"):
    input_folder = f"CT_label_maps_{dataset}/labels"
    output_folder = f"CT_label_maps_{dataset}/labels_warped"
    warp_labels_from_folder(input_folder, output_folder)


if __name__ == "__main__":
    run_warp(dataset="phantom")
