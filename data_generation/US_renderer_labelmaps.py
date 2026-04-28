import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import configargparse
from PIL import Image

from data_generation.dataloaders.ct_3d_labemaps_dataset import CT3DLabelmapDataLoader
from utils.configargparse_arguments import build_configargparser


def parse_hparams(config_path=None):
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    parser.add('-c', is_config_file=True, help='config file path')
    cli_args = ['-c', config_path] if config_path else None
    parser, hparams = build_configargparser(parser, args=cli_args)
    hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return hparams


def save_labels_images(data, label_path):
    data_np = data.squeeze().cpu().numpy()
    if data_np.ndim == 3 and data_np.shape[-1] == 1:
        data_np = data_np.squeeze(-1)
    labelmap = data_np.astype(np.uint8)
    Image.fromarray(labelmap).save(label_path, format='PNG')


def run_render_labels(dataset="phantom", hparams=None, config_path=None, manual_seed=True):
    if hparams is None:
        hparams = parse_hparams(config_path)

    hparams.inference_output_dir = f"CT_label_maps_{dataset}"

    if manual_seed:
        torch.manual_seed(2024)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    print(f"Loading CT data from {hparams.base_folder_data_path}")
    ct_dataloader = CT3DLabelmapDataLoader(hparams)
    _, _, val_dataset = ct_dataloader.train_dataloader()
    val_loader = ct_dataloader.val_dataloader()
    print(f"Loaded {len(val_dataset)} CT samples.")

    labels_dir = os.path.join(hparams.inference_output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    label_images = []
    saved_images = 12

    for i, inputs in enumerate(val_loader):
        data_tensor, label_tensor, metadata = inputs
        data_tensor = data_tensor.to(hparams.device)
        label_tensor = label_tensor.to(hparams.device)

        data_tensor = torch.rot90(data_tensor, k=0, dims=[2, 3])
        data_tensor = torch.rot90(data_tensor, k=-1, dims=[1, 2])

        label_path = os.path.join(labels_dir, f"labels_{i}.png")
        save_labels_images(data_tensor, label_path=label_path)
        print(f"Label saved at {label_path}")

        if i + 1 <= saved_images:
            label_images.append(data_tensor.squeeze().cpu().numpy())

        if i + 1 == saved_images:
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            for j, ax in enumerate(axes.flatten()):
                if len(label_images) > j:
                    ax.imshow(label_images[j], cmap="viridis")
                    ax.set_title(f"Label {j + 1}")
                ax.axis('off')
            plt.tight_layout()
            figure_path = os.path.join(hparams.inference_output_dir, "Labelmaps_1_to_12.png")
            plt.savefig(figure_path)
            print(f"Saved labelmaps figure as {figure_path}")


if __name__ == "__main__":
    run_render_labels(dataset="phantom")
