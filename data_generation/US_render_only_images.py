import os
import torch
import numpy as np
import configargparse
from PIL import Image

from models.us_rendering_model import UltrasoundRendering
from utils.configargparse_arguments import build_configargparser


def parse_hparams(config_path=None):
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    parser.add('-c', is_config_file=True, help='config file path')
    cli_args = ['-c', config_path] if config_path else None
    parser, hparams = build_configargparser(parser, args=cli_args)
    return hparams


def load_mask_as_tensor(mask_path, device):
    img = Image.open(mask_path).convert("L")
    arr = np.array(img, dtype=np.int64)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def save_us_image(tensor, path):
    arr = tensor.squeeze().detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
    Image.fromarray(arr.astype(np.uint8)).save(path, format="PNG")


def run_render_images(dataset="phantom", hparams=None, config_path=None):
    if hparams is None:
        hparams = parse_hparams(config_path)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mask_dir = f"CT_label_maps_{dataset}/labels"
    output_dir = f"CT_label_maps_{dataset}/images"
    os.makedirs(output_dir, exist_ok=True)

    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]
    print(f"Processing {len(mask_files)} masks from {mask_dir}")

    model = UltrasoundRendering(params=hparams).to(device).eval()

    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask_tensor = load_mask_as_tensor(mask_path, device).long()
        mask_tensor = torch.rot90(mask_tensor, k=1, dims=[2, 3])

        with torch.no_grad():
            us_img = model(mask_tensor.squeeze())

        base_name = os.path.splitext(mask_file)[0]
        out_path = os.path.join(output_dir, f"{base_name.replace('labels', 'images')}.png")
        save_us_image(us_img, out_path)
        print(f"Saved {out_path}")

    print(f"Ultrasound images generated in {output_dir}")


if __name__ == "__main__":
    run_render_images(dataset="phantom")
