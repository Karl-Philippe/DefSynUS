"""Orchestrator for the full data-generation pipeline.

Runs any subset of the five pipeline stages in order:
    deform  -> deform_labelmaps.py::run_deform
    labels  -> US_renderer_labelmaps.py::run_render_labels
    images  -> US_render_only_images.py::run_render_images
    warp    -> warp_labelmap.py::run_warp
    masks   -> mask_generation_multilabel.py::run_mask_generation

Example:
    python generate_dataset.py --dataset human
    python generate_dataset.py --dataset human --steps labels,warp,masks
    python -m data_generation.generate_dataset --steps images --config config/config_data_generation.yml
"""

import argparse
import time

from data_generation.deform_labelmaps import run_deform
from data_generation.US_renderer_labelmaps import run_render_labels, parse_hparams as parse_hparams_labels
from data_generation.US_render_only_images import run_render_images, parse_hparams as parse_hparams_images
from data_generation.warp_labelmap import run_warp
from data_generation.mask_generation_multilabel import run_mask_generation


STEP_ORDER = ["deform", "labels", "images", "warp", "masks"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["phantom", "human"], default="phantom",
                        help="Which dataset to process (default: phantom)")
    parser.add_argument("--steps", default=",".join(STEP_ORDER),
                        help=f"Comma-separated subset of {STEP_ORDER}. Default: all. Note: 'deform' is skipped for 'phantom' (labelmap is pre-refined).")
    parser.add_argument("--config", default="config/config_data_generation.yml",
                        help="Path to US-renderer YAML config (used by labels and images steps).")
    parser.add_argument("--n-fields", type=int, default=1000,
                        help="Number of deformation fields to apply in the deform step.")
    parser.add_argument("--skip-images", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip the 'images' stage (LOTUS renders US internally at training time). "
                             "Pass --no-skip-images to produce rendered US PNGs for debugging.")
    return parser.parse_args()


def main():
    args = parse_args()

    requested = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in requested if s not in STEP_ORDER]
    if unknown:
        raise SystemExit(f"Unknown steps: {unknown}. Valid: {STEP_ORDER}")
    steps = [s for s in STEP_ORDER if s in requested]

    if args.skip_images and "images" in steps:
        steps.remove("images")
        print("[images] skipped (--skip-images; LOTUS renders US internally). Use --no-skip-images to enable.")

    print(f"Dataset: {args.dataset}")
    print(f"Steps:   {steps}")
    print("=" * 60)

    # Parse hparams once if any renderer step is needed (avoids double parse)
    hparams_labels = None
    hparams_images = None
    if "labels" in steps:
        hparams_labels = parse_hparams_labels(args.config)
        hparams_labels.dataset_kind = args.dataset
    if "images" in steps:
        hparams_images = parse_hparams_images(args.config)
        hparams_images.dataset_kind = args.dataset

    for step in steps:
        t0 = time.time()
        print(f"\n[{step}] starting...")
        if step == "deform":
            if args.dataset == "phantom":
                print("[deform] skipped: phantom dataset is pre-refined and does not require deformation.")
                continue
            run_deform(dataset=args.dataset, n_fields=args.n_fields)
        elif step == "labels":
            run_render_labels(dataset=args.dataset, hparams=hparams_labels)
        elif step == "images":
            run_render_images(dataset=args.dataset, hparams=hparams_images)
        elif step == "warp":
            run_warp(dataset=args.dataset)
        elif step == "masks":
            run_mask_generation(dataset=args.dataset)
        print(f"[{step}] done in {time.time() - t0:.1f}s")

    print("\n" + "=" * 60)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
