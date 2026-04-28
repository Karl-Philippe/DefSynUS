Use this folder (or `slicer_modules/DefSynUSSequenceInference`) as the Slicer "Additional module path".

Do not add the repository root (`lotus_clean/`) as a Slicer module path.

Why:
- Slicer attempts to load every `.py` file in a module path as a scripted module.
- This repo contains training/inference scripts (`train.py`, `inference.py`, etc.) that are not Slicer modules.
- Some of those scripts import `monai`, which is currently incompatible with your Slicer Python 3.12 environment.

Recommended Slicer path:
- `/home/kpbeaudet/Documents/Projects/defsynUS/lotus_clean/slicer_modules/DefSynUSSequenceInference`
