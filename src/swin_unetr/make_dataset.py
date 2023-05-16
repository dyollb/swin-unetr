from pathlib import Path

import numpy as np
import SimpleITK as sitk
import typer

app = typer.Typer()


@app.command()
def make_brain(input_dir: Path, output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    for f in input_dir.rglob("labeling.nii.gz"):
        labels = sitk.ReadImage(f, sitk.sitkUInt16)
        labels_np = sitk.GetArrayFromImage(labels)
        mask_np = np.zeros_like(labels_np)

        mask_np[labels_np == 16] = 1  # Brain-Stem
        mask_np[labels_np == 2] = 2  # Left-Cerebral-White-Matter
        mask_np[labels_np == 41] = 2  # Right-Cerebral-White-Matter
        mask_np[labels_np == 3] = 3  # Left-Cerebral-Cortex
        mask_np[labels_np == 42] = 3  # Right-Cerebral-Cortex
        mask_np[labels_np == 7] = 4  # Left-Cerebellum-White-Matter
        mask_np[labels_np == 46] = 4  # Right-Cerebellum-White-Matter
        mask_np[labels_np == 8] = 5  # Left-Cerebellum-Cortex
        mask_np[labels_np == 47] = 5  # Right-Cerebellum-Cortex
        mask_np[labels_np == 512] = 6  # Spinal-Cord
        mask_np[labels_np == 530] = 7  # Optic-Nerve
        mask_np[labels_np == 508] = 8  # Rectus-Muscles
        mask_np[labels_np == 509] = 9  # Mucosa

        mask = sitk.GetImageFromArray(mask_np)
        mask.CopyInformation(labels)

        while True:
            if input_dir.samefile(f.parent):
                break
            f = f.parent

        sitk.WriteImage(mask, output_dir / f"{f.name}.nii.gz")

    with open(output_dir / "tissue.txt", "w") as file:
        print("V7", file=file)
        print("N9", file=file)
        print("C0.8 0.1 0.1 0.5 Brain-Stem", file=file)
        print("C0.6 0.6 0.6 0.5 Cerebral-White-Matter", file=file)
        print("C0.3 0.3 0.3 0.5 Cerebral-Grey-Matter", file=file)
        print("C0.8 0.6 0.6 0.5 Cerebellum-White-Matter", file=file)
        print("C0.5 0.3 0.3 0.5 Cerebellum-Grey-Matter", file=file)
        print("C0.0 0.7 0.8 0.5 Spinal-Cord", file=file)
        print("C0.0 0.7 0.4 0.5 Optic-Nerve", file=file)
        print("C0.7 0.2 0.8 0.5 Rectus-Muscles", file=file)
        print("C1.0 0.6 0.7 0.5 Mucosa", file=file)


@app.command()
def make_vessels(input_dir: Path, output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    for f in input_dir.rglob("labeling.nii.gz"):
        labels = sitk.ReadImage(f, sitk.sitkUInt16)
        labels_np = sitk.GetArrayFromImage(labels)
        mask_np = np.zeros_like(labels_np)
        mask_np[labels_np == 502] = 1  # Artery
        mask_np[labels_np == 514] = 2  # Vein
        mask = sitk.GetImageFromArray(mask_np)
        mask.CopyInformation(labels)

        while True:
            if input_dir.samefile(f.parent):
                break
            f = f.parent

        sitk.WriteImage(mask, output_dir / f"{f.name}.nii.gz")

    with open(output_dir / "tissue.txt", "w") as file:
        print("V7", file=file)
        print("N2", file=file)
        print("C0.8 0.1 0.1 0.5 Artery", file=file)
        print("C0.1 0.1 0.8 0.5 Vein", file=file)


@app.command()
def make_head_mask(input_dir: Path, output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    for f in input_dir.rglob("labeling.nii.gz"):
        labels = sitk.ReadImage(f, sitk.sitkUInt16)
        labels_np = sitk.GetArrayFromImage(labels)
        mask_np = np.zeros_like(labels_np)
        mask_np[labels_np != 517] = 1  # Head
        mask = sitk.GetImageFromArray(mask_np)
        mask.CopyInformation(labels)

        while True:
            if input_dir.samefile(f.parent):
                break
            f = f.parent

        sitk.WriteImage(mask, output_dir / f"{f.name}.nii.gz")

    with open(output_dir / "tissue.txt", "w") as file:
        print("V7", file=file)
        print("N1", file=file)
        print("C0.8 0.1 0.1 0.5 Head", file=file)


if __name__ == "__main__":
    app()
