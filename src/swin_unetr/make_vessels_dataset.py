from pathlib import Path

import numpy as np
import SimpleITK as sitk
import typer


def main(input_dir: Path, output_dir: Path):
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


if __name__ == "__main__":
    typer.run(main)
