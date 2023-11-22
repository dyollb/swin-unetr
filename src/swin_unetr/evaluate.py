import json
import random
from pathlib import Path
from typing import Optional

import SimpleITK as sitk
import typer

app = typer.Typer()


def compare_labels(
    mask1: sitk.Image, mask2: sitk.Image, labels: list[int]
) -> list[float]:
    calculator = sitk.LabelOverlapMeasuresImageFilter()
    calculator.Execute(mask1, mask2)

    dice: list[float] = [calculator.GetDiceCoefficient(id) for id in labels]
    return dice


@app.command()
def evaluate_dataset(dir1: Path, dir2: Path, output_file: Optional[Path] = None):
    """Compute Dice for whole dataset (hardcoded labels: 1, 2)"""
    artery_id = 1
    vein_id = 2

    # find matching files
    rows: list[tuple[str, float, float]] = []
    for f1 in dir1.glob("*.nii.gz"):
        f2 = dir2 / f1.name
        if not f2.exists():
            continue

        label1 = sitk.ReadImage(f1, sitk.sitkUInt8)
        label2 = sitk.ReadImage(f2, sitk.sitkUInt8)
        vals = compare_labels(label1, label2, [artery_id, vein_id])
        rows.append((f1.name.replace(".nii.gz", ""), vals[0], vals[1]))

    if output_file:
        with open(output_file, "w") as f:
            for n, v1, v2 in rows:
                print(f"{n}, {v1}, {v2}", file=f)
    return rows


@app.command()
def write_datalist(
    pred_dir: Path,
    label_dir: Path,
    image_dir: Path,
    root_dir: Path,
    output_file: Path,
    N: int = 65,
):
    """find top N files each for ge, siemens, philips"""
    vals = evaluate_dataset(pred_dir, label_dir, None)

    # select top N datasets (highest Dice)
    vals_ge = sorted(
        [(v1, n) for (n, v1, _) in vals if "_ge" in n], key=lambda x: x[0], reverse=True
    )[:N]
    vals_siemens = sorted(
        [(v1, n) for (n, v1, _) in vals if "siemens" in n],
        key=lambda x: x[0],
        reverse=True,
    )[:N]
    vals_philips = sorted(
        [(v1, n) for (n, v1, _) in vals if "philips" in n],
        key=lambda x: x[0],
        reverse=True,
    )[:N]

    # shuffle datasets
    r = random.Random(10918)
    r.shuffle(vals_ge)
    r.shuffle(vals_siemens)
    r.shuffle(vals_philips)

    # use 3 x 5 as test images
    test: list[str] = []
    for _, n in vals_ge[-5:] + vals_siemens[-5:] + vals_philips[-5:]:
        test.append(str(image_dir.relative_to(root_dir) / f"{n}.nii.gz"))

    # training and validation pairs, already shuffled
    pairs: list[dict[str, str]] = []
    for id in range(N - 5):
        for n in (vals_ge[id][1], vals_siemens[id][1], vals_philips[id][1]):
            fn = f"{n}.nii.gz"
            pairs.append(
                {
                    "image": str(image_dir.relative_to(root_dir) / fn),
                    "label": str(label_dir.relative_to(root_dir) / fn),
                }
            )

    # min 20% for validation
    num_training = int(0.8 * len(pairs))
    data = {
        "test": test,
        "training": pairs[:num_training],
        "validation": pairs[num_training:],
    }
    output_file.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    app()

    # python src\swin_unetr\evaluate.py evaluate-dataset C:\Users\lloyd\datasets\CC\vein_artery_all_pred C:\Users\lloyd\datasets\CC\charm_artery_veins --output-file C:\Users\lloyd\datasets\CC\vein_artery_all_pred.txt
    # python src\swin_unetr\evaluate.py write-datalist C:\Users\lloyd\datasets\CC\vein_artery_all_pred C:\Users\lloyd\datasets\CC\charm_artery_veins C:\Users\lloyd\datasets\CC\t1w_n4_1mm C:\Users\lloyd\datasets\CC C:\Users\lloyd\datasets\CC\vein_artery_best.json
