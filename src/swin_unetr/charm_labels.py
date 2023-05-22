import json
from pathlib import Path
from typing import Dict

import itk
import numpy as np
import SimpleITK as sitk
import typer


def map_image(
    labels: sitk.Image, label_map: np.ndarray, dissolve_label: int = -1
) -> sitk.Image:
    labels_np = sitk.GetArrayFromImage(labels)
    labels_np[:] = label_map[labels_np[:]]

    if dissolve_label > 0:
        labels_itk = itk.image_from_array(labels_np).astype(itk.US)
        mask_np = np.zeros(labels_np.shape, dtype=int)
        mask_np[labels_np[:] == dissolve_label] = 1
        mask_itk = itk.image_from_array(mask_np).astype(itk.US)
        modified_labels = itk.dissolve_mask_image_filter(
            labels_itk, mask_image=mask_itk
        )
        labels_np = itk.array_from_image(modified_labels)

    labels_mapped = sitk.GetImageFromArray(labels_np)
    labels_mapped.CopyInformation(labels)
    return labels_mapped


def map_images(
    input_dir: Path,
    output_dir: Path,
    label_map_path: Path = Path(__file__).parent / "charm_labels.json",
):
    data = json.loads(label_map_path.read_text())
    id2name = {int(k): v for k, v in data["labels"].items()}
    name2name: Dict[str, str] = data["mapping"]
    name2id: Dict[str, int] = {"Background": 0}

    next_id = 1
    for n in name2name.values():
        if n == "undetermined":
            continue
        if n not in name2id:
            name2id[n] = next_id
            next_id += 1
    name2id["undetermined"] = next_id

    input_id_size = max(id2name.keys()) + 1
    label_map = next_id * np.ones((input_id_size,), dtype=int)
    for id in id2name.keys():
        id2 = name2id[name2name[id2name[id]]]
        label_map[id] = id2

    output_dir.mkdir(exist_ok=True, parents=True)
    for f in input_dir.rglob("labeling.nii.gz"):
        labels = sitk.ReadImage(f, sitk.sitkUInt16)
        labels = map_image(labels, label_map, dissolve_label=next_id)

        while True:
            if input_dir.samefile(f.parent):
                break
            f = f.parent
        sitk.WriteImage(labels, output_dir / f"{f.name}.nii.gz")

    name2id.pop("Background")
    name2id.pop("undetermined")
    (output_dir / "labels.json").write_text(
        json.dumps({v: k for k, v in name2id.items()}, indent=2)
    )


if __name__ == "__main__":
    typer.run(map_images)
