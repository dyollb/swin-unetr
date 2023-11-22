import json
from pathlib import Path

import itk
import numpy as np
import SimpleITK as sitk
import typer

app = typer.Typer()


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


@app.command()
def map_images(
    input_dir: Path,
    output_dir: Path,
    label_map_path: Path = Path(__file__).parent / "charm_labels.json",
):
    data = json.loads(label_map_path.read_text())
    id2name = {int(k): v for k, v in data["labels"].items()}
    name2name: dict[str, str] = data["mapping"]
    name2id: dict[str, int] = {"Background": 0}

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


@app.command()
def modify_bones(
    input_labels_dir: Path,
    input_skull_dir: Path,
    output_dir: Path,
    label_map_path: Path,
):
    data = json.loads(label_map_path.read_text())
    name2id = {v: int(k) for k, v in data.items()}

    other_tissue_id = name2id["Other-Tissues"]
    cancellous_id = name2id["Bone-Cancellous"]
    cortical_id = name2id["Bone-Cortical"]

    output_dir.mkdir(exist_ok=True, parents=True)
    for f in input_labels_dir.glob("*.nii.gz"):
        if not (input_skull_dir / f.name).exists():
            print(f"WARNING: {f.name} skull mask not available")
            # shutil.copyfile(f, input_skull_dir / f.name)
            continue
        labels = sitk.ReadImage(f, sitk.sitkUInt16)
        labels_np = sitk.GetArrayFromImage(labels)
        bones = sitk.ReadImage(input_skull_dir / f.name, sitk.sitkUInt16)
        bones = sitk.Resample(
            bones,
            labels,
            interpolator=sitk.sitkNearestNeighbor,
        )
        bones_np = sitk.GetArrayFromImage(bones)
        assert (
            labels_np.shape == bones_np.shape
        ), f"{labels_np.shape} != {bones_np.shape}"

        labels_np[labels_np == cancellous_id] = other_tissue_id
        labels_np[labels_np == cortical_id] = other_tissue_id
        bones_np[labels_np != other_tissue_id] = 0
        labels_np[bones_np == 1] = cortical_id
        labels_np[bones_np == 2] = cancellous_id

        output = sitk.GetImageFromArray(labels_np)
        output.CopyInformation(labels)
        sitk.WriteImage(output, output_dir / f.name)


if __name__ == "__main__":
    app()
