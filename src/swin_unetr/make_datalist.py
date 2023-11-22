import json
import random
from pathlib import Path

import typer


def load_tissue_list(file_name: Path) -> dict[int, str]:
    """Load tissue list in iSEG format
    Example file with three non-background tissues:
        V7
        N3
        C0.00 0.00 1.00 0.50 Bone
        C0.00 1.00 0.00 0.50 Fat
        C1.00 0.00 0.00 0.50 Skin
    """
    tissue_label_map: dict[int, str] = {0: "Background"}
    next_id = 1
    with open(file_name) as f:
        for line in f.readlines():
            if line.startswith("C"):
                tissue = "_".join(line.strip().split(" ")[4:])
                print(tissue)
                tissue_label_map[next_id] = tissue
                next_id += 1
    return tissue_label_map


def find_matching_files(input_globs: list[Path], verbose: bool = True):
    dir_0 = Path(input_globs[0].anchor)
    glob_0 = str(input_globs[0].relative_to(dir_0))
    ext_0 = input_globs[0].name.rsplit("*")[-1]

    candidate_files = {p.name.replace(ext_0, ""): [p] for p in dir_0.glob(glob_0)}

    for other_glob in input_globs[1:]:
        dir_i = Path(other_glob.anchor)
        glob_i = str(other_glob.relative_to(dir_i))
        ext_i = other_glob.name.rsplit("*")[-1]

        for p in dir_i.glob(glob_i):
            key = p.name.replace(ext_i, "")
            if key in candidate_files:
                candidate_files[key].append(p)
            elif verbose:
                print(f"No match found for {key} : {p}")

    output_files = [v for v in candidate_files.values() if len(v) == len(input_globs)]

    if verbose:
        print(f"Number of files in {input_globs[0]}: {len(candidate_files)}")
        print(f"Number of tuples: {len(output_files)}\n")

    return output_files


def make_datalist(
    data_dir: Path = Path("C:/Users/lloyd/datasets/CC"),
    t1_dir: Path = Path("t1w_n4_1mm/*.nii.gz"),
    labels_dir: Path = Path("charm_corrected4_squeezed/*.nii.gz"),
    datalist_path: Path = Path("datalists") / "charm_corrected4_all.json",
    num_channels: int = 1,
    tissuelist_path: Path | None = None,
    percent: float = 1.0,
    test_only: bool = False,
    seed: int = 104,
) -> int:
    tissuelist = {"1": "Skull", "2": "Vertebrae"}
    if tissuelist_path is not None:
        tissue_map = load_tissue_list(tissuelist_path)
        tissue_map.pop(0)
        tissuelist = {str(id): tissue_map[id] for id in tissue_map}

    if test_only:
        input_glob = data_dir / t1_dir
        dir_0 = Path(input_glob.anchor)
        glob_0 = str(input_glob.relative_to(dir_0))

        data_config = {
            "description": "Calgary-Campinas",
            "num_channels": num_channels,
            "labels": tissuelist,
            "test": [str(f.relative_to(data_dir)) for f in dir_0.glob(glob_0)],
            "training": [],
            "validation": [],
        }
        return datalist_path.write_text(json.dumps(data_config, indent=2))

    pairs = find_matching_files([data_dir / t1_dir, data_dir / labels_dir])
    pairs = [(im.relative_to(data_dir), lbl.relative_to(data_dir)) for im, lbl in pairs]

    random.Random(seed).shuffle(pairs)
    test, pairs = pairs[:10], pairs[10:]

    num_valid = int(percent * 0.2 * len(pairs))
    num_training = len(pairs) - num_valid if percent >= 1.0 else 4 * num_valid

    data_config = {
        "description": "Calgary-Campinas",
        "num_channels": num_channels,
        "labels": tissuelist,
        "test": [str(im) for im, _ in test],
        "training": [
            {"image": str(im), "label": str(lbl)} for im, lbl in pairs[:num_training]
        ],
        "validation": [
            {"image": str(im), "label": str(lbl)} for im, lbl in pairs[-num_valid:]
        ],
    }

    return datalist_path.write_text(json.dumps(data_config, indent=2))


def main():
    typer.run(make_datalist)


if __name__ == "__main__":
    main()
