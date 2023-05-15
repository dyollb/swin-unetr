import json
import random
from pathlib import Path
from typing import List

import typer


def find_matching_files(input_globs: List[Path], verbose: bool = True):
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


def main(
    data_dir: Path = Path("C:/Users/lloyd/datasets/CC"),
    t1_dir: Path = Path("t1w_n4/*.nii.gz"),
    labels_dir: Path = Path("skull_vertebrae2/*.nii.gz"),
    dataset_path: Path = Path("datalists") / "skull_vertebrae_all.json",
    percent: float = 1.0,
):
    pairs = find_matching_files([data_dir / t1_dir, data_dir / labels_dir])
    pairs = [(im.relative_to(data_dir), lbl.relative_to(data_dir)) for im, lbl in pairs]

    labels = {"1": "Skull", "2": "Vertebrae"}

    random.Random(104).shuffle(pairs)
    test, pairs = pairs[:10], pairs[10:]

    num_valid = int(percent * 0.2 * len(pairs))
    num_training = len(pairs) - num_valid if percent >= 1.0 else 4 * num_valid

    data_config = {
        "description": "Calgary-Campinas",
        "labels": labels,
        "test": [str(im) for im, _ in test],
        "training": [
            {"image": str(im), "label": str(lbl)} for im, lbl in pairs[:num_training]
        ],
        "validation": [
            {"image": str(im), "label": str(lbl)} for im, lbl in pairs[-num_valid:]
        ],
    }

    dataset_path.write_text(json.dumps(data_config, indent=2))


if __name__ == "__main__":
    typer.run(main)
