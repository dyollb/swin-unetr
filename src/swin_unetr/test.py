import json
from enum import Enum
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import torch
import typer
from monai.config import print_config
from monai.data import (
    CacheDataset,
    DataLoader,
    decollate_batch,
    load_decathlon_datalist,
)
from monai.inferers import SlidingWindowInferer
from monai.networks.layers import Norm
from monai.networks.nets import SwinUNETR, UNet
from monai.transforms import (
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    SaveImaged,
    Spacingd,
)
from tqdm import tqdm


class Model(Enum):
    SWIN = "swin"
    UNET_256 = "unet256"
    UNET_128 = "unet128"


def main(
    data_dir: Path = Path("C:/Users/lloyd/datasets/CC"),
    json_path: Path = Path("datalists/skull_vertebrae_20.json"),
    model_path: Path = Path(
        "C:/Users/lloyd/datasets/CC/skull_vertebrae_all_log/best_metric_model.pth"
    ),
    output_dir: Path = Path("C:/Users/lloyd/datasets/CC/skull_vertebrae_all_pred"),
    overlap: float = 0.5,
    network: Model = Model.SWIN.value,
    gpu_id: int = 0,
):
    freeze_support()
    print_config()

    labels = json.loads(json_path.read_text())["labels"]
    labels = {int(k): v for k, v in labels.items()}
    num_classes = max(labels.keys()) + 1

    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    test_transforms = Compose(
        [
            LoadImaged(keys=["image"], reader="ITKReader"),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear"),
            ),
            NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
            CropForegroundd(keys=["image"], source_key="image"),
            EnsureTyped(keys=["image"], dtype=np.float32, device=device),
        ]
    )

    test_files = load_decathlon_datalist(
        base_dir=data_dir,
        data_list_file_path=json_path,
        is_segmentation=True,
        data_list_key="test",
    )

    test_ds = CacheDataset(
        data=test_files,
        transform=test_transforms,
        cache_num=6,
        cache_rate=1.0,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=not torch.cuda.is_available(),
    )

    inferer = SlidingWindowInferer(
        roi_size=(96, 96, 96),
        overlap=overlap,
        mode="gaussian",
        sw_batch_size=4,
        device=device,
    )

    post_transform = Compose(
        [
            EnsureTyped(keys="pred"),
            Invertd(
                keys="pred",
                transform=test_transforms,
                orig_keys="image",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True),
            SaveImaged(
                keys="pred",
                output_dir=output_dir,
                output_postfix="",
                resample=False,
                separate_folder=False,
                print_log=False,
                writer="ITKWriter",
            ),
        ]
    )

    if network == Model.SWIN:
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=num_classes,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
        )
    else:
        network_layers_map = {Model.UNET_128: 4, Model.UNET_256: 5}
        num_layers = network_layers_map[network]
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            channels=[16 * pow(2, k) for k in range(num_layers)],
            strides=[2] * (num_layers - 1),
            dropout=0.0,
            num_res_units=2,
            norm=Norm.BATCH,
        )

    model_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_dict, strict=True)
    model.eval()
    model.to(device)

    num_steps = len(test_files)
    test_iterator = tqdm(
        test_loader, desc=f"Testing (0 / {num_steps} Steps)", dynamic_ncols=True
    )

    with torch.no_grad():
        for _step, batch in enumerate(test_iterator, start=1):
            test_inputs = batch["image"].to(device)
            test_preds = inferer(test_inputs, model)
            batch["pred"] = test_preds
            for i in decollate_batch(batch):
                post_transform(i)

            test_iterator.set_description(f"Testing ({_step} / {num_steps} Steps)")


if __name__ == "__main__":
    typer.run(main)

    # python test.py --json-path datalists\hippocampus_all.json --model-path C:\Users\lloyd\datasets\CC\hippocampus_all_log\best_metric_model.pth --output-dir C:\Users\lloyd\datasets\CC\hippocampus_all_pred
    # python test.py --json-path datalists\vessels_all.json --model-path C:\Users\lloyd\datasets\CC\vessels_all_log\best_metric_model.pth --output-dir C:\Users\lloyd\datasets\CC\vessels_all_pred

    # python src\swin_unetr\test.py --json-path datalists\brain_best.json --model-path C:\Users\lloyd\datasets\CC\brain_best_log\best_metric_model.pth --output-dir C:\Users\lloyd\datasets\CC\brain_best_pred

    # python src\swin_unetr\test.py --json-path datalists\head_mask_best.json --model-path C:\Users\lloyd\datasets\CC\head_mask_log\best_metric_model.pth --output-dir C:\Users\lloyd\datasets\CC\head_mask_pred --network unet256
