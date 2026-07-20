# tools/check_nam_network.py

import torch
from torch.utils.data import DataLoader

from data.dataset import SODDataset
from models.networks.mambavision_small_nam_sod import (
    build_model,
)


def main() -> None:
    device = torch.device("cuda")

    dataset = SODDataset(
        image_dir=(
            "datasets/DUTS/DUTS-TR/"
            "DUTS-TR-Image"
        ),
        mask_dir=(
            "datasets/DUTS/DUTS-TR/"
            "DUTS-TR-Mask"
        ),
        nam_dir=(
            "datasets/DUTS/DUTS-TR/"
            "nam"
        ),
        image_size=(352, 352),
    )

    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(data_loader))

    model = build_model().to(device)
    model.train()

    image = batch["image"].to(device)
    nam_20 = batch["nam_20"].to(device)
    nam_40 = batch["nam_40"].to(device)
    nam_60 = batch["nam_60"].to(device)

    outputs = model(
        image=image,
        nam_20=nam_20,
        nam_40=nam_40,
        nam_60=nam_60,
    )

    print("output keys:", outputs.keys())
    print("pred:", outputs["pred"].shape)

    for index, prediction in enumerate(
        outputs["aux"],
        start=2,
    ):
        print(
            f"aux {index}:",
            prediction.shape,
        )

    loss = outputs["pred"].mean()
    loss.backward()

    print("backward: OK")


if __name__ == "__main__":
    main()