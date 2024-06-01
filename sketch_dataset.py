import pandas as pd
import torch
from PIL import Image


class SketchDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_transforms=None, conditional_img_transforms=None):
        self.img_transforms = img_transforms
        self.conditional_img_transforms = conditional_img_transforms
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img = Image.open(img_name).convert("RGB")
        img = self.img_transforms(img)

        label = torch.nn.functional.one_hot(
            torch.LongTensor([1 if "RI" in img_name else 0]), num_classes=2
        )

        conditional_img = self.data.iloc[idx, 1]
        conditional_img = Image.open(conditional_img).convert("L")
        conditional_img = self.conditional_img_transforms(conditional_img)
        if conditional_img.dim == 2:
            conditional_img = conditional_img.unsqueeze(0)

        return img, conditional_img, label


def get_sketch_loader(cfg, train, img_transforms, conditional_img_transforms):
    dataset = SketchDataset(
        cfg.TRAIN.CSV,
        img_transforms=img_transforms,
        conditional_img_transforms=conditional_img_transforms,
    )
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=train,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
