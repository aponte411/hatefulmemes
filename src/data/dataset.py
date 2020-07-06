from typing import Any, Optional, Dict
from PIL import Image

import torch
from torch.utils import data
import pandas as pd


class HatefulMemesDataset(data.Dataset):
    """Uses jsonl data to preprocess and serve a
    dictionary of multimodal tensors for model input.
    """
    def __init__(
        self,
        data_path: str,
        img_dir: str,
        image_transform: Any,
        text_transform: Any,
        balance: bool = False,
        dev_limit: Optional[int] = None,
        random_state: int = 0,
    ):

        self.samples_frame = pd.read_json(data_path, lines=True)
        self.dev_limit = dev_limit

        # balance label distrubution
        if balance:
            negatives = self.samples_frame[self.samples_frame.label.eq(0)]
            positives = self.samples_frame[self.samples_frame.label.eq(1)]
            self.samples_frame = pd.concat([
                negatives.sample(positives.shape[0],
                                 random_state=random_state), positives
            ])
        # apply cutoffs
        if self.dev_limit is not None:
            if self.samples_frame.shape[0] > self.dev_limit:
                self.samples_frame = self.samples_frame.sample(
                    dev_limit, random_state=random_state)
        self.samples_frame = self.samples_frame.reset_index(drop=True)
        self.samples_frame.img = self.samples_frame.apply(lambda row:
                                                          (img_dir / row.img),
                                                          axis=1)

        # https://github.com/drivendataorg/pandas-path
        if not self.samples_frame.img.path.exists().all():
            raise FileNotFoundError
        if not self.samples_frame.img.path.is_file().all():
            raise TypeError

        self.image_transform = image_transform
        self.text_transform = text_transform

    def __len__(self) -> int:
        return len(self.samples_frame)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.samples_frame.loc[idx, "id"]

        image = Image.open(self.samples_frame.loc[idx, "img"]).convert("RGB")
        image = self.image_transform(image)

        text = torch.Tensor(
            self.text_transform.get_sentence_vector(
                self.samples_frame.loc[idx, "text"])).squeeze()

        if "label" in self.samples_frame.columns:
            label = torch.Tensor([self.samples_frame.loc[idx, "label"]
                                  ]).long().squeeze()
            sample = {
                "id": img_id,
                "image": image,
                "text": text,
                "label": label
            }
        else:
            sample = {"id": img_id, "image": image, "text": text}

        return sample
