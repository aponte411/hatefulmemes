import warnings
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import random
import json
import tempfile

import torch
import fasttext
import torchvision
import transformers
import pytorch_lightning as pl
from torch import optim, nn
from torch.utils import data
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.dataset import HatefulMemesDataset
from src.models.concat import LanguageAndVisionConcat
from src.models.bert import BertBaseUncased
from src.utils import utils

warnings.filterwarnings("ignore")
LOGGER = utils.get_logger(__name__)


class HatefulMemesModel(pl.LightningModule):
    def __init__(self, hparams: Dict[str, Any]):
        for data_key in [
                "train_path",
                "dev_path",
                "img_dir",
        ]:
            if data_key not in hparams.keys():
                raise KeyError(
                    f"{data_key} is a required hparam in this model")

        super().__init__()
        self.hparams = hparams

        # assign some hparams that get used in multiple places
        self.embedding_dim = self.hparams.get("embedding_dim", 300)
        self.language_feature_dim = self.hparams.get("language_feature_dim",
                                                     300)
        self.vision_feature_dim = self.hparams.get(
        # balance language and vision features by default
            "vision_feature_dim",
            self.language_feature_dim)
        self.output_path = Path(
            self.hparams.get("output_path", "model-outputs"))
        self.output_path.mkdir(exist_ok=True)

        # instantiate transforms, datasets
        self.text_transform = self._build_bert_transform()
        self.image_transform = self._build_image_transform()
        self.train_dataset = self._build_dataset("train_path")
        self.dev_dataset = self._build_dataset("dev_path")

        # set up model and training
        self.model = self._build_model()
        self.trainer_params = self._get_trainer_params()

    # Required LightningModule Methods (when validating)
    def forward(
            self,
            ids: torch.tensor,
            mask: torch.tensor,
            token_type_ids: torch.tensor,
            image: torch.Tensor,
            label: Optional[torch.Tensor] = None,
    ) -> Any:
        return self.model(
            ids,
            mask,
            token_type_ids,
            image,
            label,
        )

    def training_step(self, batch, batch_nb) -> Dict:
        preds, loss = self.forward(
            ids=batch["text_id"],
            mask=batch["mask"],
            token_type_ids=batch["token_type_ids"],
            image=batch["image"],
            label=batch["label"],
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_nb) -> Dict:
        preds, loss = self.eval().forward(
            ids=batch["text_id"],
            mask=batch["mask"],
            token_type_ids=batch["token_type_ids"],
            image=batch["image"],
            label=batch["label"],
        )

        return {"batch_val_loss": loss}

    def validation_epoch_end(self, outputs) -> Dict:
        avg_loss = torch.stack(
            tuple(output["batch_val_loss"] for output in outputs)).mean()

        return {
            "val_loss": avg_loss,
            "progress_bar": {
                "avg_val_loss": avg_loss,
            },
        }

    def configure_optimizers(self) -> Tuple:
        optimizers = [
            optim.AdamW(self.model.parameters(),
                        lr=self.hparams.get("lr", 0.001))
        ]
        schedulers = [optim.lr_scheduler.ReduceLROnPlateau(optimizers[0])]
        return optimizers, schedulers

    @pl.data_loader
    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 16),
        )

    @pl.data_loader
    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.dev_dataset,
            shuffle=False,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 16),
        )

    def fit(self) -> None:
        """Train multimodal model."""
        self._set_seed(self.hparams.get("random_state", 42))
        self.trainer = pl.Trainer(**self.trainer_params)
        self.trainer.fit(self)

    def _set_seed(self, seed: int) -> None:
        """Set random seed."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_fasttext_transform(self) -> Any:
        """Build fasttext transform."""
        with tempfile.NamedTemporaryFile() as ft_training_data:
            ft_path = Path(ft_training_data.name)
            with ft_path.open("w") as ft:
                training_data = [
                    json.loads(line)["text"] + "/n" for line in open(
                        self.hparams.get("train_path")).read().splitlines()
                ]
                for line in training_data:
                    ft.write(line + "\n")
                language_transform = fasttext.train_unsupervised(
                    str(ft_path),
                    model=self.hparams.get("fasttext_model", "cbow"),
                    dim=self.embedding_dim,
                )
        return language_transform

    def _build_bert_transform(self) -> Any:
        """Build BERT text transform."""
        return transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True,
        )

    def _build_image_transform(self) -> Any:
        """Build image transform."""
        image_dim = self.hparams.get("image_dim", 224)
        image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(image_dim, image_dim)),
            torchvision.transforms.ToTensor(),
            # all torchvision models expect the same
            # normalization mean and std
            # https://pytorch.org/docs/stable/torchvision/models.html
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225)),
        ])
        return image_transform

    def _build_dataset(self, dataset_key: str) -> HatefulMemesDataset:
        """Build HatefulMemesDataset for a given dataset."""
        return HatefulMemesDataset(
            data_path=self.hparams.get(dataset_key, dataset_key),
            img_dir=self.hparams.get("img_dir"),
            image_transform=self.image_transform,
            text_transform=self.text_transform,
            # limit training samples only
            dev_limit=(self.hparams.get("dev_limit", None)
                       if "train" in str(dataset_key) else None),
            balance=True if "train" in str(dataset_key) else False,
        )

    def _build_model(self) -> LanguageAndVisionConcat:
        """Build multimodal model."""
        # BERT features
        language_module = BertBaseUncased(n_outputs=self.language_feature_dim)

        # finetuning Resnet152, Resnet is 2048 out
        vision_module = torchvision.models.resnet152(pretrained=True)
        vision_module.fc = nn.Linear(
            in_features=2048,
            out_features=self.vision_feature_dim,
        )

        return LanguageAndVisionConcat(
            num_classes=self.hparams.get("num_classes", 2),
            loss_fn=nn.CrossEntropyLoss(),
            language_module=language_module,
            vision_module=vision_module,
            language_feature_dim=self.language_feature_dim,
            vision_feature_dim=self.vision_feature_dim,
            fusion_output_size=self.hparams.get("fusion_output_size", 512),
            dropout_p=self.hparams.get("dropout_p", 0.1),
        )

    def _get_trainer_params(self) -> Dict:
        """Get trainer params for experiment."""
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=self.output_path,
            monitor=self.hparams.get("checkpoint_monitor", "avg_val_loss"),
            mode=self.hparams.get("checkpoint_monitor_mode", "min"),
            verbose=self.hparams.get("verbose", True),
        )

        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor=self.hparams.get("early_stop_monitor", "avg_val_loss"),
            min_delta=self.hparams.get("early_stop_min_delta", 0.001),
            patience=self.hparams.get("early_stop_patience", 3),
            verbose=self.hparams.get("verbose", True),
        )

        trainer_params = {
            "checkpoint_callback":
            checkpoint_callback,
            "early_stop_callback":
            early_stop_callback,
            "accumulate_grad_batches":
            self.hparams.get("accumulate_grad_batches", 1),
            "gpus":
            self.hparams.get("n_gpu", 1),
            "max_epochs":
            self.hparams.get("max_epochs", 100),
            "gradient_clip_val":
            self.hparams.get("gradient_clip_value", 1),
        }
        return trainer_params

    @torch.no_grad()
    def make_submission_frame(self, test_path: str) -> pd.DataFrame:
        """Conduct inference and format predictions for submission."""
        test_dataset = self._build_dataset(test_path)
        submission_frame = pd.DataFrame(index=test_dataset.samples_frame.id,
                                        columns=["proba", "label"])
        test_dataloader = data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 16),
        )
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            preds, _ = self.model.eval().to("cpu")(
                batch["text_id"],
                batch["mask"],
                batch["token_type_ids"],
                batch["image"],
            )
            submission_frame.loc[batch["id"], "proba"] = preds[:, 1]
            submission_frame.loc[batch["id"], "label"] = preds.argmax(dim=1)

        submission_frame.proba = submission_frame.proba.astype(float)
        submission_frame.label = submission_frame.label.astype(int)
        submission_frame.to_csv("submission.csv", index=True)
        return submission_frame
