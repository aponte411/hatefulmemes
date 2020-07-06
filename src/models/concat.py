from typing import Tuple, Any, Optional

import torch.nn.functional as F
import torch
from torch import nn


class LanguageAndVisionConcat(nn.Module):
    def __init__(
        self,
        num_classes: int,
        loss_fn: Any,
        language_module: Any,
        vision_module: Any,
        language_feature_dim: int,
        vision_feature_dim: int,
        fusion_output_size: int,
        dropout_p: int,
    ):
        super().__init__()
        self.language_module = language_module
        self.vision_module = vision_module
        self.fusion = nn.Linear(
            in_features=(language_feature_dim + vision_feature_dim),
            out_features=fusion_output_size,
        )
        self.fc = nn.Linear(in_features=fusion_output_size,
                            out_features=num_classes)
        self.loss_fn = loss_fn
        self.dropout = nn.Dropout(dropout_p)

    def forward(
            self,
            text: torch.Tensor,
            image: torch.Tensor,
            label: Optional[torch.Tensor] = None,
    ) -> Tuple:
        text_features = F.relu(self.language_module(text))
        image_features = F.relu(self.vision_module(image))
        combined = torch.cat([text_features, image_features], dim=1)
        fused = self.dropout(F.relu(self.fusion(combined)))
        logits = self.fc(fused)
        pred = F.softmax(logits)
        loss = (self.loss_fn(pred, label) if label is not None else label)
        return (pred, loss)
