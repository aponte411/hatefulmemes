from torch import nn
import transformers
import torch


class BertBaseUncased(nn.Module):
    def __init__(self, n_outputs: int):
        super().__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        # bert base has 768 hidden dimensions
        self.out = nn.Linear(768, n_outputs)

    def forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ):
        hidden_states, pooler_output = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
        )
        output = self.dropout(pooler_output)
        return self.out(output)
