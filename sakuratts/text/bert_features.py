"""PyTorch research implementation of the official Chinese BERT feature layer.

This module is a correctness and resource experiment, not the final runtime
dependency. It accepts already-tokenized inputs and does not alter the frontend.
"""

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, BertConfig, BertModel


def feature_config(source: BertConfig) -> BertConfig:
    """Select hidden_states[-3], counting the embedding output at index zero."""
    if source.model_type != "bert" or source.is_decoder or source.add_cross_attention:
        raise ValueError("Expected an encoder-only BERT checkpoint")
    if source.num_hidden_layers < 3:
        raise ValueError("The feature extractor requires at least three BERT layers")
    config = deepcopy(source)
    config.num_hidden_layers -= 2
    config.output_hidden_states = False
    config.output_attentions = False
    return config


class BertFeatures(nn.Module):
    """Embedding plus the first N-2 layers; returns every token unchanged."""

    def __init__(self, bert: BertModel, source_num_hidden_layers: int):
        super().__init__()
        self.bert = bert
        self.source_num_hidden_layers = source_num_hidden_layers

    @classmethod
    def from_pretrained(cls, path: str | Path, dtype=torch.float32) -> "BertFeatures":
        source = AutoConfig.from_pretrained(path, local_files_only=True)
        if source.architectures != ["BertForMaskedLM"]:
            raise ValueError("Expected an unpruned BertForMaskedLM checkpoint")
        config = feature_config(source)
        bert, loading = BertModel.from_pretrained(
            path, config=config, add_pooling_layer=False, torch_dtype=dtype,
            local_files_only=True, output_loading_info=True,
        )
        # The removed encoder layers and MLM head are expected unused weights.
        # Missing retained weights would silently initialize random features.
        if loading["missing_keys"] or loading["mismatched_keys"] or loading["error_msgs"]:
            raise ValueError(f"Incomplete BERT feature checkpoint: {loading}")
        return cls(bert, source.num_hidden_layers).eval()

    def forward(self, **inputs) -> torch.Tensor:
        return self.bert(
            **inputs, output_hidden_states=False, output_attentions=False, return_dict=True,
        ).last_hidden_state
