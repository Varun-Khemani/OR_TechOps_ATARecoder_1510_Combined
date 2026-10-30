"""Inference for the saved DAPT20 hierarchical ATA classifier, seed 42."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer
from .ranking import joint_topk, normalize_model_text


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).to(last_hidden.dtype)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


class HierATASeed42(nn.Module):
    """Model definition matching model_enhancements/train_hier_ata.py."""

    def __init__(self, encoder_config_dir: str, num_labels_2: int, num_labels_4: int):
        super().__init__()
        self.config = AutoConfig.from_pretrained(encoder_config_dir, local_files_only=True)
        self.encoder = AutoModel.from_config(self.config)
        hidden_size = getattr(self.config, "hidden_size", getattr(self.config, "d_model", 1024))
        self.drop = nn.Dropout(0.1)
        self.head2 = nn.Linear(hidden_size, num_labels_2)
        self.head4 = nn.Linear(hidden_size + num_labels_2, num_labels_4)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.drop(mean_pool(encoded.last_hidden_state, attention_mask))
        logits2 = self.head2(pooled)
        fused = torch.cat([pooled, torch.softmax(logits2, dim=-1)], dim=-1)
        return {"logits2": logits2, "logits4": self.head4(fused)}


@dataclass(frozen=True)
class Prediction:
    code: str
    score: float


class Seed42Predictor:
    def __init__(
        self,
        model_dir: str | Path,
        encoder_config_dir: str | Path,
        device: str | torch.device = "cuda",
        max_len: int = 256,
        stride: int = 128,
        alpha: float = 1.0,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.best_dir = self.model_dir / "best_model"
        self.device = torch.device(device)
        self.max_len = max_len
        self.stride = stride
        self.alpha = alpha

        with (self.model_dir / "label_maps.json").open(encoding="utf-8") as handle:
            maps = json.load(handle)
        self.label2id_2 = maps["label2id_2"]
        self.id2label_4 = maps["id2label_4"]
        self.four_to_two_ids = np.asarray(
            [
                self.label2id_2[str(self.id2label_4[str(index)])[:2]]
                for index in range(len(maps["label2id_4"]))
            ],
            dtype=np.int64,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.best_dir, local_files_only=True, use_fast=True
        )
        self.model = HierATASeed42(
            str(encoder_config_dir), len(self.label2id_2), len(self.id2label_4)
        )
        state_path = self.best_dir / "model.safetensors"
        state = load_file(str(state_path), device="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, text: Any, k: int = 5) -> list[Prediction]:
        clean_text = normalize_model_text(text)
        if not clean_text:
            return [Prediction("9999", 0.0) for _ in range(k)]

        encoded = self.tokenizer(
            clean_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_overflowing_tokens=True,
            stride=self.stride,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        output = self.model(input_ids, attention_mask)

        # Match the saved single-seed validation/test path: average window probabilities.
        probs2 = torch.softmax(output["logits2"], dim=-1).mean(dim=0).cpu().numpy()
        probs4 = torch.softmax(output["logits4"], dim=-1).mean(dim=0).cpu().numpy()
        top_ids, top_scores = joint_topk(
            probs2, probs4, self.four_to_two_ids, k=k, alpha=self.alpha
        )
        predictions = [
            Prediction(self.id2label_4[str(int(code_id))].zfill(4), float(score))
            for code_id, score in zip(top_ids, top_scores)
        ]
        predictions.extend(Prediction("9999", 0.0) for _ in range(k - len(predictions)))
        return predictions


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device)
