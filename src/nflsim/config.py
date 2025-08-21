from __future__ import annotations
from pydantic import BaseModel
from typing import List
import yaml

class ModelCfg(BaseModel):
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.2

class TrainCfg(BaseModel):
    lr: float = 3e-4
    weight_decay: float = 1e-2
    max_steps: int = 10_000

class DataCfg(BaseModel):
    path: str = "data/"
    years: List[int] = [1999, 2024]

class DecodingCfg(BaseModel):
    top_p: float = 0.9
    proe_shift: float = 0.0
    fourth_down_aggr: float = 1.0
    tempo: float = 1.0

class FullConfig(BaseModel):
    seed: int = 42
    data: DataCfg = DataCfg()
    model: ModelCfg = ModelCfg()
    train: TrainCfg = TrainCfg()
    decoding: DecodingCfg = DecodingCfg()

def load_config(path: str) -> FullConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return FullConfig.model_validate(raw)
