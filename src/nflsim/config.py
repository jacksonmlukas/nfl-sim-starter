from pydantic import BaseModel
from typing import List

class DataConfig(BaseModel):
    seasons: List[int]
    processed_path: str
    sample_plays: int | None = None

class ModelConfig(BaseModel):
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.2
    context_k: int = 6

class TrainConfig(BaseModel):
    batch_size: int = 256
    grad_accum: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_steps: int = 8000
    teacher_forcing_final: float = 0.6

class DecodeConfig(BaseModel):
    top_p: float = 0.9
    max_plays_per_drive: int = 25
    max_drives_per_game: int = 24

class FullConfig(BaseModel):
    seed: int = 1337
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    decode: DecodeConfig
