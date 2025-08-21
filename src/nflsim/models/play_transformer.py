from __future__ import annotations
import torch
import torch.nn as nn
from nflsim.models.components import MonotonicQuantileHead

class PlayTransformer(nn.Module):
    """
    Encoder-style transformer over K-step context.
    Inputs are integer index tensors of shape [B,K] for each bucketized feature.
    We sum feature embeddings per token + positional embedding, then encode with TransformerEncoder
    (Pre-LN via norm_first=True). Heads:
      - play_type logits
      - yards quantiles (run/pass separate; monotone)
      - fg_attempt_logit, punt_attempt_logit (aux heads for 4th down)
    """
    def __init__(
        self,
        d_model=512, n_layers=8, n_heads=8, dropout=0.2,
        n_play_types=6,
        n_down=5, n_flags=4, n_yard=128, n_dist=22, n_time=121, n_score=29,
        n_team=40, n_season=20,
        n_roof=5, n_surface=6, n_temp=6, n_wind=5, n_g2g=2, n_rz=2,
        max_len=16,
    ):
        super().__init__()
        E = d_model
        # token feature embeddings
        self.embed_down    = nn.Embedding(n_down, E)
        self.embed_flags   = nn.Embedding(n_flags, E)
        self.embed_yard    = nn.Embedding(n_yard, E)
        self.embed_dist    = nn.Embedding(n_dist, E)
        self.embed_time    = nn.Embedding(n_time, E)
        self.embed_score   = nn.Embedding(n_score, E)
        self.embed_posteam = nn.Embedding(n_team, E)
        self.embed_defteam = nn.Embedding(n_team, E)
        self.embed_season  = nn.Embedding(n_season, E)
        self.embed_roof    = nn.Embedding(n_roof, E)
        self.embed_surface = nn.Embedding(n_surface, E)
        self.embed_temp    = nn.Embedding(n_temp, E)
        self.embed_wind    = nn.Embedding(n_wind, E)
        self.embed_g2g     = nn.Embedding(n_g2g, E)
        self.embed_rz      = nn.Embedding(n_rz, E)

        self.pos_emb = nn.Embedding(max_len, E)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=E, nhead=n_heads, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

        # heads
        self.head_play_type      = nn.Linear(E, n_play_types)
        self.head_yards_run      = MonotonicQuantileHead(E, 9)
        self.head_yards_pass     = MonotonicQuantileHead(E, 9)
        self.head_fg_attempt     = nn.Linear(E, 1)   # logit
        self.head_punt_attempt   = nn.Linear(E, 1)   # logit

    def token_embed(self, x):
        h = (
            self.embed_down(x['down'])
          + self.embed_flags(x['flags'])
          + self.embed_yard(x['yard_bucket'])
          + self.embed_dist(x['dist_bucket'])
          + self.embed_time(x['time_bucket'])
          + self.embed_score(x['score_bucket'])
          + self.embed_posteam(x['posteam'])
          + self.embed_defteam(x['defteam'])
          + self.embed_season(x['season'])
          + self.embed_roof(x['roof_bucket'])
          + self.embed_surface(x['surface_bucket'])
          + self.embed_temp(x['temp_bucket'])
          + self.embed_wind(x['wind_bucket'])
          + self.embed_g2g(x['g2g_flag'])
          + self.embed_rz(x['rz_flag'])
        )
        B, K, _ = h.shape
        pos = torch.arange(K, device=h.device).unsqueeze(0).expand(B, K)
        return self.dropout(h + self.pos_emb(pos))

    def forward(self, x):
        """
        Required keys in x (all [B,K] Long):
          down, flags, yard_bucket, dist_bucket, time_bucket, score_bucket,
          posteam, defteam, season, roof_bucket, surface_bucket, temp_bucket, wind_bucket,
          g2g_flag, rz_flag
        """
        h = self.token_embed(x)
        h = self.encoder(h)
        h_last = h[:, -1, :]
        return {
            'play_type_logits':     self.head_play_type(h_last),
            'yards_q_run':          self.head_yards_run(h_last),
            'yards_q_pass':         self.head_yards_pass(h_last),
            'fg_attempt_logit':     self.head_fg_attempt(h_last).squeeze(-1),
            'punt_attempt_logit':   self.head_punt_attempt(h_last).squeeze(-1),
        }
