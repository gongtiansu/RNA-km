import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np
from .base_model import BaseModel
from .attention import TransformerEncoder
from .positional_encoding import RelativePositionBias

class BaseVocab:
    BASE = "ATCGX"
    assert len(BASE) == 5
    unk_idx = 5
    mask_idx = 6
    pad_idx = 7
    _vocab = np.full(128, unk_idx, dtype=np.uint8)
    _vocab[np.array(BASE, "c").view(np.uint8)] = np.arange(len(BASE))
    _vocab.setflags(write=False)

    @staticmethod
    def encode(seq):
        assert isinstance(seq, str)
        return BaseVocab._vocab[np.array(seq, "c").view(np.uint8)]

    @staticmethod
    def size():
        return 8

class PretrainModel(BaseModel):
    def __init__(self, pretrain=True):
        super(PretrainModel, self).__init__()
        self.d_model = 1024
        self.n_layer = 12
        self.d_pair = 512
        self.pretrain = pretrain

        n_head = self.d_model // 64

        dropout = 0.1
        self.embed_tokens = nn.Sequential(
            nn.Embedding(BaseVocab.size(), self.d_model, padding_idx=BaseVocab.pad_idx),
            nn.Dropout(dropout),
        )
        self.embed_rp = RelativePositionBias(
            num_buckets=64, max_distance=256, n_head=n_head
        )

        self.encoder = TransformerEncoder(
            d_model=self.d_model,
            d_key=self.d_model,
            n_layer=self.n_layer,
            n_head=n_head,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
        )

        self.layer_norm_after = nn.LayerNorm(self.d_model)
        self.fc_pair_q = nn.Linear(self.d_model, self.d_pair)
        self.fc_pair_k = nn.Linear(self.d_model, self.d_pair)

        self.fc_pair_rp = RelativePositionBias(
            num_buckets=64, max_distance=256, n_head=self.d_pair
        )

        self.fc_pair_cls = nn.Sequential(
            nn.Linear(self.d_pair, self.d_pair),
            nn.ReLU(),
            nn.Linear(self.d_pair, 6**2),
        )
        if not pretrain:
            self.fc_pair_cls[-1].weight.requires_grad = False
            self.fc_pair_cls[-1].bias.requires_grad = False

    def forward(self, data):
        """
        tokens has shape (1, L)
        """
        # 1. make embedding
        L = data["src"].shape[1]
        x = self.embed_tokens(data["src"])

        # 2. make attention bias
        pos = torch.arange(L, device=x.device)
        pos = pos.unsqueeze(1) - pos.unsqueeze(0)
        rp = self.embed_rp(pos)
        rp_bias = rp.permute(2, 0, 1).unsqueeze(0)
        mask = data["src"] == BaseVocab.pad_idx
        mask = mask[:, None, None]
        mask_bias = mask.float().masked_fill(mask, float("-inf"))
        bias = mask_bias + rp_bias
        # 3. encoder
        if True or self.pretrain or not self.training:
            x, attn_weight = self.encoder(x, bias)
        else:
            x, attn_weight = torch.utils.checkpoint.checkpoint(self.encoder, x, bias)

        x = self.layer_norm_after(x)
        out = {"residue_repr": x, "attn_weight": attn_weight}
        # 4. pair output
        q = self.fc_pair_q(x)
        k = self.fc_pair_k(x)

        if self.pretrain:
            out.update(self._compute_loss(q, k, data["tgt"]))
        else:
            px = torch.einsum("bhc,blc->bhlc", q, k)
            rp = self.fc_pair_rp(pos)
            px = px + rp

            for fn in self.fc_pair_cls[:-1]:
                px = fn(px)
            out["residue_pair_repr"] = px
            pair_prob = self.fc_pair_cls[-1](px)
            out["residue_pair_prob"] = pair_prob
        return out

