"""A small, readable decoder-only transformer used in Lab 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    max_seq_len: int = 512
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 384
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            split_heads(q),
            split_heads(k),
            split_heads(v),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class TinyGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, length = input_ids.shape
        if length > self.config.max_seq_len:
            raise ValueError(f"Sequence length {length} exceeds {self.config.max_seq_len}")
        positions = torch.arange(length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat((input_ids, next_id), dim=1)
                continue
            next_logits = next_logits / temperature
            if top_k > 0:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[:, [-1]]] = -float("inf")
            probabilities = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
        return input_ids

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def checkpoint(self) -> dict:
        return {"config": asdict(self.config), "state_dict": self.state_dict()}
