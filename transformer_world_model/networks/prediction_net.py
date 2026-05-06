from typing import Tuple, Any, Dict, Union, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
import functools
import chex
from flax.linen.initializers import orthogonal, constant, glorot_uniform
from flax.linen.linear import PrecisionLike
from .transformer import RopeTransformer, make_causal_mask_shape

class OutHead(nn.Module):
    output_dim: int
    hidden_dim: int = 256
    num_layers: int = 1
    normalize: bool = False

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers):
            x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), name=f'layers_{2*i}')(x)
            if self.normalize:
                x = nn.LayerNorm(name=f'layers_{2*i+1}')(x)
            x = nn.gelu(x)
        return nn.Dense(self.output_dim, kernel_init=orthogonal(jnp.sqrt(2)), name=f'layers_{2*i+2}')(x)
        

class PredictionNet(nn.Module):
    embed_dim: int
    num_modalities: int

    embeds: Dict
    out_heads: Dict
    modality_order: Tuple

    dropout_p: float
    feedforward_dim: int
    num_heads: int
    num_layers: int
    decode: bool = False
    normalize_qk: bool = False
    normalize_out_heads: bool = False
    precision: PrecisionLike = None
    dtype: Optional[Any] = None

    context_size: Optional[int] = None
    concat_tokens: bool = False
    sum_tokens: bool = False

    def setup(self):
        if self.concat_tokens:
            embed_dim = self.embed_dim // 2
        else:
            embed_dim = self.embed_dim

        self.embed_mods = {
            name: nn.Embed(embed['in_dim'], embed_dim) if embed['categorical'] else
            nn.Sequential([
                nn.Dense(embed_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.LayerNorm(),
            ])
            for name, embed in self.embeds.items()
        }

        self.transformer = RopeTransformer(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            feedforward_dim=self.feedforward_dim,
            dropout_p=self.dropout_p,
            decode=self.decode,
            normalize_qk=self.normalize_qk,
            precision=self.precision,
            dtype=self.dtype,
        )

        self.out_head_mods = {
            name: OutHead(out_head["out_dim"], out_head["hidden_dim"], out_head["num_layers"], self.normalize_out_heads)
            for name, out_head in self.out_heads.items()
        }

    
    @nn.compact
    def __call__(self, inputs, deterministic=True):
        modality_order = self.modality_order

        ordered_embeds = [self.embed_mods[name](inputs[name]) for name in modality_order]

        if self.concat_tokens:
            inputs = jnp.concatenate(ordered_embeds, axis=-1)
        elif self.sum_tokens:
            inputs = jnp.stack(ordered_embeds, axis=-1)
            inputs = jnp.sum(inputs, axis=-1)
        else:
            # CURRENTLY, IF NOT CONCATING/SUMMING EMBEDS, CONTEXT SIZE WILL BE EFFECTIVELY HALVED
            inputs = jnp.stack(ordered_embeds, axis=1)
            inputs = inputs.reshape(-1, *inputs.shape[2:])

        # Swapaxes as data is (time, batch, ....)
        inputs = jnp.swapaxes(inputs, 0, 1)

        if self.decode:
            mask = None
        else:
            mask = make_causal_mask_shape(inputs.shape[:-1], context_size=self.context_size)
        hiddens = self.transformer(inputs, mask=mask, deterministic=deterministic)
        
        # Swapaxes back as data is (time, batch, ....)
        hiddens = jnp.swapaxes(hiddens, 0, 1)

        if not (self.concat_tokens or self.sum_tokens):
            out_idxs = jnp.flip(jnp.arange(hiddens.shape[0] - 1, -1, -self.num_modalities), 0)
            hiddens = hiddens[out_idxs]

        out = {name: head(hiddens) for name, head in self.out_head_mods.items()}
        return out, hiddens