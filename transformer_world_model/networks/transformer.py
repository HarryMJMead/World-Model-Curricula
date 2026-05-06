from typing import Tuple, Any, Dict, Union, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
import functools
import chex
from flax.linen.initializers import orthogonal, constant, glorot_uniform
from flax.linen import MultiHeadDotProductAttention, DenseGeneral, LayerNorm, merge_param, combine_masks
from flax.linen.linear import PrecisionLike

class RopeTransformer(nn.Module):
    num_layers: int
    num_heads: int
    qkv_features: int
    feedforward_dim: int
    dropout_p: float = 0
    decode: bool = False
    normalize_qk: bool = False
    precision: PrecisionLike = None
    dtype: Optional[Any] = None

    @nn.compact
    def __call__(
        self,
        inputs: chex.Array,
        mask: Optional[chex.Array] = None,
        deterministic: bool = True
    ):
        x = inputs
        for i in range(self.num_layers):
            x = RopeTransformerLayer(
                num_heads=self.num_heads,
                qkv_features=self.qkv_features,
                feedforward_dim=self.feedforward_dim,
                dropout_p=self.dropout_p,
                decode=self.decode,
                normalize_qk=self.normalize_qk,
                precision=self.precision,
                dtype=self.dtype,
                name=f'layer_{i}'
            )(x, mask=mask, deterministic=deterministic)
        return x


class RopeTransformerLayer(nn.Module):
    num_heads: int
    qkv_features: int
    feedforward_dim: int
    dropout_p: float = 0
    decode: bool = False
    normalize_qk: bool = False
    precision: PrecisionLike = None
    dtype: Optional[Any] = None

    @nn.compact
    def __call__(
        self,
        inputs: chex.Array,
        mask: Optional[chex.Array] = None,
        deterministic: bool = True
    ):
        # Self-attention with RoPE
        attn_output = SelfAttentionRoPE(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            dropout_rate=self.dropout_p,
            decode=self.decode,
            normalize_qk=self.normalize_qk,
            precision=self.precision,
            dtype=self.dtype,
            name='self_attention'
        )(inputs, mask=mask, deterministic=deterministic)
        attn_output = nn.Dropout(self.dropout_p)(attn_output, deterministic=deterministic)
        out1 = LayerNorm(name='attn_layer_norm')(inputs + attn_output)

        # Feedforward network
        ff_output = nn.Dense(
            self.feedforward_dim, 
            kernel_init=orthogonal(jnp.sqrt(2)),
            precision=self.precision,
            dtype=self.dtype,
            name='ff_dense1',
        )(out1)
        ff_output = nn.gelu(ff_output)
        ff_output = nn.Dropout(self.dropout_p)(ff_output, deterministic=deterministic)
        ff_output = nn.Dense(
            inputs.shape[-1], 
            kernel_init=orthogonal(jnp.sqrt(2)),
            precision=self.precision,
            dtype=self.dtype,
            name='ff_dense2',
        )(ff_output)
        ff_output = nn.Dropout(self.dropout_p)(ff_output, deterministic=deterministic)
        out2 = LayerNorm(name='ff_layer_norm')(out1 + ff_output)
        return out2


class SelfAttentionRoPE(MultiHeadDotProductAttention):
    use_rope: bool = True  # Whether to use RoPE positional embeddings.
    max_decode_length: int = 600

    def _get_sin_cos(self, seq_len: int, head_dim: int, offset: int = 0, base: float = 10000.0) -> Tuple[chex.Array, chex.Array]:
        """Get sin and cos positional embeddings for RoPE."""
        theta = jnp.power(base, -jnp.arange(0, head_dim, 2) / head_dim)
        positions = jnp.arange(seq_len) + offset
        angles = jnp.outer(positions, theta)
        sin = jnp.sin(angles)
        cos = jnp.cos(angles)
        return sin, cos
    
    def _apply_rope(self, x: chex.Array, sin: chex.Array, cos: chex.Array) -> chex.Array:
        """Apply RoPE to the input tensor."""
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        sin = sin[:, None, :]
        cos = cos[:, None, :]
        x_rotated = jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
        return x_rotated
    
    def setup(self):
        qkv_features = self.qkv_features
        if qkv_features is None:
            raise ValueError(
                "For setup-time RoPE table creation, qkv_features "
                "must be set on the module."
            )
        if qkv_features % self.num_heads != 0:
            raise ValueError(
                f"qkv_features ({qkv_features}) must be divisible by num_heads ({self.num_heads})."
            )

        self.head_dim = qkv_features // self.num_heads
        if self.use_rope:
            if self.head_dim % 2 != 0:
                raise ValueError(f"RoPE requires even head_dim, got {self.head_dim}.")
            
            self.rope_sin, self.rope_cos = self._get_sin_cos(self.max_decode_length, self.head_dim)
            

    @nn.compact
    def __call__(
        self,
        inputs_q: chex.Array,
        mask: Optional[chex.Array] = None,
        deterministic: Optional[bool] = None,
    ):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        Args:
        inputs_q: input queries of shape `[batch_sizes..., length, features]`.
        mask: attention mask of shape `[batch_sizes..., num_heads, query_length,
            key/value_length]`. Attention weights are masked out if their
            corresponding mask value is `False`.
        deterministic: if false, the attention weight is masked randomly using
            dropout, whereas if true, the attention weights are deterministic.

        Returns:
            output of shape `[batch_sizes..., length, features]`.
        """
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert qkv_features % self.num_heads == 0, (
            f'Memory dimension ({qkv_features}) must be divisible by number of'
            f' heads ({self.num_heads}).'
        )
        head_dim = self.head_dim

        dense = functools.partial(
            DenseGeneral,
            axis=-1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, head_dim),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
            dot_general=self.qkv_dot_general,
            dot_general_cls=self.qkv_dot_general_cls,
        )
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        query, key, value = (
            dense(name='query')(inputs_q),
            dense(name='key')(inputs_q),
            dense(name='value')(inputs_q),
        )

        if self.normalize_qk:
            # Normalizing query and key projections stabilizes training with higher
            # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
            query = LayerNorm(name='query_ln', use_bias=False)(query)  # type: ignore[call-arg]
            key = LayerNorm(name='key_ln', use_bias=False)(key)  # type: ignore[call-arg]
        
        if self.use_rope:
            seq_len = query.shape[-3]
            assert seq_len <= self.max_decode_length, (
                f'Input length ({seq_len}) cannot exceed the'
                f' initialised maximum decode length ({self.max_decode_length})'
            )

            sin, cos = self.rope_sin[:seq_len], self.rope_cos[:seq_len]

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.decode:
            # detect if we're initializing by absence of existing cache data.
            is_initialized = self.has_variable('cache', 'cached_key')
            cached_key = self.variable(
                'cache', 'cached_key', jnp.zeros, key.shape, key.dtype
            )
            cached_value = self.variable(
                'cache', 'cached_value', jnp.zeros, value.shape, value.dtype
            )
            cache_index = self.variable(
                'cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.int32)
            )
            if is_initialized:
                (
                    *batch_dims,
                    max_length,
                    num_heads,
                    depth_per_head,
                ) = cached_key.value.shape
                # shape check of cached keys against query input
                num_tokens = query.shape[-3]
                expected_shape = tuple(batch_dims) + (num_tokens, num_heads, depth_per_head)
                if expected_shape != query.shape:
                    raise ValueError(
                        'Autoregressive cache shape error, '
                        'expected query shape %s instead got %s.'
                        % (expected_shape, query.shape)
                    )
                # update key, value caches with our new 1d spatial slices
                cur_index = cache_index.value
                slot = cur_index % max_length

                indices: tuple[Union[int, jax.chex.Array], ...] = (0,) * len(batch_dims) + (
                    slot,
                    0,
                    0,
                )

                if self.use_rope:
                    q_sin = self.rope_sin[max_length-num_tokens:max_length]
                    q_cos = self.rope_cos[max_length-num_tokens:max_length]
                    query = self._apply_rope(query, q_sin, q_cos)

                    sin, cos = self.rope_sin[:max_length], self.rope_cos[:max_length]
                    idx = (jnp.arange(max_length) - (slot + num_tokens)) % max_length
                    sin = jnp.take(sin, idx, axis=0)
                    cos = jnp.take(cos, idx, axis=0)

                key = jax.lax.dynamic_update_slice(cached_key.value, key, indices)
                value = jax.lax.dynamic_update_slice(cached_value.value, value, indices)
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cur_index + num_tokens
                # causal mask for cached decoder self-attention:
                # our single query position should only attend to those key
                # positions that have already been generated and cached,
                # not the remaining zero elements.
                mask = nn.combine_masks(
                    mask,
                    jnp.broadcast_to(
                        jnp.arange(max_length) <= (jnp.arange(num_tokens)[:, None] + cur_index),
                        tuple(batch_dims) + (1, num_tokens, max_length),
                    ),
                )

        # Apply RoPE to query and key tensors
        if self.use_rope:
            if not self.decode:
                query = self._apply_rope(query, sin, cos)
            key = self._apply_rope(key, sin, cos)

        dropout_rng = None
        if (
            self.dropout_rate > 0.0
        ):  # Require `deterministic` only if using dropout.
            m_deterministic = merge_param(
                'deterministic', self.deterministic, deterministic
            )
            if not m_deterministic:
                dropout_rng = self.make_rng('dropout')
        else:
            m_deterministic = True

        # apply attention
        x = self.attention_fn(
            query,
            key,
            value,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=m_deterministic,
            dtype=self.dtype,
            precision=self.precision,
        )  # pytype: disable=wrong-keyword-args
        # back to the original inputs dimensions
        out = DenseGeneral(
            features=features,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            dot_general=self.out_dot_general,
            dot_general_cls=self.out_dot_general_cls,
            name='out',  # type: ignore[call-arg]
        )(x)
        return out
    
def in_context_region(a, b, n):
    return jnp.logical_and(a >= b, a < b + n)

def make_causal_mask_shape(
    shape: chex.Array,
    extra_batch_dims: int = 0,
    dtype: jnp.dtype = jnp.bool_,
    context_size: Optional[int] = None
):
    if context_size == None:
        mask_fn = jnp.greater_equal
    else:
        mask_fn = lambda a, b: in_context_region(a, b, context_size)
    
    idxs = jnp.broadcast_to(jnp.arange(shape[-1], dtype=jnp.int32), shape)
    return nn.make_attention_mask(
        idxs,
        idxs,
        mask_fn,
        extra_batch_dims=extra_batch_dims,
        dtype=dtype,
    )