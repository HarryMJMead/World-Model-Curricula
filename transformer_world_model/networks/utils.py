import jax
import jax.numpy as jnp
import distrax
import flax.linen as nn

def add_uniform_noise(dist, mix, vary_noise=False, rng=None):
    noise = distrax.Categorical(logits=jnp.zeros_like(dist.probs))
    if vary_noise:
        mix = jax.random.uniform(rng, dist.probs.shape[:-2], maxval=mix)[..., None, None]
    return distrax.Categorical(probs=(1 - mix) * dist.probs + mix * noise.probs)

def two_hot(x, bins):
    # Find index of left bin
    idx = jnp.searchsorted(bins, x, side="right") - 1
    idx = jnp.clip(idx, 0, len(bins) - 2)

    left = bins[idx]
    right = bins[idx + 1]

    # Linear interpolation weights
    w_right = (x - left) / (right - left)
    w_left = 1.0 - w_right

    # Create encoding
    n_bins = bins.shape[0]
    l_encoding = nn.one_hot(idx, n_bins) * w_left[..., None]
    r_encoding = nn.one_hot(idx+1, n_bins) * w_right[..., None]

    encoding = l_encoding + r_encoding

    return encoding