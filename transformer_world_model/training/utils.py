import jax
import chex
import jax.numpy as jnp
from typing import Tuple
from flax import struct

# Computate Generalised Advantage Estimates
def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """This takes in arrays of shape (NUM_STEPS, NUM_ENVS) and returns the advantages and targets.

    Args:
        gamma (float): 
        lambd (float): 
        last_value (chex.Array):  Shape (NUM_ENVS)
        values (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        rewards (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        dones (chex.Array): Shape (NUM_STEPS, NUM_ENVS)

    Returns:
        Tuple[chex.Array, chex.Array]: advantages, targets; each of shape (NUM_STEPS, NUM_ENVS)
    """
    def compute_gae_at_timestep(carry, x):
        gae, next_value = carry
        value, reward, done = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambd * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value),
        (values, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values


def compute_clipped_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
    use_max_value: bool = False,
) -> Tuple[chex.Array, chex.Array]:
    """This takes in arrays of shape (NUM_STEPS, NUM_ENVS) and returns the advantages and targets.

    Args:
        gamma (float): 
        lambd (float): 
        last_value (chex.Array):  Shape (NUM_ENVS)
        values (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        rewards (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        dones (chex.Array): Shape (NUM_STEPS, NUM_ENVS)

    Returns:
        Tuple[chex.Array, chex.Array]: advantages, targets; each of shape (NUM_STEPS, NUM_ENVS)
    """
    extended_values = jnp.append(values, last_value[None, ...], axis=0)
    deltas = rewards + gamma * extended_values[1:] * (1 - dones) - extended_values[:-1]

    start_index = jnp.arange(values.shape[0])
    def compute_gae_at_timestep(carry, x):
        gae_array, td_total_array, is_done_array, cur_idx = carry
        
        delta, done = x
        delta = delta[None, ...] * (cur_idx >= start_index[..., None])
        done = done[None, ...] * (cur_idx >= start_index[..., None])

        i = (cur_idx - start_index)[..., None]

        td_total_array = td_total_array + gamma**i * delta
    
        clipped_td_total_array = jnp.minimum(td_total_array, 0)
        if use_max_value:
            td_total_array = clipped_td_total_array

        gae_array = gae_array + (lambd**(i)) * clipped_td_total_array * (1 - is_done_array) + ((lambd**(i+1))/(1 - lambd)) * clipped_td_total_array * done * (1 - is_done_array)
        is_done_array = jnp.logical_or(is_done_array, done)

        return (gae_array, td_total_array, is_done_array, cur_idx + 1), None

    carry, _ = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(values), jnp.zeros_like(values), jnp.zeros_like(dones), 0),
        (deltas, dones),
        unroll=16,
    )
    gae_array, td_total_array, is_done_array, cur_idx = carry
    i = (cur_idx - start_index)[..., None]
    clipped_td_total_array = jnp.minimum(td_total_array, 0)
    advantages = (1 - lambd)*(gae_array + ((lambd**(i))/(1 - lambd)) * clipped_td_total_array * (1 - is_done_array))

    return advantages, advantages + values