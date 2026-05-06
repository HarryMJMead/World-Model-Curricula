from typing import Sequence
import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
import distrax
import chex
from flax import struct

from jaxued.linen import ResetRNN

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    img_only: bool = False
    
    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs

        if self.img_only:
            image = obs
        else:
            image = obs.image
        
        img_embed = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)

        if self.img_only:
            embedding = img_embed
        else:
            dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
            dir_embed = nn.Dense(5, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="scalar_embed")(dir_embed)
            
            embedding = jnp.concatenate((img_embed, dir_embed, obs.has_key[..., None]), axis=-1)

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)
        embedding = nn.LayerNorm()(embedding)

        actor_mean = nn.Dense(256, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.LayerNorm()(actor_mean)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(256, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.LayerNorm()(critic)
        critic = nn.tanh(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)
    
    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))
    

class AdversaryActorCritic(nn.Module):
    num_latents: int = 4
    num_cats: int = 4
    max_timesteps: int = 256
    num_actions: int = 4

    use_pro_obs: bool = False

    def _embed_obs(self, obs):
        embedding = nn.Dense(256, kernel_init=orthogonal(2), bias_init=constant(0.0), name="embed0")(obs.hidden)
        embedding = nn.LayerNorm()(embedding)
        embedding = nn.tanh(embedding)

        # New hidden obs embedding
        time_embed = nn.Embed(self.max_timesteps + 1, 10, name="time_embed", embedding_init=orthogonal(1.0))(jnp.clip(obs.step_num, None, self.max_timesteps)) 
        action_embed = nn.Embed(self.num_actions, 5, name="action_embed", embedding_init=orthogonal(1.0))(obs.agent_action)
        embedding = jnp.concatenate((embedding, time_embed, action_embed, obs.agent_value[..., None]), axis=-1)

        return embedding
    
    def _embed_pro_obs(self, obs):
        img_embed = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)

        time_embed = nn.Embed(self.max_timesteps + 1, 10, name="time_embed", embedding_init=orthogonal(1.0))(jnp.clip(obs.step_num, None, self.max_timesteps)) 
        dir_embed = nn.Embed(4, 5, name="dir_embed", embedding_init=orthogonal(1.0))(obs.agent_dir) 
        action_embed = nn.Embed(self.num_actions, 5, name="action_embed", embedding_init=orthogonal(1.0))(obs.agent_action)
        
        embedding = jnp.concatenate((img_embed, time_embed, dir_embed, action_embed, obs.has_key[..., None], obs.agent_value[..., None]), axis=-1) 
        return embedding


    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs

        if self.use_pro_obs:
            embedding = self._embed_pro_obs(obs)
        else:
            embedding = self._embed_obs(obs)

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)
        embedding = nn.LayerNorm()(embedding)

        actor_mean = nn.Dense(256, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.LayerNorm()(actor_mean)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(self.num_latents*self.num_cats, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)
        actor_mean = actor_mean.reshape(*actor_mean.shape[:-1], self.num_latents, self.num_cats)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(256, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.LayerNorm()(critic)
        critic = nn.tanh(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)
    
    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))