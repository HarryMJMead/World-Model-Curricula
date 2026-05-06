import jax
import chex
import jax.numpy as jnp
from typing import Optional, Tuple
from flax.training.train_state import TrainState

def masked_mean(arr, mask=None):
    if mask is None:
        return jnp.mean(arr)
    return jnp.sum(arr * mask) / jnp.sum(mask)

def masked_std(arr, mask=None):
    if mask is None:
        return jnp.std(arr)
    mean = masked_mean(arr, mask)
    var = jnp.sum(mask*(arr - mean)**2)/jnp.sum(mask)
    return jnp.sqrt(var)

def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    batch: chex.ArrayTree,
    num_envs: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    init_state_entropy_coeff: float = 0.0,
    update_grad: bool = True,
    multi_dim_actions: bool = False,
    action_mask: Optional[chex.Array] = None,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """This function takes in a rollout, and PPO hyperparameters, and updates the train state.

    Args:
        rng (chex.PRNGKey): 
        train_state (TrainState): 
        init_hstate (chex.ArrayTree): 
        batch (chex.ArrayTree): obs, actions, dones, log_probs, values, targets, advantages
        num_envs (int): 
        n_steps (int): 
        n_minibatch (int): 
        n_epochs (int): 
        clip_eps (float): 
        entropy_coeff (float): 
        critic_coeff (float): 
        update_grad (bool, optional): If False, the train state does not actually get updated. Defaults to True.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]: It returns a new rng, the updated train_state, and the losses. The losses have structure (loss, (l_vf, l_clip, entropy))
    """
    obs, actions, dones, log_probs, values, targets, advantages = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages, action_mask
    
    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            init_hstate, obs, actions, last_dones, log_probs, values, targets, advantages, action_mask = minibatch
            
            def loss_fn(params):
                _, pi, values_pred = train_state.apply_fn(params, (obs, last_dones), init_hstate)
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy()

                init_state_entropy = entropy[0].mean()

                log_ratio = log_probs_pred - log_probs
                if multi_dim_actions:
                    log_ratio = log_ratio.sum(axis=-1)
                    entropy = entropy.sum(axis=-1)
                entropy = masked_mean(entropy, action_mask)

                ratio = jnp.exp(log_ratio)
                A = (advantages - masked_mean(advantages, action_mask)) / (masked_std(advantages, action_mask) + 1e-5)
                l_clip = masked_mean(-jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A), action_mask)

                values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
                l_vf = 0.5 * jnp.maximum((values_pred - targets) ** 2, (values_pred_clipped - targets) ** 2).mean()

                loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy - init_state_entropy_coeff * init_state_entropy

                approx_kl = jax.lax.stop_gradient(
                    masked_mean((ratio - 1) - log_ratio, action_mask)
                )
                clipfrac = jax.lax.stop_gradient(
                    masked_mean(jnp.abs(ratio - 1) > clip_eps, action_mask)
                )

                return loss, (l_vf, l_clip, entropy, approx_kl, clipfrac, masked_mean(ratio, action_mask), init_state_entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, num_envs)
        minibatches = (
            jax.tree_map(
                lambda x: jnp.take(x, permutation, axis=0)
                .reshape(n_minibatch, -1, *x.shape[1:]),
                init_hstate,
            ),
            *jax.tree_map(
                lambda x: jnp.take(x, permutation, axis=1)
                .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
                .swapaxes(0, 1),
                batch,
            ),
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)


def update_actor_critic(
    rng: chex.PRNGKey,
    train_state: TrainState,
    batch: chex.ArrayTree,
    num_envs: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool = True,
    multi_dim_actions: bool = False,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """This function takes in a rollout, and PPO hyperparameters, and updates the train state.

    Args:
        rng (chex.PRNGKey): 
        train_state (TrainState): 
        init_hstate (chex.ArrayTree): 
        batch (chex.ArrayTree): obs, actions, dones, log_probs, values, targets, advantages
        num_envs (int): 
        n_steps (int): 
        n_minibatch (int): 
        n_epochs (int): 
        clip_eps (float): 
        entropy_coeff (float): 
        critic_coeff (float): 
        update_grad (bool, optional): If False, the train state does not actually get updated. Defaults to True.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]: It returns a new rng, the updated train_state, and the losses. The losses have structure (loss, (l_vf, l_clip, entropy))
    """
    
    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            obs, actions, log_probs, values, targets, advantages = minibatch
            
            def loss_fn(params):
                pi, values_pred = train_state.apply_fn(params, obs)
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()

                log_prob_diff = log_probs_pred - log_probs
                if multi_dim_actions:
                    log_prob_diff = log_prob_diff.sum(axis=-1)

                ratio = jnp.exp(log_prob_diff)
                A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                l_clip = (-jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)).mean()

                values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
                l_vf = 0.5 * jnp.maximum((values_pred - targets) ** 2, (values_pred_clipped - targets) ** 2).mean()

                loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy

                return loss, (l_vf, l_clip, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, num_envs)
        minibatches = jax.tree_map(
            lambda x: jnp.take(x, permutation, axis=1)
            .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
            .swapaxes(0, 1),
            batch,
        )

        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)


def update_world_model(
    rng: chex.PRNGKey,
    train_state: TrainState,
    batch: chex.ArrayTree,
    num_envs: int,
    n_minibatch: int,
    n_epochs: int,
    entropy_coeff: float,
    update_grad: bool=True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """This function takes in a rollout, and PPO hyperparameters, and updates the train state.

    Args:
        rng (chex.PRNGKey): 
        train_state (TrainState): 
        init_hstate (chex.ArrayTree): 
        batch (chex.ArrayTree): obs, actions, dones, log_probs, values, targets, advantages
        num_envs (int): 
        n_steps (int): 
        n_minibatch (int): 
        n_epochs (int): 
        clip_eps (float): 
        entropy_coeff (float): 
        critic_coeff (float): 
        update_grad (bool, optional): If False, the train state does not actually get updated. Defaults to True.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]: It returns a new rng, the updated train_state, and the losses. The losses have structure (loss, (l_vf, l_clip, entropy))
    """
    
    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            rng, obs, actions, next_obs, dones, rewards = minibatch
            
            def loss_fn(params):
                (z_dist, pred_next_z_dist, next_z_dist), (recon_image, pred_reward, pred_done) = train_state.apply_fn(params, rng, obs.image, actions, next_image=next_obs.image, deterministic=False)

                l_pred = jnp.mean(((recon_image - obs.image) ** 2).sum(axis=(2,3,4))) + jnp.mean((pred_reward - rewards) ** 2) + jnp.mean((pred_done - dones) ** 2)
                l_dyn = ((jax.lax.stop_gradient(next_z_dist).kl_divergence(pred_next_z_dist)).sum(axis=-1)).mean()
                entropy = z_dist.entropy().mean()

                loss = l_pred + l_dyn - entropy_coeff * entropy

                return loss, (l_pred, l_dyn, entropy)
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        rng, train_state = carry
        rng, rng_perm, rng_wm = jax.random.split(rng, 3)
        permutation = jax.random.permutation(rng_perm, num_envs)
        minibatches = (
            jax.random.split(rng_wm, n_minibatch),
            *jax.tree_map(
                lambda x: jnp.take(x, permutation, axis=1)
                .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
                .swapaxes(0, 1),
                batch,
            ),
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)


def update_latent_world_model(
    rng: chex.PRNGKey,
    train_state: TrainState,
    batch: chex.ArrayTree,
    num_envs: int,
    n_minibatch: int,
    n_epochs: int,
    entropy_coeff: float,
    update_grad: bool=True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """This function takes in a rollout, and PPO hyperparameters, and updates the train state.

    Args:
        rng (chex.PRNGKey): 
        train_state (TrainState): 
        init_hstate (chex.ArrayTree): 
        batch (chex.ArrayTree): obs, actions, dones, log_probs, values, targets, advantages
        num_envs (int): 
        n_steps (int): 
        n_minibatch (int): 
        n_epochs (int): 
        clip_eps (float): 
        entropy_coeff (float): 
        critic_coeff (float): 
        update_grad (bool, optional): If False, the train state does not actually get updated. Defaults to True.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]: It returns a new rng, the updated train_state, and the losses. The losses have structure (loss, (l_vf, l_clip, entropy))
    """
    
    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            rng, obs, actions, _, dones, rewards = minibatch
            dones = dones[:-1]
            rewards = rewards[:-1]
            
            def loss_fn(params):
                (z_dist, pred_next_z_dist), (recon_image, pred_reward, pred_done) = train_state.apply_fn(params, rng, obs.image, actions, deterministic=False)
                next_z_dist = z_dist[1:]

                l_pred = jnp.mean(((recon_image - obs.image) ** 2).sum(axis=(2,3,4))) + jnp.mean((pred_reward - rewards) ** 2) + jnp.mean((pred_done - dones) ** 2)
                l_dyn = ((jax.lax.stop_gradient(next_z_dist).kl_divergence(pred_next_z_dist)).sum(axis=-1)).mean()
                entropy = z_dist.entropy().mean()

                loss = l_pred + l_dyn - entropy_coeff * entropy

                return loss, (l_pred, l_dyn, entropy)
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        rng, train_state = carry
        rng, rng_perm, rng_wm = jax.random.split(rng, 3)
        permutation = jax.random.permutation(rng_perm, num_envs)
        minibatches = (
            jax.random.split(rng_wm, n_minibatch),
            *jax.tree_map(
                lambda x: jnp.take(x, permutation, axis=1)
                .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
                .swapaxes(0, 1),
                batch,
            ),
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)