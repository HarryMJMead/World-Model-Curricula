from typing import Tuple, Any, Dict, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
import functools
import chex
import optax
from flax.linen.initializers import orthogonal, constant, glorot_uniform
from flax.linen.linear import PrecisionLike
from jaxued.environments.maze.env import Observation
from .prediction_net import PredictionNet, OutHead
from .utils import add_uniform_noise, two_hot

class Encoder(nn.Module):
    """Simple CNN encoder -> embedding vector"""
    z_dim: int
    hidden: int = 256
    layers: int = 3
    activation: str = "relu"
    conv_channels: int = 0
    conv_layers: int = 0
    use_layer_norm: bool = False
    image_only: bool = False
    include_key: bool = False

    def setup(self):
        if self.activation == "relu":
            self.activation_fn = nn.relu
        elif self.activation == "gelu":
            self.activation_fn = nn.gelu
        else:
            raise ValueError(f"Invalid activation function: {self.activation}")

    @nn.compact
    def __call__(self, obs: Observation, deterministic: bool = True, action: Optional[chex.Array] = None):
        x = obs.image
        if self.conv_layers > 0:
            for _ in range(self.conv_layers):
                x = nn.Conv(self.conv_channels, kernel_size=(1, 1), strides=(1, 1), padding="VALID")(x)
                x = self.activation_fn(x)
        x = x.reshape(*x.shape[:-3], -1)
        if not self.image_only:
            if self.include_key:
                dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
                key_embed = jax.nn.one_hot(obs.has_key.astype(jnp.uint8), 2)
                x = jnp.concatenate((x, dir_embed, key_embed), axis=-1)
            else:
                dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
                x = jnp.concatenate((x, dir_embed), axis=-1)
            
            if action != None:
                action_embed = jax.nn.one_hot(action, 4) # Assuming 4 actions - could be passed in as a parameter
                x = jnp.concatenate((x, action_embed), axis=-1)

        for _ in range(self.layers):
            x = nn.Dense(self.hidden, kernel_init=orthogonal(jnp.sqrt(2)))(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation_fn(x)

        x = nn.Dense(self.z_dim, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        return x

class Decoder(nn.Module):
    """Image decoder from combined latent (det + cat)"""
    out_shape: Tuple[int,int,int]
    hidden: int = 256
    layers: int = 2
    activation: str = "relu"
    conv_channels: int = 0
    conv_layers: int = 0
    use_layer_norm: bool = False
    image_only: bool = False
    include_key: bool = False

    def setup(self):
        if self.activation == "relu":
            self.activation_fn = nn.relu
        elif self.activation == "gelu":
            self.activation_fn = nn.gelu
        else:
            raise ValueError(f"Invalid activation function: {self.activation}")            


    @nn.compact
    def __call__(self, x: chex.Array, return_logits: bool = False, deterministic: bool = True):
        for _ in range(self.layers):
            x = nn.Dense(self.hidden, kernel_init=orthogonal(jnp.sqrt(2)))(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation_fn(x)

        if self.conv_layers > 0:
            image = nn.Dense(self.out_shape[0]*self.out_shape[1]*self.conv_channels)(x)
            image = self.activation_fn(image)
            image = image.reshape(*image.shape[:-1], *self.out_shape[:2], self.conv_channels)
            if self.conv_layers > 1:    
                for _ in range(self.conv_layers - 1):
                    image = nn.Conv(self.conv_channels, kernel_size=(1, 1), strides=(1, 1), padding="VALID")(image)
            image = nn.Conv(self.out_shape[2], kernel_size=(1, 1), strides=(1, 1), padding="VALID")(image)
        else:
            image = nn.Dense(self.out_shape[0]*self.out_shape[1]*self.out_shape[2])(x)
            image = image.reshape(*image.shape[:-1], *self.out_shape)
        if self.image_only:
            return Observation(image=image, agent_dir=None)
        else:
            dir_logits = nn.Dense(4)(x)
            dir = dir_logits.argmax(axis=-1)
            if self.include_key:
                key_logits = nn.Dense(2)(x)
                has_key = key_logits.argmax(axis=-1).astype(bool)
                recon_obs = Observation(image=image, agent_dir=dir, has_key=has_key)
                logits = (dir_logits, key_logits)
            else:
                recon_obs = Observation(image=image, agent_dir=dir)
                logits = dir_logits
            
            if return_logits:
                return recon_obs, logits
            else:
                return recon_obs

    
class DynamicsModel(nn.Module):
    config: Dict
    z_dim: int
    num_actions: int
    
    decode: bool = False
    latent_gen_dim: int = 0
    hybrid_gen: bool = False

    precision: PrecisionLike = None
    dtype: Optional[Any] = None

    def setup(self):
        config = self.config
        embeds = {
            'z': {'categorical': False, 'in_dim': self.z_dim},
            'action': {'categorical': True, 'in_dim': self.num_actions}
        }

        if config["use_encoded_outputs"]:
            self.num_bins = config['num_reward_bins']
            self.reward_bins = jnp.linspace(config['min_reward_bin'], config['max_reward_bin'], self.num_bins)
            reward_dim = self.num_bins
            done_dim = 2
        else:
            reward_dim = 1
            done_dim = 1

        # Currently manually setting out_head hiddem dims here
        out_heads = {
            'next_z': {'out_dim': self.z_dim, 'hidden_dim': 512, 'num_layers': config['out_heads_num_layers']},
            'reward': {'out_dim': reward_dim, 'hidden_dim': 256, 'num_layers': config['out_heads_num_layers']},
            'done': {'out_dim': done_dim, 'hidden_dim': 256, 'num_layers': config['out_heads_num_layers']}
        }

        # CURRENTLY - LATENT GEN ACTIONS vs LATENT GEN ACTION depending on if hybrid or not
        if self.latent_gen_dim > 0:
            if self.hybrid_gen:
                out_heads.update({
                    'latent_gen_actions': {'out_dim': self.latent_gen_dim, 'hidden_dim': 512, 'num_layers': config['out_heads_num_layers']},
                    'output_select': {'out_dim': 2, 'hidden_dim': 512, 'num_layers': config['out_heads_num_layers']}
                })
            else:
                out_heads = {
                    'latent_gen_action': {'out_dim': self.latent_gen_dim, 'hidden_dim': 512, 'num_layers': config['out_heads_num_layers']}
                }

        modality_order = ('action', 'z')
        num_modalities = len(modality_order)

        self.prediction_net = PredictionNet(
            embed_dim=config["embed_dim"],
            num_modalities=num_modalities,
            embeds=embeds,
            out_heads=out_heads,
            modality_order=modality_order,
            dropout_p=config["dropout_p"],
            feedforward_dim=config["feedforward_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            normalize_qk=config["normalize_qk"],
            normalize_out_heads=config["normalize_out_heads"],
            decode=self.decode,
            precision=self.precision,
            dtype=self.dtype,
            context_size=config['context_size'],
            concat_tokens=config["concat_tokens"],
            sum_tokens=config["sum_tokens"],
        )
    
    @nn.compact
    def __call__(self, z, action, return_logits: bool = False, deterministic=True):
        inputs = {'z': z, 'action': action}

        out, hiddens = self.prediction_net(inputs, deterministic=deterministic)
        dynamics_outputs = {'hiddens': hiddens}

        if self.latent_gen_dim > 0:
            if self.hybrid_gen:
                dynamics_outputs['latent_gen_actions'] = out['latent_gen_actions'] # ACTION/ACTIONS BODGE - need to fix
                dynamics_outputs['output_select'] = out['output_select']
            else:
                dynamics_outputs['latent_gen_actions'] = out['latent_gen_action'] # ACTION/ACTIONS BODGE - need to fix
                return dynamics_outputs

        if self.config['use_encoded_outputs']:
            reward_logits = out['reward']
            reward_probs = nn.softmax(reward_logits)
            reward = (reward_probs * self.reward_bins).sum(axis=-1)
            done_logits = out['done']
            done = done_logits.argmax(axis=-1)

            if return_logits:
                dynamics_outputs['logits'] = (reward_logits, done_logits)
        else:
            reward = out['reward'][..., 0]
            done_logits = out['done'][..., 0]
            done = nn.sigmoid(done_logits)

        dynamics_outputs['outputs'] = (out['next_z'], reward, done)
        return dynamics_outputs


class WorldModel(nn.Module):
    image_shape: Tuple[int,int,int]
    config: Dict
    no_dynamics: bool = False
    decode: bool = False
    precision: PrecisionLike = None
    dtype: Optional[Any] = None

    @property
    def latent_gen_dim(self):
        return 0

    def setup(self):
        config = self.config

        self.num_latents = config['num_latents']
        self.num_cats = config['num_cats']
        self.z_dim = self.num_latents * self.num_cats
        self.hybrid_gen = config["use_hybrid_gen"]

        self.encoder = Encoder(
            hidden=config['encoder_hidden'],
            layers=config['encoder_layers'],
            activation=config['encoder_activation'],
            use_layer_norm=config["encoder_normalize"],
            conv_layers=config["encoder_conv_layers"],
            conv_channels=config["encoder_conv_channels"],
            z_dim=self.z_dim, 
            include_key=config['include_key']
        )
        self.decoder = Decoder(
            hidden=config['decoder_hidden'],
            layers=config['decoder_layers'],
            activation=config["decoder_activation"],
            use_layer_norm=config["decoder_normalize"],
            conv_layers=config["decoder_conv_layers"],
            conv_channels=config["decoder_conv_channels"],
            out_shape=self.image_shape, 
            include_key=config['include_key']
        )
        
        if config['use_frozen_encoder']:
            self.encoder = self._get_stopgrad_module(self.encoder)
            self.decoder = self._get_stopgrad_module(self.decoder)
        
        if not self.no_dynamics:
            self.dynamics_model = DynamicsModel(
                config=self.config,
                z_dim=self.z_dim,
                num_actions=self.config['num_actions'],
                decode=self.decode,
                latent_gen_dim=self.latent_gen_dim,
                hybrid_gen=self.hybrid_gen,
                precision=self.precision,
                dtype=self.dtype,
            )


    def _get_stopgrad_module(self, module):
        def _stopgrad_module(*args, **kwargs):
            return jax.lax.stop_gradient(module(*args, **kwargs))
        return _stopgrad_module
    

    def _add_uniform_noise(self, dist, mix, vary_noise=False, rng=None):
        noise = distrax.Categorical(logits=jnp.zeros_like(dist.probs))
        if vary_noise:
            mix = jax.random.uniform(rng, dist.probs.shape[:-2], maxval=mix)[..., None, None]
        return distrax.Categorical(probs=(1 - mix) * dist.probs + mix * noise.probs)


    def _recon_loss(self, obs: Observation, recon_obs: Observation, dir_logits=None, key_logits=None):
        loss = jnp.mean(((recon_obs.image - obs.image) ** 2).sum(axis=(2,3,4)))
        if dir_logits != None:
            loss += optax.softmax_cross_entropy_with_integer_labels(dir_logits, obs.agent_dir).mean()
        if key_logits != None:
            loss += optax.softmax_cross_entropy_with_integer_labels(key_logits, obs.has_key.astype(jnp.uint8)).mean()
        return loss
    
    def _dyn_loss(self, z_dist: distrax.Distribution, pred_z_dist: distrax.Distribution):
        return ((jax.lax.stop_gradient(z_dist).kl_divergence(pred_z_dist)).sum(axis=-1)).mean()
    
    def _pred_loss(self, dynamics_outputs, rewards, dones):
        num_time_steps = rewards.shape[0]
        pred_next_z_logits, pred_reward, pred_done = dynamics_outputs['outputs']
        if self.config["use_encoded_outputs"]:
            pred_reward_logits, pred_done_logits = dynamics_outputs['logits']
            two_hot_rewards = two_hot(rewards, self.dynamics_model.reward_bins)
            pred_reward_probs = nn.softmax(pred_reward_logits)
            reward_loss = -(two_hot_rewards * jnp.log(pred_reward_probs[:num_time_steps])).mean()
            done_loss = optax.softmax_cross_entropy_with_integer_labels(pred_done_logits[:num_time_steps], dones.astype(jnp.int32)).mean()
            pred_loss = reward_loss + done_loss
        else:
            pred_loss = jnp.mean((pred_reward[:num_time_steps] - rewards) ** 2) + jnp.mean((pred_done[:num_time_steps] - dones) ** 2)

        pred_next_z_logits = pred_next_z_logits.reshape(*pred_next_z_logits.shape[:-1], self.num_latents, self.num_cats)
        pred_next_z_dist = distrax.Categorical(logits=pred_next_z_logits)

        return pred_next_z_dist, pred_reward, pred_done, pred_loss

    def _get_z(self, rng, obs, deterministic=True):
        rng_noise, rng_sample = jax.random.split(rng)
        z_logits = self.encoder(obs, deterministic=deterministic)
        z_logits = z_logits.reshape(*z_logits.shape[:-1], self.num_latents, self.num_cats)
        z_dist = distrax.Categorical(logits=z_logits)
        z_dist_noised = self._add_uniform_noise(z_dist, self.config['noise_z'], vary_noise=self.config['vary_z_noise'], rng=rng_noise)

        # SAMPLE Z
        if deterministic:
            z_sample = jax.nn.one_hot(z_dist.mode(), self.num_cats)
        else:
            z_sample = jax.nn.one_hot(z_dist_noised.sample(seed=rng_sample), self.num_cats)

        z_sample += (z_dist_noised.probs - jax.lax.stop_gradient(z_dist_noised.probs))
        z = z_sample.reshape(*z_sample.shape[:-2], -1)
        return z, z_dist
    
    def _get_recon(self, obs, z, deterministic=True):
        recon_obs, logits = self.decoder(z, return_logits=True, deterministic=deterministic)
        if self.config["include_key"]:
            dir_logits, key_logits = logits
        else:
            dir_logits = logits
            key_logits = None
        recon_loss = self._recon_loss(obs, recon_obs, dir_logits, key_logits)
        return recon_obs, recon_loss

    @nn.compact
    def __call__(self, rng, obs, actions, rewards, dones, deterministic=True):
        z, z_dist = self._get_z(rng, obs, deterministic=deterministic)

        recon_obs, recon_loss = self._get_recon(obs, z, deterministic=deterministic)

        if self.no_dynamics:
            return z_dist, recon_obs, recon_loss

        dynamics_outputs = self.dynamics_model(z, actions, return_logits=True, deterministic=deterministic)
        
        pred_next_z_dist, pred_reward, pred_done, pred_loss = self._pred_loss(dynamics_outputs, rewards, dones)
        dyn_loss = self._dyn_loss(z_dist[1:], pred_next_z_dist[:-1])

        return (z_dist, pred_next_z_dist), (recon_obs, pred_reward, pred_done), (recon_loss, pred_loss, dyn_loss)


class WorldModelDecode(WorldModel):
    @nn.compact
    def __call__(self, z, actions, return_dist=False):
        dynamics_outputs = self.dynamics_model(z, actions, return_logits=True)
        pred_next_z_logits, pred_reward, pred_done = dynamics_outputs['outputs']
        
        pred_next_z_logits = pred_next_z_logits.reshape(*pred_next_z_logits.shape[:-1], self.num_latents, self.num_cats)
        pred_next_z_dist = distrax.Categorical(logits=pred_next_z_logits)

        pred_next_z_sample = jax.nn.one_hot(pred_next_z_dist.mode(), self.num_cats)
        pred_next_z = pred_next_z_sample.reshape(*pred_next_z_sample.shape[:-2], -1)

        if return_dist:
            return pred_next_z, pred_reward, pred_done, pred_next_z_dist
        return pred_next_z, pred_reward, pred_done


class LatentDecoder(nn.Module):
    z_dim: int
    config: Dict

    def setup(self):
        if self.config['use_output_heads']:
            if self.config["use_encoded_outputs"]:
                self.num_bins = self.config['num_reward_bins']
                self.reward_bins = jnp.linspace(self.config['min_reward_bin'], self.config['max_reward_bin'], self.num_bins)
                reward_dim = self.num_bins
                done_dim = 2
            else:
                reward_dim = 1
                done_dim = 1

            out_heads = {
                'next_z': {'out_dim': self.z_dim, 'hidden_dim': 512, 'num_layers': self.config['out_heads_num_layers']},
                'reward': {'out_dim': reward_dim, 'hidden_dim': 256, 'num_layers': self.config['out_heads_num_layers']},
                'done': {'out_dim': done_dim, 'hidden_dim': 256, 'num_layers': self.config['out_heads_num_layers']}
            }

            if self.config['use_hybrid_override']:
                out_heads['output_select'] = {'out_dim': 2, 'hidden_dim': 512, 'num_layers': self.config['out_heads_num_layers']}


            self.out_head_mods = {
                name: OutHead(out_head["out_dim"], out_head["hidden_dim"], out_head['num_layers'], self.config["normalize_out_heads"])
                for name, out_head in out_heads.items()
            }
        

    @nn.compact
    def __call__(self, latent_gen_actions, hiddens, return_logits: bool = False):
        latent_gen_outputs = {}
        x = jnp.concatenate([latent_gen_actions, hiddens], axis=-1)
        if self.config['use_output_heads']:
            next_z = self.out_head_mods['next_z'](x)

            if self.config['use_hybrid_override']:
                x_stop_grad = jnp.concatenate([jax.lax.stop_gradient(latent_gen_actions), hiddens], axis=-1)
                latent_gen_outputs['output_select'] = self.out_head_mods['output_select'](x_stop_grad) # Set this to hiddens as a test

            if self.config['use_encoded_outputs']:
                reward_logits = self.out_head_mods['reward'](x)
                reward_probs = nn.softmax(reward_logits)
                reward = (reward_probs * self.reward_bins).sum(axis=-1)
                done_logits = self.out_head_mods['done'](x)
                done = done_logits.argmax(axis=-1)

                if return_logits:
                    latent_gen_outputs['logits'] = (reward_logits, done_logits)
            else:
                reward = (self.out_head_mods['reward'](x))[..., 0]
                done_logits = (self.out_head_mods['done'](x))[..., 0]
                done = nn.sigmoid(done_logits)
        else:
            next_z = nn.Dense(self.z_dim, kernel_init=orthogonal(jnp.sqrt(2)))(x)
            reward = nn.Dense(1, kernel_init=orthogonal(jnp.sqrt(2)))(x)[..., 0]
            done = nn.Dense(1, kernel_init=orthogonal(jnp.sqrt(2)))(x)[..., 0]
            done = nn.sigmoid(done)
        
        latent_gen_outputs['outputs'] = (next_z, reward, done)
        return latent_gen_outputs 


class WorldModelwLatent(WorldModel):
    @property
    def latent_gen_dim(self):
        return self.config['num_gen_latents'] * self.config['num_gen_cats']

    def setup(self):
        self.num_gen_latents = self.config['num_gen_latents']
        self.num_gen_cats = self.config['num_gen_cats']

        super().setup()
        if not self.config["use_history_for_latent_encoder"]:
            self.latent_encoder = Encoder(
                hidden=self.config['encoder_hidden'],
                layers=self.config['encoder_layers'],
                activation=self.config['encoder_activation'],
                use_layer_norm=self.config["encoder_normalize"],
                conv_layers=self.config["encoder_conv_layers"],
                conv_channels=self.config["encoder_conv_channels"],
                z_dim=self.config['num_gen_latents']*self.config['num_gen_cats'], 
                include_key=self.config['include_key']
            )
        self.latent_decoder = LatentDecoder(z_dim=self.z_dim, config=self.config)
    

    def _get_latent_gen_actions(self, rng, latent_gen_logits, hiddens, deterministic=True, gen_first_obs=False):
        latent_gen_logits = latent_gen_logits.reshape(*latent_gen_logits.shape[:-1], self.num_gen_latents, self.num_gen_cats)
        latent_gen_dist = distrax.Categorical(logits=latent_gen_logits)

        # Add noise to latent gen distribution
        latent_gen_dist = self._add_uniform_noise(latent_gen_dist, self.config['noise_latent_gen'])

        if deterministic:
            latent_gen_sample = jax.nn.one_hot(latent_gen_dist.mode(), self.num_gen_cats)
        else:
            latent_gen_sample = jax.nn.one_hot(latent_gen_dist.sample(seed=rng), self.num_gen_cats)

        latent_gen_sample += (latent_gen_dist.probs - jax.lax.stop_gradient(latent_gen_dist.probs))
        latent_gen_actions = latent_gen_sample.reshape(*latent_gen_sample.shape[:-2], -1)

        # Get Latent Gen Actions for next step - if training to generate the first obs, return all latent gen actions
        if gen_first_obs:
            return latent_gen_actions, hiddens[:-1]
        # Otherwise, ignore the first latent gen action
        return latent_gen_actions[1:], hiddens[:-1]
    
    def _select_loss(self, z_dist, pred_next_z_dist, pred_next_z_dist_alt, output_select, gen_first_obs):
        compare_length = z_dist.probs.shape[0] - 1

        # Get Mode Accuracies
        accuracy = jax.lax.stop_gradient(pred_next_z_dist[gen_first_obs:gen_first_obs+compare_length].mode() == z_dist[1:].mode()).mean(axis=-1)
        accuracy_alt = jax.lax.stop_gradient(pred_next_z_dist_alt[:compare_length].mode() == z_dist[1:].mode()).mean(axis=-1)
        
        # Determine which is more accurate, and get selector loss
        use_alt_true = accuracy_alt >= accuracy
        select_loss = optax.softmax_cross_entropy_with_integer_labels(output_select[:compare_length], use_alt_true.astype(jnp.int32)).mean()

        # Get predicted selections
        use_alt = output_select.argmax(axis=-1).astype(bool)
        return (use_alt, use_alt_true), select_loss
    
    def _select_outputs(self, pred, pred_alt, selector):
        extra_dims = pred.ndim - selector.ndim
        selector = selector.reshape(selector.shape + (1,)*extra_dims)
        pred_alt = pred_alt[:pred.shape[0]]
        selector = selector[:pred.shape[0]]
        return jax.lax.stop_gradient(jnp.where(selector, pred_alt, pred))

    @nn.compact
    def __call__(self, rng, obs, actions, rewards, dones, latent_gen_actions=None, deterministic=True, gen_first_obs=False):
        rng_z, rng_gen = jax.random.split(rng)
        z, z_dist = self._get_z(rng_z, obs, deterministic=deterministic)

        recon_obs, recon_loss = self._get_recon(obs, z, deterministic=deterministic)

        dynamics_outputs = self.dynamics_model(z, actions, return_logits=True, deterministic=deterministic)
        latent_gen_logits = dynamics_outputs['latent_gen_actions']
        hiddens = dynamics_outputs['hiddens']

        if not self.config["use_history_for_latent_encoder"]:
            prev_actions = jnp.roll(actions, 1, axis=0) # Shift actions one step forward
            latent_gen_logits = self.latent_encoder(obs, action=prev_actions, deterministic=deterministic)

        if latent_gen_actions == None:
            latent_gen_actions, hiddens = self._get_latent_gen_actions(rng_gen, latent_gen_logits, hiddens, deterministic=deterministic, gen_first_obs=gen_first_obs)
        
        if gen_first_obs:
            # Calculate dynamic loss for all observations
            z_dist_offset = 0

            # Pad the first hidden state with zeros
            hiddens = jnp.pad(hiddens, ((1, 0), (0, 0), (0, 0)))
            latent_decoder_outputs = self.latent_decoder(latent_gen_actions, hiddens, return_logits=True)

            # Discard all but the state for the first prediction
            pred_next_z_logits = latent_decoder_outputs['outputs'][0]
            pred_reward_done_outputs = latent_decoder_outputs['outputs'][1:]
            pred_reward_done_outputs, pred_reward_done_logits = jax.tree_map(lambda x: x[1:], (pred_reward_done_outputs, latent_decoder_outputs['logits']))
            latent_decoder_outputs['outputs'] = (pred_next_z_logits,) + pred_reward_done_outputs
            latent_decoder_outputs['logits'] = pred_reward_done_logits
        else:
            # Ignore the first observation when prediction dynamic loss
            z_dist_offset = 1
            latent_decoder_outputs = self.latent_decoder(latent_gen_actions, hiddens, return_logits=True)

        pred_next_z_dist, pred_reward, pred_done, pred_loss = self._pred_loss(latent_decoder_outputs, rewards, dones)
        dyn_loss = self._dyn_loss(z_dist[z_dist_offset:], pred_next_z_dist[:z_dist.probs.shape[0]-z_dist_offset])

        if self.hybrid_gen:
            # Compute the predicted next z distribution, reward, and done from the dynamics model output (no latent gen actions)
            pred_next_z_dist_alt, pred_reward_alt, pred_done_alt, pred_loss_alt = self._pred_loss(dynamics_outputs, rewards, dones)
            dyn_loss_alt = self._dyn_loss(z_dist[1:], pred_next_z_dist_alt[:-1])
            (use_alt, use_alt_true), select_loss = self._select_loss(z_dist, pred_next_z_dist, pred_next_z_dist_alt, dynamics_outputs['output_select'], gen_first_obs)

            if self.config['use_hybrid_override']:
                latent_decoder_dynamics_select = latent_decoder_outputs['output_select'][gen_first_obs:]
                (use_alt_override, _), select_override_loss = self._select_loss(z_dist, pred_next_z_dist, pred_next_z_dist_alt, latent_decoder_dynamics_select, gen_first_obs)
                output_switch = jnp.logical_or(use_alt[:use_alt_override.shape[0]], use_alt_override)
            else:
                use_alt_override = select_override_loss = 0
                output_switch = use_alt

            # Select which outputs to use - when use_alt is True, select the outputs from the dynamics model, otherwise select the outputs from the latent decoder
            (pred_next_z_dist_hybrid, pred_reward_hybrid, pred_done_hybrid) = jax.tree_map(
                lambda pred, pred_alt: self._select_outputs(pred, pred_alt, output_switch),
                (pred_next_z_dist[gen_first_obs:], pred_reward, pred_done),
                (pred_next_z_dist_alt, pred_reward_alt, pred_done_alt)
            )
            return (z_dist, pred_next_z_dist_hybrid), (recon_obs, pred_reward_hybrid, pred_done_hybrid), (use_alt, use_alt_true, use_alt_override), (recon_loss, pred_loss, dyn_loss, pred_loss_alt, dyn_loss_alt, select_loss, select_override_loss)

        return (z_dist, pred_next_z_dist), (recon_obs, pred_reward, pred_done), (recon_loss, pred_loss, dyn_loss)


class WorldModelDynamicsDecode(WorldModelwLatent):
    @nn.compact
    def __call__(self, z, actions):
        return self.dynamics_model(z, actions, return_logits=True)

    
class WorldModelLatentDecode(WorldModelwLatent):
    @nn.compact
    def __call__(self, dynamics_outputs, latent_gen_actions, return_dist=False, return_use_alt=False):
        hiddens = dynamics_outputs['hiddens']

        latent_decoder_outputs = self.latent_decoder(latent_gen_actions, hiddens)
        pred_next_z_logits, pred_reward, pred_done = latent_decoder_outputs['outputs']

        if self.hybrid_gen:
            pred_next_z_logits_alt, pred_reward_alt, pred_done_alt = dynamics_outputs['outputs']
            use_alt = dynamics_outputs['output_select'].argmax(axis=-1).astype(bool)

            if self.config['use_hybrid_override']:
                use_alt_override = latent_decoder_outputs['output_select'].argmax(axis=-1).astype(bool)
                output_switch = jnp.logical_or(use_alt, use_alt_override)
            else:
                output_switch = use_alt

            (pred_next_z_logits, pred_reward, pred_done) = jax.tree_map(
                lambda pred, pred_alt: self._select_outputs(pred, pred_alt, output_switch),
                (pred_next_z_logits, pred_reward, pred_done),
                (pred_next_z_logits_alt, pred_reward_alt, pred_done_alt)
            )
        
        pred_next_z_logits = pred_next_z_logits.reshape(*pred_next_z_logits.shape[:-1], self.num_latents, self.num_cats)
        pred_next_z_dist = distrax.Categorical(logits=pred_next_z_logits)

        pred_next_z_sample = jax.nn.one_hot(pred_next_z_dist.mode(), self.num_cats)
        pred_next_z = pred_next_z_sample.reshape(*pred_next_z_sample.shape[:-2], -1)

        if return_dist:
            return pred_next_z, pred_reward, pred_done, pred_next_z_dist, use_alt
        elif return_use_alt:
            return pred_next_z, pred_reward, pred_done, use_alt
        return pred_next_z, pred_reward, pred_done
        


