# networks.py
from typing import Sequence, Callable
import flax.linen as nn
import jax.numpy as jnp
import distrax

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


import flax.linen as nn
import functools as ft
import numpy as np
import jax.nn as jnn
import jax.numpy as jnp

from typing import Type, Tuple
from abc import ABC, abstractproperty, abstractmethod

from glac.networks.distribution import TanhTransformedDistribution, tfd
from glac.utils.typing import Action, Array
from glac.utils.graph import GraphsTuple
from glac.networks.utils import default_nn_init, scaled_init
from glac.networks.gnn import GNN
from glac.networks.mlp import MLP
from glac.utils.typing import PRNGKey, Params


class PolicyDistribution(nn.Module, ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs) -> tfd.Distribution:
        pass

    @abstractproperty
    def nu(self) -> int:
        pass


class TanhNormal(PolicyDistribution):
    base_cls: Type[GNN]
    _nu: int
    scale_final: float = 0.01
    std_dev_min: float = 1e-5
    std_dev_init: float = 0.5

    @property
    def std_dev_init_inv(self):
        # inverse of log(sum(exp())).
        inv = np.log(np.exp(self.std_dev_init) - 1)
        assert np.allclose(np.logaddexp(inv, 0), self.std_dev_init)
        return inv

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(obs, node_type=0, n_type=n_agents)
        # x = x.nodes
        scaler_init = scaled_init(default_nn_init(), self.scale_final)
        feats_scaled = nn.Dense(256, kernel_init=scaler_init, name="ScaleHid")(x)

        means = nn.Dense(self.nu, kernel_init=default_nn_init(), name="OutputDenseMean")(feats_scaled)
        stds_trans = nn.Dense(self.nu, kernel_init=default_nn_init(), name="OutputDenseStdTrans")(feats_scaled)
        stds = jnn.softplus(stds_trans + self.std_dev_init_inv) + self.std_dev_min

        distribution = tfd.Normal(loc=means, scale=stds)
        return tfd.Independent(TanhTransformedDistribution(distribution), reinterpreted_batch_ndims=1)

    @property
    def nu(self):
        return self._nu


class Gnn_net(nn.Module):
    base_cls: Type[GNN]
    _nu: int

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int, *args, **kwargs) -> Action:
        x = self.base_cls()(obs, node_type=0, n_type=n_agents)  # shape: (batch_graphs, n_agents, out_dim)
        return x


class MultiAgentPolicy(ABC):

    def __init__(self, node_dim: int, edge_dim: int, n_agents: int, action_dim: int):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_agents = n_agents
        self.action_dim = action_dim

    @abstractmethod
    def get_action(self, params: Params, obs: GraphsTuple) -> Action:
        pass

    @abstractmethod
    def sample_action(self, params: Params, obs: GraphsTuple, key: PRNGKey) -> Tuple[Action, Array]:
        pass

    @abstractmethod
    def eval_action(self, params: Params, obs: GraphsTuple, action: Action, key: PRNGKey) -> Tuple[Array, Array]:
        pass


class DeterministicPolicy(MultiAgentPolicy):

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            n_agents: int,
            action_dim: int,
            gnn_layers: int = 1,
    ):
        super().__init__(node_dim, edge_dim, n_agents, action_dim)
        self.policy_base = ft.partial(
            GNN,
            msg_dim=128,
            hid_size_msg=(256, 256),
            hid_size_aggr=(128, 128),
            hid_size_update=(256, 256),
            out_dim=128,
            n_layers=gnn_layers
        )
        self.policy_head = ft.partial(
            MLP,
            hid_sizes=(256, 256),
            act=nn.relu,
            act_final=False,
            name='PolicyHead'
        )
        self.net = Deterministic(base_cls=self.policy_base, head_cls=self.policy_head, _nu=action_dim)
        self.std = 0.1

    def get_action(self, params: Params, obs: GraphsTuple) -> Action:
        return self.net.apply(params, obs, self.n_agents)

    def sample_action(self, params: Params, obs: GraphsTuple, key: PRNGKey) -> Tuple[Action, Array]:
        action = self.get_action(params, obs)
        log_pi = jnp.zeros_like(action)
        return action, log_pi

    def eval_action(self, params: Params, obs: GraphsTuple, action: Action, key: PRNGKey) -> Tuple[Array, Array]:
        raise NotImplementedError


class GnnFeatureExtractor(nn.Module):
    """
    A standard Flax module wrapping the GNN feature extractor.
    Takes a graph and extracts features for agent nodes.
    """
    gnn_layers: int = 1
    out_dim: int = 128  # GNN output feature dimension

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int):
        # Configure the GNN using ft.partial
        gnn_base = ft.partial(
            GNN,
            msg_dim=128,
            hid_size_msg=(256, 256),
            hid_size_aggr=(128, 128),
            hid_size_update=(256, 256),
            out_dim=self.out_dim,
            n_layers=self.gnn_layers
        )
        # Instantiate and call GNN
        # GNN output shape: (n_graphs_in_batch, n_agents, out_dim)
        features = gnn_base()(obs, node_type=0, n_type=n_agents)
        return features


class ActorWithGNN(nn.Module):
    """
    Policy network (Actor) using GNN as the feature extractor.
    """
    action_dim: int
    n_agents: int # Required for GNN call even in single-agent setting
    hidden_dims: Sequence[int] = (256, 256)
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, obs: GraphsTuple):
        # 1. Instantiate and call the GNN feature extractor
        extractor = GnnFeatureExtractor(gnn_layers=1, out_dim=128)
        # features shape: (batch_size, n_agents, 128)
        features = extractor(obs, self.n_agents)

        # 2. Pass features through the MLP head
        x = features
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = self.activation(x)
            
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        
        # Return a Normal distribution
        return distrax.Normal(loc=mean, scale=std)


class CriticWithGNN(nn.Module):
    """
    Value network (Critic) using an independent GNN as the feature extractor.
    """
    n_agents: int
    hidden_dims: Sequence[int] = (256, 256)
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, obs: GraphsTuple, actions: jnp.ndarray):
        # actions shape: (batch_size, n_agents, action_dim)
        
        # 1. Instantiate and call the independent GNN feature extractor
        extractor = GnnFeatureExtractor(gnn_layers=1, out_dim=128)
        # state_features shape: (n_agents, 128)
        state_features = extractor(obs, self.n_agents)
        # Path 2: process actions only
        action_path = nn.Dense(128)(actions)
        action_path = self.activation(action_path)
        
        # --- b. Merge the two paths at an intermediate layer ---
        # Merged feature vector x
        x = jnp.concatenate([state_features, action_path], axis=-1)
        #x = self.activation(x)
        # # 2. Concatenate state features and actions
        # #    x shape: (n_agents, 128 + action_dim)
        # x = jnp.concatenate([state_features, actions], axis=-1)
        
        # 3. Pass concatenated features through the MLP head
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = self.activation(x)
        
        # 4. Output Q values
        q_values = nn.Dense(1)(x)  # shape: (n_agents, 1)
        
        return jnp.squeeze(q_values, axis=-1)  # final shape: (batch_size, n_agents)

class DoubleCriticWithGNN(nn.Module):
    """Double GNN Critic"""
    n_agents: int
    hidden_dims: Sequence[int] = (256, 256)
    
    @nn.compact
    def __call__(self, obs: GraphsTuple, actions: jnp.ndarray):
        # Create two independent Critic networks, each with its own GNN
        critic1 = CriticWithGNN(n_agents=self.n_agents, hidden_dims=self.hidden_dims)
        critic2 = CriticWithGNN(n_agents=self.n_agents, hidden_dims=self.hidden_dims)
                            
        q1 = critic1(obs, actions)
        q2 = critic2(obs, actions)
        
        return q1, q2

class LyapunovCritic(nn.Module):
    """
    Value network (Critic) using an independent GNN as the feature extractor.
    """
    n_agents: int
    hidden_dims: Sequence[int] = (256, 256)
    activation: Callable = nn.relu
    @nn.compact
    def __call__(self, obs: GraphsTuple, actions: jnp.ndarray):
        # actions shape: (batch_size, n_agents, action_dim)
        
        # 1. Instantiate and call the independent GNN feature extractor
        extractor = GnnFeatureExtractor(gnn_layers=1, out_dim=128)
        # state_features shape: (n_agents, 128)
        state_features = extractor(obs, self.n_agents)
        # Path 2: process actions only
        action_path = nn.Dense(128)(actions)
        action_path = self.activation(action_path)
        
        # --- b. Merge the two paths at an intermediate layer ---
        # Merged feature vector x
        x = jnp.concatenate([state_features, action_path], axis=-1)
        #x = self.activation(x)
        # # 2. Concatenate state features and actions
        # #    x shape: (n_agents, 128 + action_dim)
        # x = jnp.concatenate([state_features, actions], axis=-1)
        
        # 3. Pass concatenated features through the MLP head
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = self.activation(x)
        
        # 4. Output Lyapunov value (non-negative via relu)
        lyapunov_value = nn.relu(nn.Dense(1)(x))  # shape: (n_agents, 1)
        
        return jnp.squeeze(lyapunov_value, axis=-1)  # final shape: (n_agents,)