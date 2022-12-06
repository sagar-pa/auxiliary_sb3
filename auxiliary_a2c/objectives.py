import torch as th
from torch import nn
import numpy as np
from typing import Callable, Tuple, Dict, List, Any
from torch.distributions import Normal
from abc import ABC

class AuxiliaryObjective(ABC):
    def __init__(self):
        super().__init__()

    def calculate_loss(self, 
            obs: th.Tensor,
            features: th.Tensor,
            actions: th.Tensor,
            returns: th.Tensor,
            infos: List[List[Dict[str, Any]]]) -> th.Tensor:

        """
        Take as input the observations, features (embeddings), 
            actions and infos, and return the loss induced by them.
        """
        raise NotImplementedError


def discount_n_step(x: np.ndarray, n_step: int, gamma: float) -> np.ndarray:
    """
    Taken from RLlib: https://github.com/ray-project/ray/blob/66650cdadbbc19735d7be4bc613b9c3de30a44da/rllib/evaluation/postprocessing.py#L21
    Args:
        x: The array of rewards 
        n_step: The number of steps to look ahead and adjust.
        gamma: The discount factor.

    Examples:
        n-step=3
        Trajectory=o0 r0 d0, o1 r1 d1, o2 r2 d2, o3 r3 d3, o4 r4 d4=True o5
        gamma=0.9
        Returned trajectory:
        0: o0 [r0 + 0.9*r1 + 0.9^2*r2 + 0.9^3*r3] d3 o0'=o3
        1: o1 [r1 + 0.9*r2 + 0.9^2*r3 + 0.9^3*r4] d4 o1'=o4
        2: o2 [r2 + 0.9*r3 + 0.9^2*r4] d4 o1'=o5
        3: o3 [r3 + 0.9*r4] d4 o3'=o5
        4:
    """
    returns = np.array(x, copy=True, dtype=np.float64)
    len_ = returns.shape[0]
    # Change rewards in place.
    for i in range(len_):
        for j in range(1, n_step):
            if i + j < len_:
                returns[i] += (
                    (gamma ** j) * returns[i + j]
                )
    return returns

def discount_n_step_2d(rewards: np.ndarray, n_step: int, gamma: float) -> np.ndarray:
    gammas = gamma ** np.arange(n_step)
    returns = rewards[:, :n_step] @ gammas
    return returns

def compute_discounted_threshold(reward: float, n_step: int, gamma: float) -> float:
    """
    Calculate the n_step threshold associated with the reward, assuming that you'd get this exact reward for n_steps.
    
    Args:
        reward: The undiscounted reward to calculate threshold for.
        n_step: How many steps we'd get this reward for
        gamma: The discount for the n_step
    
    Returns:
        The threshold for the discount
    """
    shape = n_step + 1
    rewards = np.array([reward]*shape)
    return discount_n_step(rewards, n_step, gamma)[0]


def create_minmax_scaler(scale: float, max: float) -> Tuple[Callable, Callable]:
    def normalizer(value):
        value = value * scale
        value = value / max
        value = np.clip(value, 1e-10, 1.) # numerical stability
        return value

    def unnormalizer(value):
        value = value * max
        value = value / scale
        return value

    return normalizer, unnormalizer


class PredictorTail(nn.Module):
    def __init__(self, input_dim: int, net_arch: List[int]):
        super(PredictorTail, self).__init__()
        modules = []
        arch = [input_dim] + net_arch
        for idx in range(len(arch) - 1):
            modules.append(nn.Linear(arch[idx], arch[idx + 1]))
            modules.append(nn.ELU())
        self.shared_net = nn.Sequential(*modules)
        self.mean_predictor = nn.Sequential(
            nn.Linear(arch[-1], 1)
        )
        self.std_predictor = nn.Sequential(
            nn.Linear(arch[-1], 1),
            nn.Softplus()
        )

    def forward(self, data: th.Tensor) -> Normal:
        features = self.shared_net(data)
        mean, std = self.mean_predictor(features), self.std_predictor(features)
        dist = Normal(mean, std)
        return dist


class DecomposedReturnsObjective(AuxiliaryObjective, nn.Module):
    def __init__(self, keys_to_predict: List[str],
            scales: Dict[str, float], 
            maximums: Dict[str, float], 
            n_step: int, 
            gamma: float,
            feature_dim: int,
            action_dim: int,
            shared_net_arch: List[int],
            predictor_net_arch: List[int],
            loss_weights: Dict[str, float],
            device: str = "cpu"):
        """
        A unified implementation of an auxiliary loss that predicts decomposed returns.
        Args:
            keys_to_predict: The keys of the values in the information 
                dictionary (dict returned by the environment) to predict
            scales: The scalers (specified as a dictionary from keys to scaler) 
                to normalize the values with
            maximums: The maximum values of the keys to normalize with 
                (specified as a dictionary from keys to max values)
            n_step: The minimum and maximum n-step to bound the horizon with
            gamma: The discount factor to use for the calculation of returns
            feature_dim: Feature dim of the features (used to init the shared network)
            action_dim: Dimension of the actions (used to init the shared network)
            shared_net_arch: The shared network arch (specified as ints of MLP layer size)
            predictor_net_arch: The network arch used by each predictor 
                (specified as ints of MLP laye size)
            loss_weights: The weights for each predictor in the final loss calculation
                (specified as dict from key to float)
            device: The torch to keep the internal model onto
        
        """
        super().__init__()
        self.keys_to_predict = keys_to_predict
        self.scales = scales
        self.maximums = maximums
        self.n_step = n_step
        self.gamma = gamma
        self.loss_weights = loss_weights
        self.device = device
        self.create_networks(feature_dim = feature_dim, action_dim= action_dim, 
            shared_net_arch= shared_net_arch, predictor_net_arch= predictor_net_arch)
        self.create_normalizers()

    def create_normalizers(self):
        self.normalizers = {}
        self.unnormalizers = {}
        for key in self.keys_to_predict:
            max = self.maximums[key]
            scale = self.scales[key]
            max = compute_discounted_threshold(max, self.n_step, self.gamma)
            normalizer, unnormalizer = create_minmax_scaler(scale, max)
            self.normalizers[key] = normalizer
            self.unnormalizers[key] = unnormalizer

    def create_networks(self, feature_dim: int, action_dim: int, 
            shared_net_arch: List[int], predictor_net_arch: List[int]):
        arch = [feature_dim + action_dim] + shared_net_arch
        predictor_input_dim = shared_net_arch[-1] + action_dim
        modules = []
        for idx in range(len(arch) - 1):
            modules.append(nn.Linear(arch[idx], arch[idx + 1]))
            modules.append(nn.ELU())
        self.shared_net = nn.Sequential(*modules)
        self.shared_net = self.shared_net.to(device = self.device)
        for key in self.keys_to_predict:
            tail = PredictorTail(input_dim= predictor_input_dim, net_arch=predictor_net_arch)
            tail = tail.to(device = self.device)
            setattr(self, f"{key}_tail", tail)
        
    def forward(self, features: th.Tensor, actions: th.Tensor) -> Dict[str, Normal]:
        output = {}
        shared_features = self.shared_net(th.cat((features, actions), dim=1))
        shared_features = th.cat((shared_features, actions), dim=1)
        for key in self.keys_to_predict:
            tail = getattr(self, f"{key}_tail")
            pred = tail(shared_features)
            output[key] = pred
        return output

    def calculate_loss(self, obs: th.Tensor, 
            features: th.Tensor, 
            actions: th.Tensor, 
            returns: th.Tensor, 
            infos: List[List[Dict[str, Any]]]) -> th.Tensor:
        mask = th.from_numpy(np.array([len(info_samples) >= self.n_step for info_samples in infos]))
        masked_features = features[mask]
        masked_actions = actions[mask]
        pred = self.forward(features = masked_features, actions = masked_actions)
        loss = None

        for key in self.keys_to_predict:
            rewards = []
            for i, info_samples in enumerate(infos):
                if not mask[i].item():
                    continue
                rewards.append([info.get(key) for info in info_samples[:self.n_step]])
            facet_returns = discount_n_step_2d(rewards = np.array(rewards), 
                n_step=self.n_step, gamma=self.gamma)
            target = th.from_numpy(self.normalizers[key](facet_returns))
            target = target.to(device= self.device)
            target = target.reshape(-1, 1)
            facet_loss = -th.mean(pred[key].log_prob(target))
            if loss is None:
                loss = self.loss_weights[key] * facet_loss
            else:
                loss += self.loss_weights[key] * facet_loss
        
        return loss
    