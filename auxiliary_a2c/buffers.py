from typing import NamedTuple, List, Union, Dict, Any, Optional
import torch as th
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import RolloutBuffer
from gymnasium import spaces
import numpy as np
from copy import deepcopy

class AuxiliaryBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    infos: List[List[Dict[str, Any]]]

class AuxiliaryBuffer(RolloutBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO, 
        with an added tracking of information dictionaries.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.
    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """
    def __init__(self,         
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1):
        super().__init__(buffer_size, observation_space, action_space, device, 
            gae_lambda=gae_lambda, gamma=gamma, n_envs=n_envs)
        
        lengths = [buffer_size] * n_envs
        self.info_bins = np.cumsum(np.array(lengths))
        self.reset()

    def reset(self) -> None:
        self.infos = [[None] * self.buffer_size for __ in range(self.n_envs)]
        super().reset()

    def add(self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
            infos: List[Dict[str, Any]], 
            ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        :param infos: the information dictionaries returned by the environment
        """
        for i, info in enumerate(infos):
            self.infos[i][self.pos] = deepcopy(info)
        super().add(obs, action, reward, episode_start, value, log_prob)

    

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> AuxiliaryBufferSamples:
        rollout_samples = super()._get_samples(batch_inds, env)
        infos = []
        env_indices = np.digitize(batch_inds, bins=self.info_bins, right=False)
        last_env_indices = np.clip(env_indices - 1, 0, self.info_bins.shape[0])
        step_indices = batch_inds - self.info_bins[last_env_indices]
        for ep_idx, step_idx in zip(env_indices, step_indices):
            infos.append(self.infos[ep_idx][step_idx:])
        return AuxiliaryBufferSamples(*rollout_samples, infos=infos)
