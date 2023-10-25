from stable_baselines3 import A2C
from typing import Any, Dict, Optional, Type, Union
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor, explained_variance
from gymnasium import spaces
from torch.nn import functional as F
import torch as th
import numpy as np
from auxiliary_a2c.buffers import AuxiliaryBuffer
from auxiliary_a2c.objectives import AuxiliaryObjective
from auxiliary_a2c.policies import AuxiliaryMlpPolicy, AuxiliaryCnnPolicy
    

class AuxiliaryA2C(A2C):
    """
    Advantage Actor Critic (A2C) with the ability to provide an Auxiliary Objective using a Callback
    Paper: https://arxiv.org/abs/1602.01783
    Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (https://github.com/hill-a/stable-baselines)
    Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752
    :param policy: The policy model to use (One of AuxiliaryMlpPolicy or AuxiliaryCnnPolicy)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param normalize_advantage: Whether to normalize or not the advantage
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param auxiliary_objective: The auxiliary objective, whose loss to add to training
    :param auxiliary_coef: Auxiliary function coefficient for the loss calculation 
    """
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "AuxiliaryMlpPolicy": AuxiliaryMlpPolicy,
        "AuxiliaryCnnPolicy": AuxiliaryCnnPolicy,
    }
    def __init__(
            self,
            env: Union[GymEnv, str],
            policy: Union[str, Type[AuxiliaryMlpPolicy]],
            learning_rate: Union[float, Schedule] = 7e-4,
            n_steps: int = 5,
            gamma: float = 0.99,
            gae_lambda: float = 1.0,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            rms_prop_eps: float = 1e-5,
            use_rms_prop: bool = True,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            normalize_advantage: bool = False,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            auxiliary_objective: AuxiliaryObjective = None,
            auxiliary_coef: float = 0.5,
            ):

        self.auxiliary_objective = auxiliary_objective
        self.auxiliay_coef = auxiliary_coef

        super().__init__(
            policy = policy,
            env = env,
            learning_rate= learning_rate,
            n_steps= n_steps,
            gamma = gamma,
            gae_lambda= gae_lambda,
            ent_coef = ent_coef,
            vf_coef = vf_coef,
            max_grad_norm = max_grad_norm,
            rms_prop_eps = rms_prop_eps,
            use_rms_prop = use_rms_prop,
            use_sde = use_sde,
            sde_sample_freq = sde_sample_freq,
            normalize_advantage = normalize_advantage,
            tensorboard_log = tensorboard_log,
            policy_kwargs = policy_kwargs,
            verbose = verbose,
            seed = seed,
            device = device,
            _init_setup_model = False
        )

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = AuxiliaryBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: AuxiliaryBuffer,
            n_rollout_steps: int,
        ) -> bool:
            """
            Collect experiences using the current policy and fill a ``AuxiliaryBuffer``.
            The term rollout here refers to the model-free notion and should not
            be used with the concept of rollout used in model-based RL or planning.
            :param env: The training environment
            :param callback: Callback that will be called at each step
                (and at the beginning and end of the rollout)
            :param rollout_buffer: Buffer to fill with rollouts
            :param n_rollout_steps: Number of experiences to collect per environment
            :return: True if function returned with at least `n_rollout_steps`
                collected, False if callback terminated rollout prematurely.
            """
            assert self._last_obs is not None, "No previous observation was provided"
            # Switch to eval mode (this affects batch norm / dropout)
            self.policy.set_training_mode(False)

            n_steps = 0
            rollout_buffer.reset()
            # Sample new weights for the state dependent exploration
            if self.use_sde:
                self.policy.reset_noise(env.num_envs)

            callback.on_rollout_start()

            while n_steps < n_rollout_steps:
                if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.policy.reset_noise(env.num_envs)

                with th.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    actions, values, log_probs = self.policy(obs_tensor)
                actions = actions.cpu().numpy()

                # Rescale and perform action
                clipped_actions = actions
                # Clip the actions to avoid out of bound error
                if isinstance(self.action_space, spaces.Box):
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

                new_obs, rewards, dones, infos = env.step(clipped_actions)

                self.num_timesteps += env.num_envs

                # Give access to local variables
                callback.update_locals(locals())
                if callback.on_step() is False:
                    return False

                self._update_info_buffer(infos)
                n_steps += 1

                if isinstance(self.action_space, spaces.Discrete):
                    # Reshape in case of discrete action
                    actions = actions.reshape(-1, 1)

                # Handle timeout by bootstraping with value function
                # see GitHub issue #633
                for idx, done in enumerate(dones):
                    if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                    ):
                        terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                        with th.no_grad():
                            terminal_value = self.policy.predict_values(terminal_obs)[0]
                        rewards[idx] += self.gamma * terminal_value

                rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs, infos)
                self._last_obs = new_obs
                self._last_episode_starts = dones

            with th.no_grad():
                # Compute value for the last timestep
                values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

            rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

            callback.on_rollout_end()

            return True

    def train(self) -> None:
            """
            Update policy using the currently gathered
            rollout buffer (one gradient step over whole data).
            """
            # Switch to train mode (this affects batch norm / dropout)
            self.policy.set_training_mode(True)

            # Update optimizer learning rate
            self._update_learning_rate(self.policy.optimizer)

            # This will only loop once (get all data in one go)
            for rollout_data in self.rollout_buffer.get(batch_size=None):

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = actions.long().flatten()

                values, log_prob, entropy, features = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                # Normalize advantage (not present in the original implementation)
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy gradient loss
                policy_loss = -(advantages * log_prob).mean()

                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values)

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                # Auxiliary loss
                if self.auxiliary_objective is not None:
                    auxiliary_loss = self.auxiliary_objective.calculate_loss(
                        obs = rollout_data.observations, features=features,
                        actions = rollout_data.actions, returns = rollout_data.returns,
                        infos = rollout_data.infos
                    )
                else:
                    auxiliary_loss = None
            

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                if auxiliary_loss is not None:
                    loss = loss + self.auxiliay_coef * auxiliary_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()

                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

            self._n_updates += 1
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/explained_variance", explained_var)
            self.logger.record("train/entropy_loss", entropy_loss.item())
            self.logger.record("train/policy_loss", policy_loss.item())
            self.logger.record("train/value_loss", value_loss.item())
            if auxiliary_loss is not None:
                self.logger.record("train/auxiliary_loss", auxiliary_loss.item())
            if hasattr(self.policy, "log_std"):
                self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
