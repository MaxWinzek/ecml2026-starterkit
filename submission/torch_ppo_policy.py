from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Sequence

import numpy as np
import torch

from flatland.envs.RailEnvPolicy import RailEnvPolicy
from flatland.envs.rail_env_action import RailEnvActions

from .torch_solution.DeadlockChecker import DeadlockChecker
from .torch_solution.GreedyChecker import GreedyChecker
from .torch_solution.PPOController import PPOController
from .torch_solution.SimpleObservation import SimpleObservation
from .torch_solution.env.compat import RailAgentStatus


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "checkpoints" / "run18" / "final_controller.torch"

RUN18_MODEL_CONFIG = SimpleNamespace(
    state_sz=203,
    action_sz=3,
    neighbours_depth=3,
    actor_layers_sz=[256],
    critic_layers_sz=[],
    gamma=0.99,
    entropy_coeff=0.01,
)


@dataclass(frozen=True)
class TorchPolicyObservation:
    policy_observation: np.ndarray | None
    internal_observation: np.ndarray
    encountered: tuple[int, ...]
    available_actions: tuple[int, ...]
    is_greedy: bool
    is_ready_to_depart: bool
    is_waiting: bool
    is_done: bool


class DeterministicLaunchCap:
    def __init__(
        self,
        initial_fraction: float = 0.4,
        min_active: int = 4,
        cap_increment: int = 1,
        ramp_interval: int = 40,
        min_agents_to_limit: int = 12,
        shuffle: bool = True,
    ):
        self.initial_fraction = initial_fraction
        self.min_active = min_active
        self.cap_increment = cap_increment
        self.ramp_interval = ramp_interval
        self.min_agents_to_limit = min_agents_to_limit
        self.shuffle = shuffle

    def reset(self, env):
        self.env = env
        self.ready_to_depart = [0] * len(self.env.agents)
        self.active_agents = set()
        self.timer = 0
        self.cur_threshold = float(self._allowed_cap())
        self.launch_pos = 0
        if self.shuffle:
            self.launch_order = np.random.permutation(len(self.env.agents)).tolist()
        else:
            self.launch_order = list(range(len(self.env.agents)))

    def update(self):
        self.timer += 1
        self.update_finished()
        allowed_cap = self._allowed_cap()
        self.cur_threshold = float(allowed_cap)

        while len(self.active_agents) < allowed_cap and self.launch_pos < len(self.launch_order):
            handle = self.launch_order[self.launch_pos]
            self.launch_pos += 1
            if self.ready_to_depart[handle] == 0:
                self._start_agent(handle)

    def update_finished(self):
        for handle in list(self.active_agents):
            if (
                self.env.agents[handle].state in (RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED)
                or self.env.obs_builder.deadlock_checker.is_deadlocked(handle)
            ) and self.ready_to_depart[handle] == 1:
                self._finish_agent(handle)

    def is_ready(self, handle: int) -> bool:
        return self.ready_to_depart[handle] != 0

    def update_net_params(self, net_params):
        return None

    def get_net_params(self, device=None):
        return {}

    def load_judge(self, path):
        return None

    def save_judge(self, dirpath, name="judge.torch"):
        return None

    def get_rollout(self):
        return torch.empty(0), torch.empty((0, 0))

    def _allowed_cap(self) -> int:
        n_agents = len(self.env.agents)
        if n_agents < self.min_agents_to_limit:
            return n_agents

        base_cap = max(self.min_active, int(np.ceil(self.initial_fraction * n_agents)))
        if self.ramp_interval > 0:
            base_cap += (self.env._elapsed_steps // self.ramp_interval) * self.cap_increment
        return min(n_agents, base_cap)

    def _start_agent(self, handle: int):
        self.ready_to_depart[handle] = 1
        self.active_agents.add(handle)

    def _finish_agent(self, handle: int):
        self.ready_to_depart[handle] = 2
        self.active_agents.remove(handle)


class TorchPPOObservationBuilder(SimpleObservation):
    def __init__(self):
        super().__init__(
            max_depth=3,
            neighbours_depth=3,
            timetable=DeterministicLaunchCap(),
            deadlock_checker=DeadlockChecker(),
            greedy_checker=GreedyChecker(),
            parallel=False,
            eval=False,
        )

    def get(self, handle: int) -> TorchPolicyObservation:
        policy_observation = None
        if self._get_checks(handle):
            internal_observation = np.asarray(self._get_internal(handle), dtype=np.float32)
            policy_observation = internal_observation
        else:
            internal_observation = np.asarray(self._get_internal(handle), dtype=np.float32)

        agent = self.env.agents[handle]
        return TorchPolicyObservation(
            policy_observation=policy_observation,
            internal_observation=internal_observation,
            encountered=tuple(self.encountered.get(handle, ())),
            available_actions=self._get_available_actions(handle),
            is_greedy=self.greedy_checker.greedy_position(handle),
            is_ready_to_depart=agent.state == RailAgentStatus.READY_TO_DEPART,
            is_waiting=agent.state == RailAgentStatus.WAITING,
            is_done=agent.state in (RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED),
        )

    def _get_available_actions(self, handle: int) -> tuple[int, ...]:
        agent = self.env.agents[handle]
        position = agent.position
        direction = agent.direction

        if agent.state in (RailAgentStatus.READY_TO_DEPART, RailAgentStatus.WAITING):
            position = agent.initial_position
            direction = agent.initial_direction

        if position is None or direction is None:
            return ()

        transitions = self.env.rail.get_transitions((position, direction))
        available_actions = []
        for delta in range(-1, 2):
            new_direction = (direction + delta + 4) % 4
            if transitions[new_direction]:
                available_actions.append(delta + 2)
        return tuple(available_actions)


class TorchPPOPolicy(RailEnvPolicy):
    def __init__(self):
        super().__init__()
        self.device = torch.device(os.environ.get("TORCH_PPO_DEVICE", "cpu"))
        self.model_path = Path(os.environ.get("TORCH_PPO_MODEL_PATH", str(DEFAULT_MODEL_PATH)))

        self._ppo = PPOController(RUN18_MODEL_CONFIG, self.device)
        self._load_model(self.model_path)

        self._ppo.actor_net.eval()
        self._ppo.critic_net.eval()
        self._ppo.target_actor.eval()

    def act_many(self, handles, observations, **kwargs) -> Dict[int, RailEnvActions]:
        observations_by_handle = self._observations_by_handle(handles, observations)
        valid_handles = []
        state_dict = {}
        neighbours = {}

        for handle in handles:
            obs = observations_by_handle[handle]
            state_dict[handle] = torch.tensor(obs.internal_observation, dtype=torch.float32, device=self.device)
            if obs.policy_observation is not None:
                valid_handles.append(handle)
                neighbours[handle] = list(obs.encountered)

        ppo_actions = self._ppo.fast_select_actions(valid_handles, state_dict, neighbours, train=False)
        action_dict = {}

        for handle in handles:
            obs = observations_by_handle[handle]
            if obs.policy_observation is None:
                action_dict[handle] = self._fallback_action(obs)
                continue

            ppo_action = ppo_actions.get(handle, 0)
            if isinstance(ppo_action, torch.Tensor):
                ppo_action = ppo_action.item()
            action_dict[handle] = self._transform_action(obs, int(ppo_action))

        return action_dict

    def act(self, observation: TorchPolicyObservation, **kwargs) -> RailEnvActions:
        if observation.policy_observation is None:
            return self._fallback_action(observation)

        neighbours_count = len(observation.encountered) or (2 ** (RUN18_MODEL_CONFIG.neighbours_depth + 1) - 2)
        neighbours_state = torch.zeros(
            (1, neighbours_count, RUN18_MODEL_CONFIG.state_sz),
            dtype=torch.float32,
            device=self.device,
        )
        state = torch.tensor(observation.policy_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self._ppo._make_logits(state, neighbours_state)
            ppo_action = torch.argmax(logits, dim=1).item()
        return self._transform_action(observation, ppo_action)

    def _load_model(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Missing PPO checkpoint at {path}")

        model = torch.load(path, map_location=self.device, weights_only=False)
        if "actor" in model and "critic" in model:
            self._ppo.actor_net.load_state_dict(model["actor"])
            self._ppo.critic_net.load_state_dict(model["critic"])
        elif "actor_state" in model and "critic_state" in model:
            self._ppo.actor_net.load_state_dict(model["actor_state"])
            self._ppo.critic_net.load_state_dict(model["critic_state"])
        else:
            raise ValueError(f"Unsupported PPO checkpoint format in {path}")
        self._ppo.hard_update()

    def _fallback_action(self, observation: TorchPolicyObservation) -> RailEnvActions:
        if observation.is_done:
            return RailEnvActions.DO_NOTHING
        if observation.is_greedy:
            return self._transform_action(observation, 0)
        return RailEnvActions.DO_NOTHING

    def _transform_action(self, observation: TorchPolicyObservation, ppo_action: int) -> RailEnvActions:
        if observation.is_done:
            return RailEnvActions.DO_NOTHING

        if ppo_action == 2:
            if observation.is_ready_to_depart or observation.is_waiting:
                return RailEnvActions.DO_NOTHING
            return RailEnvActions.STOP_MOVING

        if not observation.available_actions:
            return RailEnvActions.MOVE_FORWARD

        if len(observation.available_actions) == 1 and ppo_action == 1:
            return RailEnvActions(observation.available_actions[0])

        action_index = min(ppo_action, len(observation.available_actions) - 1)
        return RailEnvActions(observation.available_actions[action_index])

    @staticmethod
    def _observations_by_handle(handles: Sequence[int], observations: Sequence[TorchPolicyObservation]) -> Dict[int, TorchPolicyObservation]:
        if not handles:
            return {}
        if observations and len(observations) > max(handles):
            return {handle: observations[handle] for handle in handles}
        return {handle: observation for handle, observation in zip(handles, observations)}
