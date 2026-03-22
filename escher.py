# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements the ESCHER algorithm.

See https://arxiv.org/abs/2206.04122.

ESCHER is an unbiased model-free method that does not require any importance
sampling. Emperically, the variance of the estimated regret of ESCHER is orders
of magnitude lower than DREAM and other baselines.
"""

import json
import logging
import os
import time
import typing

import numpy as np
import open_spiel
import torch
import torch.utils.tensorboard as _

from open_spiel.python.algorithms import exploitability as _
from open_spiel.python.pytorch import deep_cfr as _

import util

# pylint: disable=invalid-name


class Config:
    """A Config is an configuration for an Escher agent."""

    def __init__(self):
        """Initialize the configuration.

        These parameters are designed for Kuhn poker to achieve an
        exploitability around 0.05 within 100 iterations.
        """
        self.value_traversals = 512
        self.value_exploration = 0.1
        self.value_memory_capacity = int(1e6)
        self.value_net = [64]
        self.value_batch_size = 256
        self.value_batch_steps = 512
        self.value_learning_rate = 1e-3

        self.regret_traversals = 1024
        self.regret_memory_capacity = int(1e6)
        self.regret_net = [64]
        self.regret_batch_size = 256
        self.regret_batch_steps = 375
        self.regret_learning_rate = 1e-3

        self.avg_policy_memory_capacity = int(1e6)
        self.avg_policy_net = [64]
        self.avg_policy_batch_size = 256
        self.avg_policy_batch_steps = 2500
        self.avg_policy_learning_rate = 1e-3


class Agent:
    """An Agent is an ESCHER agent.

    See https://arxiv.org/abs/2206.04122.
    """

    def __init__(self, game, cfg):
        """Initialize the agent.

        Args:
                game: Openspiel game.
                cfg: (Config) configuration for the agent.
        """
        self.cfg = cfg
        self.t = 0

        # Get game state dimensions.
        state = game.new_initial_state()
        history_dim = _state_history(game.num_players(), state).shape[0]
        obs_dim = game.information_state_tensor_size()
        action_dim = game.num_distinct_actions()

        # Initialize average policy network.
        ReservoirBuffer = open_spiel.python.pytorch.deep_cfr.ReservoirBuffer
        infoset = np.zeros(obs_dim, dtype=float)
        policy = np.zeros(action_dim, dtype=float)
        scalar = np.array(0, dtype=float)
        self.avg_policy_buffer = ReservoirBuffer.init(
                cfg.avg_policy_memory_capacity,
                Behaviour(state=infoset, policy=policy, t=scalar),
        )
        MLP = open_spiel.python.pytorch.deep_cfr.MLP
        self.avg_policy_net = MLP(obs_dim, cfg.avg_policy_net, action_dim)

        # Initialize regret network.
        sr = StateRegret(state=infoset, regret=policy, mask=policy, t=scalar)
        self.regret_buffers = [
                ReservoirBuffer.init(cfg.regret_memory_capacity, sr)
                for _ in range(game.num_players())
        ]
        self.regret_nets = [
                MLP(obs_dim, cfg.regret_net, action_dim)
                for _ in range(game.num_players())
        ]

        # Initialize value network.
        history = np.zeros(history_dim, dtype=float)
        sav = StateActionValue(state=history, action=scalar, value=scalar)
        self.value_buffers = [
                ReservoirBuffer.init(cfg.value_memory_capacity, sav)
                for _ in range(game.num_players())
        ]
        self.value_nets = [
                MLP(history_dim, cfg.value_net, 1) for _ in range(game.num_players())
        ]

        self.num_touched = 0
        self.avg_policy_t = 0
        self.regret_t = 0
        self.value_t = 0

        self.device = None

    def action_probabilities(self, state):
        """action_probabilities returns the action probabilities of state.

        Args:
                state: (pyspiel.State) The state to compute probabilities for.

        Returns:
                Action probabilities of state.
        """
        obs = np.array(state.information_state_tensor(), dtype=float)
        mask_np = np.array(state.legal_actions_mask(), dtype=int)

        with torch.no_grad():
            x = torch.from_numpy(obs).to(torch.float32).to(self.device)
            mask = torch.from_numpy(mask_np).to(self.device)

            logits = self.avg_policy_net(x)
            probs = torch.nn.functional.softmax(logits, dim=0)

            probs = torch.mul(probs, mask)
            probs = probs / torch.sum(probs)

        return probs.cpu().numpy()

    def _set_device(self, device):
        self.device = device
        self.avg_policy_net.to(device)
        for i in range(len(self.regret_nets)):
            self.regret_nets[i].to(device)
            self.value_nets[i].to(device)


def _train_avg_policy(cfg, agent):
    """Trains the average policy network."""
    num_epoch = 8
    epoch_steps = int(np.ceil(agent.cfg.avg_policy_batch_steps / num_epoch))

    buf = agent.avg_policy_buffer
    dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(buf.experience.state).to(cfg.device),
            torch.from_numpy(buf.experience.policy).to(cfg.device),
            torch.from_numpy(buf.experience.t).to(cfg.device),
    )
    optimizer = torch.optim.Adam(
            agent.avg_policy_net.parameters(), lr=agent.cfg.avg_policy_learning_rate
    )

    for _ in range(num_epoch):
        agent.avg_policy_t += 1
        metrics = {}

        agent.avg_policy_net.train()
        for _ in range(epoch_steps):
            indices = np.random.choice(
                    len(buf), size=(agent.cfg.avg_policy_batch_size,), replace=False
            )
            batch = Behaviour(
                    state=dataset.tensors[0][indices],
                    policy=dataset.tensors[1][indices],
                    t=dataset.tensors[2][indices],
            )

            loss = _get_avg_policy_loss(agent, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = util.update_metric(metrics, "avg_policy/loss", loss)

        for k, v in metrics.items():
            cfg.summary_writer.add_scalar(k, v.compute(), agent.avg_policy_t)


def _gather_regret_data(game, agent, player):
    """Gathers regret data for training."""
    start_time = time.time()
    for _ in range(agent.cfg.regret_traversals):
        state = game.new_initial_state()
        agent.num_touched += 1
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                a = np.random.choice(actions, p=probs)
                state.apply_action(a)
                continue

            # Get policy.
            current_player = state.current_player()
            obs = np.array(state.information_state_tensor(), dtype=float)
            mask = np.array(state.legal_actions_mask(), dtype=int)
            policy = _match_regret(agent.regret_nets[current_player], obs, mask, agent.device)

            # Add data to buffer.
            if current_player == player:
                regret = _get_regret(agent, state, policy, game.num_players())
                sr = StateRegret(state=obs, regret=regret, mask=mask, t=agent.t)
                agent.regret_buffers[player].append(sr)
            else:
                behaviour = Behaviour(state=obs, policy=policy, t=agent.t)
                agent.avg_policy_buffer.append(behaviour)

            # Update state with policy.
            if current_player == player:
                sample_policy = mask / np.sum(mask)
            else:
                sample_policy = policy
            action = np.random.choice(range(len(sample_policy)), p=sample_policy)
            state = state.child(action)
            agent.num_touched += 1
    logging.info("gather regret %d player %d duration %d secs", agent.t, player, int(time.time()-start_time))


def _train_regret(cfg, agent):
    """Trains the regret network."""
    for player in range(cfg.game.num_players()):
        _train_value(cfg, agent, player)
        _gather_regret_data(cfg.game, agent, player)

        start_time = time.time()
        num_epoch = 8
        epoch_steps = int(np.ceil(agent.cfg.regret_batch_steps / num_epoch))
        buf = agent.regret_buffers[player]
        dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(buf.experience.state).to(cfg.device),
                torch.from_numpy(buf.experience.regret).to(cfg.device),
                torch.from_numpy(buf.experience.mask).to(cfg.device),
                torch.from_numpy(buf.experience.t).to(cfg.device),
        )
        regret_net = agent.regret_nets[player]
        regret_net.reset()
        optimizer = torch.optim.Adam(
                regret_net.parameters(), lr=agent.cfg.regret_learning_rate
        )

        for _ in range(num_epoch):
            agent.regret_t += 1
            metrics = {}

            for _ in range(epoch_steps):
                indices = np.random.choice(
                        len(buf), size=(agent.cfg.regret_batch_size,), replace=False
                )
                batch = StateRegret(
                        state=dataset.tensors[0][indices],
                        regret=dataset.tensors[1][indices],
                        mask=dataset.tensors[2][indices],
                        t=dataset.tensors[3][indices],
                )

                loss = _get_regret_loss(agent, player, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                k = "regret/{}/loss".format(player)
                metrics = util.update_metric(metrics, k, loss)

            for k, v in metrics.items():
                cfg.summary_writer.add_scalar(k, v.compute(), agent.regret_t)
        logging.info("train regret %d player %d duration %d secs", agent.t, player, int(time.time()-start_time))


def _gather_value_data(game, agent, player):
    """Gathers value data for training."""
    start_time = time.time()
    value_buffer = agent.value_buffers[player]
    value_buffer.clear()
    for _ in range(agent.cfg.value_traversals):
        state = game.new_initial_state()
        agent.num_touched += 1
        transitions = []
        while True:
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                a = np.random.choice(actions, p=probs)
                state.apply_action(a)
                continue

            action, importance = -1, 1
            if not state.is_terminal():
                # Get policy.
                obs = np.array(state.information_state_tensor(), dtype=float)
                mask = np.array(state.legal_actions_mask(), dtype=int)
                regret_net = agent.regret_nets[state.current_player()]
                policy = _match_regret(regret_net, obs, mask, agent.device)

                # Sample action.
                epsilon = agent.cfg.value_exploration
                uniform = mask / np.sum(mask)
                sample_policy = epsilon * uniform + (1 - epsilon) * policy
                action = np.random.choice(range(len(sample_policy)), p=sample_policy)
                action = _win_action(state, action)
                importance = policy[action] / sample_policy[action]

            # Add transition.
            history = _state_history(game.num_players(), state)
            returns = np.array(state.returns(), dtype=float)
            tn = Transition(
                    history=history, importance=importance, action=action, returns=returns
            )
            transitions.append(tn)

            if state.is_terminal():
                break
            state = state.child(action)
            agent.num_touched += 1

        value = np.zeros(transitions[0].returns.shape, dtype=float)
        for i in range(len(transitions) - 1, -1, -1):
            tn = transitions[i]

            value = tn.importance * (tn.returns + value)
            value_buffer.append(
                    StateActionValue(
                            state=tn.history, action=tn.action, value=value[player]
                    )
            )
    logging.info("gather value %d player %d duration %d secs", agent.t, player, int(time.time()-start_time))


def _train_value(cfg, agent, player):
    """Trains the value network."""
    _gather_value_data(cfg.game, agent, player)

    start_time = time.time()
    num_epoch = 8
    epoch_steps = int(np.ceil(agent.cfg.value_batch_steps / num_epoch))
    buf = agent.value_buffers[player]
    dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(buf.experience.state).to(cfg.device),
            torch.from_numpy(buf.experience.action).to(cfg.device),
            torch.from_numpy(buf.experience.value).to(cfg.device),
    )
    value_net = agent.value_nets[player]
    value_net.reset()
    optimizer = torch.optim.Adam(
            value_net.parameters(), lr=agent.cfg.value_learning_rate
    )

    for _ in range(num_epoch):
        agent.value_t += 1
        metrics = {}

        agent.value_nets[player].train()
        for _ in range(epoch_steps):
            indices = np.random.choice(
                    len(buf), size=(agent.cfg.value_batch_size,), replace=False
            )
            batch = StateActionValue(
                    state=dataset.tensors[0][indices],
                    action=dataset.tensors[1][indices],
                    value=dataset.tensors[2][indices],
            )

            loss = _get_value_loss(agent, player, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            k = "value/{}/loss".format(player)
            metrics = util.update_metric(metrics, k, loss)

        for k, v in metrics.items():
            cfg.summary_writer.add_scalar(k, v.compute(), agent.value_t)
    logging.info("train value %d player %d duration %d secs", agent.t, player, int(time.time()-start_time))


def _get_avg_policy_loss(agent, batch):
    """Returns the loss for the average policy network."""
    x = batch.state.to(torch.float32)
    y_policy = batch.policy

    logits = agent.avg_policy_net(x)

    loss = torch.nn.functional.cross_entropy(logits, y_policy)

    # Linear CFR.
    weight = batch.t / agent.t
    loss = torch.mul(loss, weight)

    return torch.mean(loss)


def _get_regret_loss(agent, player, batch):
    """Returns the loss for the regret network."""
    x = batch.state.to(torch.float32)
    mask = batch.mask
    y_regret = batch.regret

    regret = agent.regret_nets[player](x)

    loss = torch.pow(regret - y_regret, 2)

    # Linear CFR.
    weight = batch.t / agent.t
    weight = weight.unsqueeze(-1).expand(-1, loss.shape[-1])
    loss = torch.mul(loss, weight)

    loss = torch.sum(torch.mul(loss, mask)) / torch.sum(mask)
    return loss


def _get_value_loss(agent, player, batch):
    x = batch.state.to(torch.float32)
    y_value = batch.value

    value = agent.value_nets[player](x)
    value = torch.squeeze(value, dim=[1])

    loss = torch.pow(value - y_value, 2)
    return torch.mean(loss)


def _match_regret(net, obs, mask_np, device):
    """Returns the policy after applying regret matching."""
    with torch.no_grad():
        x = torch.from_numpy(obs).to(torch.float32).to(device)
        regrets = net(x)
        raw_regrets = regrets.cpu().numpy()

    regrets = np.clip(raw_regrets, a_min=0, a_max=None)
    regrets = regrets * mask_np
    summed = np.sum(regrets)
    if summed > 1e-6:
        return regrets / summed

    # Just use the best regret, if regrets cannot be normalized.
    max_id, max_regret = -1, float("-inf")
    for i, m in enumerate(mask_np):
        if m == 1 and raw_regrets[i] > max_regret:
            max_id, max_regret = i, raw_regrets[i]
    policy = np.zeros(regrets.shape, dtype=regrets.dtype)
    policy[max_id] = 1
    return policy


def _get_regret(agent, state, policy, num_players):
    """Returns the regret for the current state."""
    player = state.current_player()

    mask = state.legal_actions_mask()
    children_values = np.zeros(len(mask), dtype=float)
    for a, m in enumerate(mask):
        if m == 1:
            child = state.child(a)

            with torch.no_grad():
                history = _state_history(num_players, child)
                x = torch.from_numpy(history).to(torch.float32).to(agent.device)
                children_values[a] = agent.value_nets[player](x)

    value = np.sum(policy * children_values)
    regret = children_values - value
    return regret


class TrainConfig:
    """A TrainConfig is a configuration for the training of an Escher agent."""

    def __init__(self, game):
        """Initialize the training configuration.

        Args:
                game: Openspiel game.
        """
        self.game = game
        self.device_name = "cpu"
        self.iterations = 999999
        self.evaluation_interval = 1
        self.nashconv = False
        self.games_vs_random = 1000

        self.run_dir = ""

    def setup(self):
        self.summary_writer = torch.utils.tensorboard.SummaryWriter(self.run_dir)
        self.device = torch.device(self.device_name)


def train(cfg, agent):
    """Trains an Escher agent.

    Args:
            cfg: (TrainConfig) The configuration for the training.
            agent: (Agent) The Escher agent to be trained.
    """
    cp_dir = os.path.join(cfg.run_dir, "checkpoint")
    _load_checkpoint(agent, cp_dir)
    agent._set_device(cfg.device)
    _save_agent_config(cfg.run_dir, agent)

    metrics = {}
    for _ in range(cfg.iterations):
        agent.t += 1
        _train_regret(cfg, agent)

        if agent.t % 2 == 0 and agent.t % cfg.evaluation_interval != 0:
            _save_checkpoint(cp_dir, agent)
        if agent.t % cfg.evaluation_interval == 0:
            _train_avg_policy(cfg, agent)
            _save_checkpoint(cp_dir, agent)
            cfg.summary_writer.add_scalar("states_touched", agent.num_touched, agent.t)
            if cfg.nashconv:
                conv = _calc_nashconv(cfg.game, agent)
                logging.info(
                        "iteration %d states %d nashconv %f",
                        agent.t,
                        agent.num_touched,
                        conv,
                )
                cfg.summary_writer.add_scalar("nashconv", conv, agent.t)
            reward = _play_against_random(cfg.game, agent, cfg.games_vs_random)
            logging.info(
                    "iteration %d states %d reward_vs_random %f",
                    agent.t,
                    agent.num_touched,
                    reward,
            )
            cfg.summary_writer.add_scalar("reward_vs_random", reward, agent.t)
            cfg.summary_writer.flush()


def _play_once_against_random(game, agent):
    """Plays one game against a random policy and returns the reward."""
    reward = 0
    for player in range(game.num_players()):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                a = np.random.choice(outcomes, p=probs)
                state.apply_action(a)
                continue

            if state.current_player() == player:
                policy = agent.action_probabilities(state)
            else:
                mask = np.array(state.legal_actions_mask(), dtype=int)
                policy = mask / np.sum(mask)
            action = np.random.choice(range(len(policy)), p=policy)
            state.apply_action(action)

        reward += state.returns()[player]

    return reward / game.num_players()


def _play_against_random(game, agent, n):
    reward = 0
    for _ in range(n):
        reward += _play_once_against_random(game, agent)
    return reward / n


def _calc_nashconv(game, agent):
    """Calculates the NashConv of the current policy."""
    def _action_probabilities(state):
        probs = agent.action_probabilities(state)

        prob_dict = {}
        for a, m in enumerate(state.legal_actions_mask()):
            if m == 1:
                prob_dict[a] = probs[a]
        return prob_dict

    policy = open_spiel.python.policy.tabular_policy_from_callable(
            game, _action_probabilities
    )
    conv = open_spiel.python.algorithms.exploitability.nash_conv(game, policy)
    return conv


class Transition(typing.NamedTuple):
    history: np.ndarray
    importance: np.ndarray
    action: np.ndarray
    returns: np.ndarray


class StateActionValue(typing.NamedTuple):
    state: np.ndarray
    action: np.ndarray
    value: np.ndarray


class StateRegret(typing.NamedTuple):
    state: np.ndarray
    regret: np.ndarray
    mask: np.ndarray
    t: np.ndarray


class Behaviour(typing.NamedTuple):
    state: np.ndarray
    policy: np.ndarray
    t: np.ndarray


def _state_history(num_players, state):
    history = []
    for p in range(num_players):
        history += state.information_state_tensor(p)
    return np.array(history, dtype=float)


def _save_agent_config(run_dir, agent):
    jstr = json.dumps(agent.cfg.__dict__)
    cfg_dir = os.path.join(run_dir, "config")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, "config_{:06d}.json".format(agent.t))
    with open(cfg_path, "w") as f:
        f.write(jstr)


def _save_checkpoint(cp_dir, agent):
    import gc
    gc.collect()

    cp = {}
    cp["t"] = agent.t
    cp["avg_policy_net"] = agent.avg_policy_net.state_dict()
    for p, _ in enumerate(agent.regret_nets):
        cp["regret_net_{}".format(p)] = agent.regret_nets[p].state_dict()
        cp["value_net_{}".format(p)] = agent.value_nets[p].state_dict()
    cp["num_touched"] = agent.num_touched
    cp["avg_policy_t"] = agent.avg_policy_t
    cp["regret_t"] = agent.regret_t
    cp["value_t"] = agent.value_t

    # Buffers.
    cp["avg_policy_buffer"] = agent.avg_policy_buffer.__dict__
    for p, _ in enumerate(agent.regret_nets):
        cp["regret_buffer_{}".format(p)] = agent.regret_buffers[p].__dict__
        # cp["value_buffer_{}".format(p)] = agent.value_buffers[p].__dict__

    os.makedirs(cp_dir, exist_ok=True)
    cp_path = util.get_checkpoint_path(cp_dir, agent.t)
    torch.save(cp, cp_path)
    util.delete_old_checkpoints(cp_dir)


def _load_checkpoint(agent, cp_dir):
    torch.serialization.add_safe_globals([StateActionValue, StateRegret, Behaviour])

    cp, cp_path = util.load_checkpoint(cp_dir)
    if not cp:
        logging.info("no checkpoint")
        return

    agent.t = cp["t"]
    agent.avg_policy_net.load_state_dict(cp["avg_policy_net"])
    for p, _ in enumerate(agent.regret_nets):
        agent.regret_nets[p].load_state_dict(cp["regret_net_{}".format(p)])
        agent.value_nets[p].load_state_dict(cp["value_net_{}".format(p)])
    agent.num_touched = cp["num_touched"]
    agent.avg_policy_t = cp["avg_policy_t"]
    agent.regret_t = cp["regret_t"]
    agent.value_t = cp["value_t"]

    # Buffers.
    agent.avg_policy_buffer.__dict__.update(cp["avg_policy_buffer"])
    for p, _ in enumerate(agent.regret_nets):
        agent.regret_buffers[p].__dict__.update(cp["regret_buffer_{}".format(p)])
        # agent.value_buffers[p].__dict__.update(cp["value_buffer_{}".format(p)])

    logging.info("loaded checkpoint %s", cp_path)


def _win_action(state, action):
    player = state.current_player()
    mask = state.legal_actions_mask()
    for a, m in enumerate(mask):
        if m == 0:
            continue
        child = state.child(a)
        reward = child.returns()[player]
        if reward == 1:
            return a
    return action
