# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the ESCHER agent."""

import os
import tempfile

from absl import app
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import pyspiel
import torch

import escher


class EscherTest(parameterized.TestCase):

    @parameterized.parameters("kuhn_poker", "leduc_poker")
    def test_escher_runs(self, game_name):
        game = pyspiel.load_game(game_name)

        cfg = escher.Config()
        cfg.trunk = [2]
        cfg.value_traversals = 2
        cfg.value_net = []
        cfg.value_batch_size = 2
        cfg.value_batch_steps = 2
        cfg.regret_traversals = 2
        cfg.regret_net = []
        cfg.regret_batch_size = 2
        cfg.regret_batch_steps = 2
        cfg.avg_policy_net = []
        cfg.avg_policy_batch_size = 2
        cfg.avg_policy_batch_steps = 2
        agent = escher.Agent(game, cfg)

        with tempfile.TemporaryDirectory() as run_dir:
            train_cfg = escher.TrainConfig(game)
            train_cfg.iterations = 2
            train_cfg.nashconv = True
            train_cfg.games_vs_random = 1
            train_cfg.run_dir = run_dir
            train_cfg.setup()
            escher.train(train_cfg, agent)

    def test_checkpoint(self):
        # Create game and agent.
        game = pyspiel.load_game("kuhn_poker")
        cfg = escher.Config()
        agent_saved = escher.Agent(game, cfg)

        # Append data to buffer.
        obs_dim = game.information_state_tensor_size()
        action_dim = game.num_distinct_actions()
        infoset = np.zeros(obs_dim, dtype=float)
        policy = np.zeros(action_dim, dtype=float)
        infoset[0] = 3.14159
        beh = escher.Behaviour(state=infoset, policy=policy, t=321)
        agent_saved.avg_policy_buffer.append(beh)

        with tempfile.TemporaryDirectory() as run_dir:
            # Save agent.
            cp_dir = os.path.join(run_dir, "checkpoint")
            escher._save_checkpoint(cp_dir, agent_saved)

            # Load a 2nd agent and check that it is identical to the 1st.
            agent = escher.Agent(game, cfg)
            escher._load_checkpoint(agent, cp_dir, True)
            beh2 = agent.avg_policy_buffer[0]
            np.testing.assert_array_equal(beh2.state, beh.state)
            np.testing.assert_array_equal(beh2.t, beh.t)

    def test_weights_shared(self):
        game = pyspiel.load_game("kuhn_poker")
        cfg = escher.Config()
        cfg.trunk = [2]
        cfg.value_traversals = 2
        cfg.value_net = []
        cfg.value_batch_size = 2
        cfg.value_batch_steps = 2
        cfg.regret_traversals = 2
        cfg.regret_net = []
        cfg.regret_batch_size = 2
        cfg.regret_batch_steps = 2
        cfg.avg_policy_net = []
        cfg.avg_policy_batch_size = 2
        cfg.avg_policy_batch_steps = 2
        agent_saved = escher.Agent(game, cfg)
        with tempfile.TemporaryDirectory() as run_dir:
            # Save agent.
            cp_dir = os.path.join(run_dir, "checkpoint")
            escher._save_checkpoint(cp_dir, agent_saved)

            # Load agent.
            agent = escher.Agent(game, cfg)
            escher._load_checkpoint(agent, cp_dir, True)

            # Train agent.
            train_cfg = escher.TrainConfig(game)
            train_cfg.iterations = 2
            train_cfg.games_vs_random = 1
            train_cfg.run_dir = run_dir
            train_cfg.setup()
            escher.train(train_cfg, agent)

        # Check weights are shared after loading and training.
        # First see that value_nets weights are different from target.
        target = 3.14159
        weight0 = list(agent.value_nets[0].trunk.model.modules())[-2].weight
        weight1 = list(agent.value_nets[1].trunk.model.modules())[-2].weight
        self.assertNotEqual(weight0[0, 0], target)
        self.assertNotEqual(weight1[0, 0], target)
        # Modify trunk.
        with torch.no_grad():
            trunk_w = list(agent.value_trunk.model.modules())[-2].weight
            trunk_w[0, 0] = target
        # Check that value_nets are modified, since they share the same trunk.
        self.assertEqual(weight0[0, 0], target)
        self.assertEqual(weight1[0, 0], target)


def main(_):
    absltest.main()


if __name__ == "__main__":
    app.run(main)
