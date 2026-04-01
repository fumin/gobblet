import argparse
import datetime
import logging

import pyspiel

import escher


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    [lg.removeHandler(h) for h in lg.handlers]
    lg.addHandler(logging.StreamHandler())
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest="run_dir", default="runs/test")
    args = parser.parse_args()

    game = pyspiel.load_game("kuhn_poker")

    cfg = escher.Config()
    agent = escher.Agent(game, cfg)

    train_cfg = escher.TrainConfig(game)
    train_cfg.device_name = "cpu"
    train_cfg.evaluation_interval = 20
    train_cfg.nashconv = True
    train_cfg.run_dir = args.run_dir
    train_cfg.setup()
    logging.info("run_dir \"%s\"", train_cfg.run_dir)

    # import numpy as np
    # import os
    # state = game.new_initial_state()
    # history_dim = escher._state_history(game.num_players(), state).shape[0]
    # obs_dim = game.information_state_tensor_size()
    # action_dim = game.num_distinct_actions()
    # infoset = np.zeros(obs_dim, dtype=float)
    # policy = np.zeros(action_dim, dtype=float)
    # infoset[0] = 3.14159
    # beh = escher.Behaviour(state=infoset, policy=policy, t=321)
    # agent.avg_policy_buffer.append(beh)
    # cp_dir = os.path.join(train_cfg.run_dir, "checkpoint")
    # escher._save_checkpoint(cp_dir, agent)
    # agent2 = escher.Agent(game, cfg)
    # logging.info("%s", agent2.avg_policy_buffer[0])
    # escher._load_checkpoint(agent2, cp_dir, True)
    # logging.info("loaded %s", agent2.avg_policy_buffer[0])
    # return

    escher.train(train_cfg, agent)


if __name__ == "__main__":
    main()
