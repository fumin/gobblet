import argparse
import datetime
import logging

import pyspiel

import escher
import gobblet


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

    game = pyspiel.load_game("gobblet", {"egocentric_obs_tensor": True})

    cfg = escher.Config()
    cfg.trunk = [256, 256, 256]
    cfg.value_traversals = 4096
    cfg.value_exploration = 0.05
    cfg.value_net = []
    cfg.value_batch_size = 2048
    cfg.value_batch_steps = 8192
    cfg.value_learning_rate = 1e-3
    cfg.regret_traversals = 2048
    cfg.regret_memory_capacity = int(1e6)
    cfg.regret_net = []
    cfg.regret_batch_size = 2048
    cfg.regret_batch_steps = 8192
    cfg.regret_learning_rate = 1e-3
    cfg.avg_policy_memory_capacity = int(1e6)
    cfg.avg_policy_net = []
    cfg.avg_policy_batch_size = 2048
    cfg.avg_policy_batch_steps = 16384
    cfg.avg_policy_learning_rate = 1e-3
    agent = escher.Agent(game, cfg)

    train_cfg = escher.TrainConfig(game)
    train_cfg.device_name = "cuda:0"
    train_cfg.evaluation_interval = 5
    train_cfg.games_vs_random = 2000
    train_cfg.verbose = True
    train_cfg.run_dir = args.run_dir
    train_cfg.setup()
    logging.info("run_dir \"%s\"", train_cfg.run_dir)

    escher.train(train_cfg, agent)


if __name__ == "__main__":
    main()
