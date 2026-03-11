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

    game = pyspiel.load_game("gobblet")

    cfg = escher.Config()
    cfg.value_traversals = 4096
    cfg.value_exploration = 0.05
    cfg.value_net = [1024, 512]
    cfg.value_batch_size = 1024
    cfg.value_batch_steps = 4096
    cfg.regret_traversals = 1024
    cfg.regret_memory_capacity = int(2e5)
    cfg.regret_net = [1024, 512]
    cfg.regret_batch_size = 1024
    cfg.regret_batch_steps = 4096
    cfg.avg_policy_memory_capacity = int(5e5)
    cfg.avg_policy_net = [1024, 512]
    cfg.avg_policy_batch_size = 1024
    cfg.avg_policy_batch_steps = 8192
    agent = escher.Agent(game, cfg)

    train_cfg = escher.TrainConfig(game)
    train_cfg.device_name = "cuda:0"
    train_cfg.evaluation_interval = 5
    train_cfg.games_vs_random = 2000
    train_cfg.run_dir = args.run_dir
    train_cfg.setup()
    logging.info("run_dir \"%s\"", train_cfg.run_dir)

    escher.train(train_cfg, agent)


if __name__ == "__main__":
    main()
