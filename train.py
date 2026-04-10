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

    escher.train(train_cfg, agent)


if __name__ == "__main__":
    main()
