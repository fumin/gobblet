import argparse
import datetime
import json
import logging
import os

import pyspiel

import escher
import gobblet
import util


def _load_config(run_dir):
    cfg_dir = os.path.join(run_dir, "config")
    cfg_paths = util.get_paths_desc(cfg_dir)
    last_path = cfg_paths[-1]["path"]
    with open(last_path, "r") as f:
        jstr = f.read()
    cfg_dict = json.loads(jstr)

    cfg = escher.Config()
    cfg.__dict__.update(cfg_dict)
    return cfg


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    [lg.removeHandler(h) for h in lg.handlers]
    lg.addHandler(logging.StreamHandler())
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", dest="game", default="kuhn_poker")
    parser.add_argument("-d", dest="run_dir", default="runs/test")
    args = parser.parse_args()

    game = pyspiel.load_game(args.game)

    cfg = _load_config(args.run_dir)
    agent = escher.Agent(game, cfg)
    cp_dir = os.path.join(args.run_dir, "checkpoint")
    escher._load_checkpoint(agent, cp_dir, True)
    logging.info("avg_policy_buffer %d", agent.avg_policy_buffer.add_calls)
    for p in range(len(agent.regret_buffers)):
        logging.info("regret_buffer_%d %d", p, agent.regret_buffers[p].add_calls)

    if args.game == "kuhn_poker":
        conv = escher._calc_nashconv(game, agent)
        logging.info("nashconv %f", conv)
    reward = escher._play_against_random(escher.TrainConfig(game), agent, 1000)
    logging.info("reward_vs_random %f", reward)


if __name__ == "__main__":
    main()
