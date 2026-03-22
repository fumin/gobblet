import argparse
import datetime
import json
import logging
import os
import time

import numpy as np
import torch
import torch.utils.tensorboard as _

import util


def read(data, idx, fpath, want):
    # start_time = time.time()
    fbody = np.loadtxt(fpath, dtype=np.uint8, delimiter=",")
    if want == fbody.shape[0]:
        data[idx:idx+want] = fbody
    else:
        indices = np.random.choice(fbody.shape[0], want, replace=False)
        data[idx:idx+len(indices)] = fbody[indices]
    # logging.info("read %s %d secs", fpath, int(time.time()-start_time))


class Data:
    def __init__(self, fdir):
        self.fdir = fdir

        meta_path = os.path.join(fdir, "meta.json")
        with open(meta_path) as f:
            meta_b = f.read()
        meta = json.loads(meta_b)

        self.cols = meta["cols"]
        self.files = meta["files"]
        for i in range(len(self.files)):
            self.files[i]["path"] = os.path.join(fdir, self.files[i]["name"])

        self.wrongs = None
        # self.wrongs = torch.from_numpy(np.loadtxt("gsolve_wrong.csv", dtype=np.uint8, delimiter=",")).to(torch.device("cuda:0"))

    def get_array(self, size):
        if self.wrongs is not None:
            return self.wrongs
        start_time = time.time()

        num_reads = 12 # len(self.files)
        files = [self.files[i] for i in np.random.choice(len(self.files), num_reads, replace=False)]
        total = np.sum([f["lines"] for f in files])
        wants = []
        for i, f in enumerate(files):
            w = int(np.ceil(size*f["lines"]/total))
            w = min(w, f["lines"])
            wants.append(w)

        data = np.zeros([np.sum(wants), self.cols], dtype=np.uint8)
        idx = 0
        for i, f in enumerate(files):
            read(data, idx, f["path"], wants[i])
            idx += wants[i]

        logging.info("get_tensor %d secs", int(time.time()-start_time))
        return data


class Model(torch.nn.Module):
    def __init__(self, sizes):
        super().__init__()

        layers = []
        for i in range(len(sizes)-2):
            layers.append(torch.nn.Linear(sizes[i], sizes[i+1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(sizes[-2], sizes[-1]))
        self.model = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)


class Config:
    def __init__(self):
        self.net = [1024, 1024, 1024]
        self.batch_size = 256
        self.learning_rate = 1e-4


class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.t = 0
        self.net = Model([54] + cfg.net + [108])

    def _set_device(self, device):
        self.net.to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.learning_rate)


def _get_loss(agent, x, y):
    x = x.to(torch.float32)
    y = torch.squeeze(y, dim=[1])

    logits = agent.net(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    return torch.mean(loss)


def _save_checkpoint(cp_dir, agent):
    cp = {}
    cp["t"] = agent.t
    cp["net"] = agent.net.state_dict()

    os.makedirs(cp_dir, exist_ok=True)
    cp_path = util.get_checkpoint_path(cp_dir, agent.t)
    torch.save(cp, cp_path)
    util.delete_old_checkpoints(cp_dir)


def _load_checkpoint(agent, cp_dir):
    torch.serialization.add_safe_globals([])

    cp, cp_path = util.load_checkpoint(cp_dir)
    if not cp:
        logging.info("no checkpoint")
        return

    agent.t = cp["t"]
    agent.net.load_state_dict(cp["net"])

    logging.info("loaded checkpoint %s", cp_path)


def _train_epoch(cfg, data_q, agent):
    ts = data_q.get()
    if torch.is_tensor(ts):
        x = ts[:, :54]
        y = ts[:, 54:]
    else:
        x = torch.from_numpy(ts[:, :54]).to(cfg.device)
        y = torch.from_numpy(ts[:, 54:]).to(cfg.device)
    del ts
    import gc
    gc.collect()

    batch_size = agent.cfg.batch_size
    indices = list(range(x.shape[0]))
    np.random.shuffle(indices)
    idxs = [indices[n:n+batch_size] for n in range(0, len(indices), batch_size)]

    metrics = {}
    for batch_i in idxs:
        xb = x[batch_i]
        yb = y[batch_i]

        loss = _get_loss(agent, xb, yb)

        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        metrics = util.update_metric(metrics, "loss", loss)

    for k, vm in metrics.items():
        v = vm.compute()
        cfg.summary_writer.add_scalar(k, v, agent.t)
        logging.info("%d %s %f", agent.t, k, v)


def _save_agent_config(run_dir, agent):
    jstr = json.dumps(agent.cfg.__dict__)
    cfg_dir = os.path.join(run_dir, "config")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, "config_{:06d}.json".format(agent.t))
    with open(cfg_path, "w") as f:
        f.write(jstr)


class TrainConfig:

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.device_name = "cpu"
        self.eval_every = 1

    def setup(self):
        self.summary_writer = torch.utils.tensorboard.SummaryWriter(self.run_dir)
        self.device = torch.device(self.device_name)


def train(cfg, data_q, agent):
    cp_dir = os.path.join(cfg.run_dir, "checkpoint")
    _load_checkpoint(agent, cp_dir)
    agent._set_device(cfg.device)
    _save_agent_config(cfg.run_dir, agent)

    while True:
        agent.t += 1
        _train_epoch(cfg, data_q, agent)
        if agent.t % cfg.eval_every == 0:
            _save_checkpoint(cp_dir, agent)
            cfg.summary_writer.flush()


def produce(data_q, data):
    while True:
        ts = data.get_array(12*5000000)
        data_q.put(ts)


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    [lg.removeHandler(h) for h in lg.handlers]
    lg.addHandler(logging.StreamHandler())
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", dest="data_dir", default="")
    parser.add_argument("-run_dir", dest="run_dir", default="runs/test")
    args = parser.parse_args()

    data = Data(args.data_dir)
    import queue
    import threading
    data_q = queue.Queue(maxsize=1)
    data_t = threading.Thread(target=produce, args=[data_q, data])
    data_t.start()

    cfg = Config()
    cfg.net = [256, 256, 256, 256, 256, 256, 256, 256]
    cfg.batch_size = 64*1024
    cfg.learning_rate = 1e-4
    agent = Agent(cfg)

    train_cfg = TrainConfig(args.run_dir)
    train_cfg.device_name = "cuda:0"
    train_cfg.eval_every = 1
    train_cfg.setup()
    train(train_cfg, data_q, agent)


if __name__ == "__main__":
    main()
