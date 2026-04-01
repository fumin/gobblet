import logging
import os
import subprocess

import numpy as np
import torch
import torcheval
from torcheval import metrics as _


_CHECKPOINT_FNAME = "checkpoint.pth"


def atoi(s):
    try:
        i = int(s)
    except Exception as e:
        return -1, str(e)
    
    return i, None


def get_checkpoint_path(cp_root, epoch):
    cp_dir = os.path.join(cp_root, "checkpoint_{:06d}".format(epoch))
    return os.path.join(cp_dir, _CHECKPOINT_FNAME)


def get_paths_desc(fdir):
    err = None
    try:
        fnames = os.listdir(fdir)
    except Exception as e:
        err = str(e)
    if err:
        return []
    
    path_ms = []
    for fname in fnames:
        fpath = os.path.join(fdir, fname)
        noext = os.path.splitext(os.path.basename(fpath))[0]
        splitted = noext.split("_")
        epoch, err = atoi(splitted[-1])
        if err:
            continue
        
        path_ms.append({"path": fpath, "epoch": epoch})
    
    path_ms.sort(key=lambda d: d["epoch"], reverse=True)
    
    return path_ms


def load_checkpoint(cp_root):
    # Torch being stupid by refusing to run numpy when loading checkpoints.
    torch.serialization.add_safe_globals([
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.dtypes.Int64DType,
        np.dtypes.Float32DType,
        np.dtypes.Float64DType,
    ])
    
    paths = get_paths_desc(cp_root)
    if len(paths) == 0:
        return None, ""
    
    for path_d in paths:
        fpath = os.path.join(path_d["path"], _CHECKPOINT_FNAME)
        checkpoint= None
        err = None
        try:
            checkpoint = torch.load(fpath, weights_only=True, map_location=torch.device("cpu"))
        except Exception as e:
            err = "error \"{}\"".format(e)
            logging.info("%s %s", fpath, err)
        if not err:
            return checkpoint, fpath
    
    return None, ""


def delete_old_checkpoints(cp_root):
    paths = get_paths_desc(cp_root)
    
    keep_num = 3
    if len(paths) <= keep_num:
        return

    for path_d in paths[keep_num:]:
        fpath = path_d["path"]
        subprocess.run(["rm", "-r", fpath])


def update_metric(metrics, k, v):
    if not(k in metrics):
        metrics[k] = torcheval.metrics.Mean().to(v.device)
    metrics[k].update(v)
    return metrics
