from Pipeline.code.GNNs.gnn_pretrainer import GNNPreTrainer
from Pipeline.code.utils import set_seed
from Pipeline.code.config import cfg, update_cfg
import pathlib
import torch
import pandas as pd
import numpy as np
path_to_dir = pathlib.Path(__file__).parent.resolve()


def main(cfg):
    set_seed(cfg.seed)
    # Extract the computed values
    df = pd.read_csv(str(path_to_dir) + f'/../computed_values/comp_vals.csv')
    df = df.dropna(axis=1, how='all')
    pretrain_values = torch.tensor(df.values)
    gnn_pretrainer = GNNPreTrainer(cfg=cfg, aux_values=pretrain_values)
    gnn_pretrainer.pretrain_and_eval()
    gnn_pretrainer.finetune_and_eval()


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    main(cfg)
