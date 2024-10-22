from code.gnns.gnn_runner import GNNRunner
from code.utils import set_seed
from code.config import cfg, update_cfg


def main(cfg):
    set_seed(cfg.seed)

    gnn_runner = GNNRunner(cfg=cfg)
    if cfg.task.mode == "scratch":
        gnn_runner.train_and_eval()
    elif cfg.task.mode == "train": # TBD
        gnn_runner.train_and_eval()
    elif cfg.task.mode == "pretrain": # TBD
        gnn_runner.train_and_eval()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    main(cfg)
