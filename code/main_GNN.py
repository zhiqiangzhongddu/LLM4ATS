from code.GNNs.gnn_pretrainer import GNNRunner
from code.utils import set_seed
from code.config import cfg, update_cfg


def main(cfg):
    set_seed(cfg.seed)

    gnn_runner = GNNRunner(cfg=cfg)
    if cfg.task.name == "train":
        gnn_runner.train_and_eval()
    elif cfg.task.name == "pretrain":
        gnn_runner.train_and_eval(task=1)
        gnn_runner.train_and_eval()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    main(cfg)
