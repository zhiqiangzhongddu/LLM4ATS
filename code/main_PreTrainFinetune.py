from code.GNNs.gnn_trainer import GNNTrainer
from code.utils import set_seed
from code.config import cfg, update_cfg


def main(cfg):
    set_seed(cfg.seed)

    gnn_trainer = GNNTrainer(cfg=cfg)
    gnn_trainer.train_and_eval()


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    main(cfg)
