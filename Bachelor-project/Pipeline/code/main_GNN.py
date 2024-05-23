from Pipeline.code.GNNs.gnn_trainer import GNNTrainer
from Pipeline.code.utils import set_seed
from Pipeline.code.config import cfg, update_cfg
from Pipeline.code.data_utils.utils import check_gnn_predictions


def main(cfg):
    set_seed(cfg.seed)

    gnn_trainer = GNNTrainer(cfg=cfg)
    gnn_trainer.train_and_eval()


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    main(cfg)
