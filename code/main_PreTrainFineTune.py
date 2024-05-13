from code.GNNs.gnn_pretrainer import GNNPreTrainer
from code.utils import set_seed
from code.config import cfg, update_cfg


def main(cfg):
    set_seed(cfg.seed)

    gnn_pretrainer = GNNPreTrainer(cfg=cfg)
    gnn_pretrainer.pretrain_and_eval()
    gnn_pretrainer.finetune_and_eval()


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    main(cfg)
