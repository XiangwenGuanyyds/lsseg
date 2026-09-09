from utils import LOSSES


def build_loss(cfg):
    """Build the configured training-loss aggregator."""
    loss_cfg = cfg.LOSS.copy()
    loss_type = loss_cfg.pop("type")

    loss_cls = LOSSES.get(loss_type)
    if loss_cls is None:
        raise KeyError(
            f"Unknown loss '{loss_type}'. Available losses: {LOSSES.keys()}"
        )

    return loss_cls(**loss_cfg)
