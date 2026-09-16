import torch

from formal.engine import EarlyStopping, psrc_residual_loss


def test_early_stopping_waits_for_ten_non_improving_epochs():
    stopper = EarlyStopping(patience=10)

    assert stopper.update(1.0) is False
    for _ in range(9):
        assert stopper.update(1.1) is False
    assert stopper.update(1.1) is True
    assert stopper.best_epoch == 1


def test_early_stopping_resets_after_an_improvement():
    stopper = EarlyStopping(patience=2)

    assert stopper.update(2.0) is False
    assert stopper.update(3.0) is False
    assert stopper.update(1.0) is False
    assert stopper.update(1.5) is False
    assert stopper.update(1.5) is True
    assert stopper.best_epoch == 3


def test_psrc_residual_target_detaches_base():
    base = torch.tensor([[1.0, 1.0]], requires_grad=True)
    correction = torch.tensor([[0.0, 0.0]], requires_grad=True)
    target = torch.tensor([[2.0, 3.0]])

    loss = psrc_residual_loss(base, correction, target, alpha=5.0)
    loss.backward()

    assert base.grad is None
    assert correction.grad is not None
    assert torch.isfinite(correction.grad).all()
