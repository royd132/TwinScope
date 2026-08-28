"""Persistence baseline."""
from ._common import LastValue
class Model(LastValue):
    def __init__(self, configs=None): super().__init__(); self.horizon=getattr(configs, "pred_len", 1)
