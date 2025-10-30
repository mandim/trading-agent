# tb_logger.py
import os, shutil
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class TBLogger:
    def __init__(self, base_dir="runs", run_tag=None, clean=False):
        """
        base_dir: parent log folder (e.g., 'runs')
        run_tag : optional string appended to the run name (e.g., 'DQN_EURUSD')
        clean   : if True, deletes the entire base_dir before creating the new run
        """
        if clean and os.path.exists(base_dir):
            shutil.rmtree(base_dir)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{stamp}" + (f"_{run_tag}" if run_tag else "")
        self.run_dir = os.path.join(base_dir, name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.run_dir)

    def add_hparams(self, hparams: dict, metrics: dict | None = None):
        # Metrics can be empty now; they’ll show in the hparams tab once you log some scalars
        self.writer.add_hparams(hparams, metrics or {})

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_figure(self, tag: str, fig, step: int):
        self.writer.add_figure(tag, fig, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()
