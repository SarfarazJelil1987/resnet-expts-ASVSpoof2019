import logging
import os
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        self.writer = SummaryWriter(os.path.join(config.output_path, 'tensorboard'))

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.log_dir, 'train.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ASVspoof')

    def log_training(self, epoch, train_loss, val_loss, eer):
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('EER', eer, epoch)

        self.logger.info(
            f'Epoch {epoch}: Train Loss = {train_loss:.4f}, '
            f'Val Loss = {val_loss:.4f}, EER = {eer:.4f}'
        )
