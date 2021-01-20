#######################################################################################################
# Logger class
#######################################################################################################


from matplotlib import pyplot as plt
import os
import torch
import numpy as np

class Logger():
    """Generic logger class. Logs loss, checkpoints the model every `chckpt_freq` epochs"""
    def __init__(self, model_name, chckpt_freq=5, save_model=True, base_savepath='/models', log_dir='/data/logs'):
        
        self.chckpt_freq = chckpt_freq
        self.save_model = True
        self.base_savepath = base_savepath
        self.model_savepath = os.path.join(base_savepath, model_name)
        self.batch_log_report_path = os.path.join(log_dir, model_name + '_batch_loss.png')
        self.epoch_log_report_path = os.path.join(log_dir, model_name + '_epoch_loss.png')

        self.batch_training_losses = []
        self.epoch_training_losses = []
        self.batch_eval_losses = []
        self.epoch_eval_losses = []
        self.eval_batches = []
        self.eval_epochs = []

        if not os.path.exists(base_savepath):
            os.makedirs(base_savepath)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def log_batch(self, batch, train_loss, eval_loss, model):
        """Does end-of-batch logging
        Args:
            batch (int): the batch number
            train_loss (float): the training loss for the batch. Note: this should be a float, 
                NOT a FloatTensor. A FloatTensor will lead to memory issues.
            eval_loss (float): the evaluation loss. Note: this should be a float, NOT a 
                FloatTensor. A FloatTensor will lead to memory issues.
            model (nn.Module): the neural network model
        """
        self.batch_training_losses.append(train_loss)
        if eval_loss is not None:
            self.eval_batches.append(batch)
            self.batch_eval_losses.append(eval_loss)

    def log_epoch(self, epoch, eval_loss, model):
        """Does end-of-epoch logging
        Args:
            batch (int): the batch number
            eval_loss (float): the evaluation loss. Note: this should be a float, NOT a 
                FloatTensor. A FloatTensor will lead to memory issues.
            model (nn.Module): the neural network model
        """
        n_losses = 10
        # Get avg loss
        avg_batch_loss = sum(self.batch_training_losses[-n_losses:]) / n_losses
        # Track loss
        self.epoch_training_losses.append(avg_batch_loss)

        # Track eval loss
        if eval_loss is not None:
            self.eval_epochs.append(epoch)
            self.epoch_eval_losses.append(eval_loss)

        # Save a model checkpoint
        if self.save_model and epoch % self.chckpt_freq == 0:
            torch.save(model.state_dict(), self.model_savepath + '.pt')

    def log_report(self, model, train_item, test_item):
        """Does end-of-training logging. Plots the losses
        Args:
            model (nn.Module): the neural network model
        """
        if self.save_model:
            torch.save(model.state_dict(), self.model_savepath + '.pt')
        # Save per-batch losses
        fig = plt.figure()
        plt.plot(np.arange(len(self.batch_training_losses)), self.batch_training_losses, label='training loss')
        plt.plot(self.eval_batches, self.batch_eval_losses, label='test loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Loss over Training')
        plt.legend()
        plt.savefig(self.batch_log_report_path)

        # Save per-epoch losses
        fig = plt.figure()
        plt.plot(np.arange(len(self.epoch_training_losses)), self.epoch_training_losses, label='training loss')
        plt.plot(self.eval_epochs, self.epoch_eval_losses, label='test loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Training')
        plt.legend()
        plt.savefig(self.epoch_log_report_path)