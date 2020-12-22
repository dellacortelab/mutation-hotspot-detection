#######################################################################################################
# Trainer class
#######################################################################################################


import torch
from torch.optim import Adam

from .logger import Logger
from .evaluate import Evaluator

class Trainer():
    """Vanilla trainer - preset optimizer, learning rate, logger, etc."""
    def __init__(self, model, train_loader, val_loader, objective, lr=None, optimizer=None, logger=None, batch_eval_freq=0, epoch_eval_freq=1, base_savepath='/model', model_name='model', device=None):
        """Initialize all objects needed for Trainer
        Args:
            model (nn.Module): the model to train
            train_loader (torch.utils.data.DataLoader): a dataloader containing the training data
            val_loader (torch.utils.data.DataLoader): a dataloader containing the val data
            objective (nn.Module): the loss function
            lr (float): the learning rate
            optimizer (torch.optim.Optimizer): the optimizer            
            logger (logger.logger.Logger): an object to log results
            batch_eval_freq (int): how many batches between evaluation runs - 0 for no 
                batch-end evaluation
            epoch_eval_freq (int): how many epochs between evaluation runs - 0 for no
                epoch-end evaluation
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.objective = objective
        self.batch_eval_freq = batch_eval_freq
        self.epoch_eval_freq = epoch_eval_freq
        self.evaluator = Evaluator(val_loader, objective, device=device)
        
        if lr is None:
            lr = .0002
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = Adam(model.parameters(), lr=lr)

        if logger is not None:
            self.logger = logger
        else:
            self.logger = Logger(base_savepath=base_savepath, model_name=model_name)

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def train(self, n_epochs=20):
        """Trains the model for the set number of epochs"""
        # Put model on GPU
        self.model.to(self.device)
        # Put model in training mode (include e.g. Dropout layers)
        self.model.train()
        # Initialize eval_loss
        eval_loss = None

        for e in range(n_epochs):

            # Eval if specified
            if self.epoch_eval_freq != 0 and e % self.epoch_eval_freq == 0:
                eval_loss = self.evaluator.eval(self.model)

            for b, x, y_truth in enumerate(self.train_loader):
                
                # Eval if specified
                if self.batch_eval_freq != 0 and b % self.batch_eval_freq == 0:
                    eval_loss = self.evaluator.eval(self.model)

                print("Mem: ", torch.cuda.memory_allocated())
                # Put data on the GPU
                x, y_truth = x.to(self.device), y_truth.to(self.device)
                # Zero the gradients on the model parameters
                self.optimizer.zero_grad()
                # Pass a batch through the network
                y_pred = self.model(x)
                train_loss = self.objective(y_pred, y_truth)
                # Compute the gradient of the loss w.r.t. the network parameters
                train_loss.backward()
                # Take a step of gradient descent
                self.optimizer.step()
                # Log results
                train_loss = train_loss.item()
                self.logger.log_batch(batch=b, train_loss=train_loss, eval_loss=eval_loss, model=self.model)
                
            self.logger.log_epoch(epoch=e, eval_loss=eval_loss, model=self.model)

        print("Final logging")
        train_item = next(iter(self.train_loader))
        # import pdb; pdb.set_trace()
        train_x = train_item[0][0].unsqueeze(0).to(self.device)
        train_y = train_item[1][0]
        val_item = next(iter(self.val_loader))
        val_x = val_item[0][0].unsqueeze(0).to(self.device)
        val_y = val_item[1][0]
        self.logger.log_report(self.model, (train_x, train_y), (val_x, val_y))