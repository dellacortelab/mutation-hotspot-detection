##########################################################################
# Evaluator class for getting loss and accuracy on the validation dataset
##########################################################################

import torch

class Evaluator():
    """Model evaluator. Evaluates model on val set"""
    def __init__(self, val_loader, objective, device=None):
        """Initialize member variables
        Args:
            val_loader (torch.utils.data.DataLoader): the data loader for the val set
            objective (nn.Module): the loss function
            device (torch.device)
        """
        self.val_loader = val_loader
        self.objective = objective

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def eval(self, model):
        """Evaluates the model on the val data
        Args:
            model (nn.Module): the model to evaluate
        Returns:
            (float): the average loss of the model
        """
        # Send model to GPU
        model.to(self.device)
        # Put model in eval mode, removing Dropout layers, etc.
        model.eval()
        # Track loss
        eval_losses = []
        # Evaluate model
        with torch.no_grad():
            for x, y_truth in self.val_loader:
                # print("Mem: ", torch.cuda.memory_allocated())
                # Put data on GPU
                x, y_truth = x.to(self.device), y_truth.to(self.device)
                # Pass a batch through the network
                y_pred = model(x)
                # Compute the loss
                loss = self.objective(y_pred, y_truth)
                eval_losses.append(loss.item())
        
        # Put model back in train mode, adding Dropout layers, etc.
        model.train()

        return sum(eval_losses) / len(eval_losses)