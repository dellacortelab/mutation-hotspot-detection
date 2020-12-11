#######################################################################################################
# Objective functions for the Transformer AutoEncoder
#######################################################################################################

from torch import nn


class AutoEncoderLoss(nn.Module):
    def __init__(self, n_classes):
        super(TransformerAutoEncoder, self).__init__()

        self.objective = nn.CrossEntropyLoss()
        self.n_classes = n_classes

    def forward(self, y_pred, y_truth):
        return self.objective(y_pred.view(-1, self.n_classes), y_truth.view(-1))


class AutoEncoderFineTuneLoss(nn.Module):
    def __init__(self, n_classes, reconstruction_weight=1., fine_tune_weight=1.):
        super(TransformerAutoEncoderReactivityPredictor, self).__init__()

        self.reconstruction_weight = reconstruction_weight
        self.fine_tune_weight = fine_tune_weight
        self.reconstruction_loss = AutoEncoderLoss(n_classes)
        # MSE loss for regression task
        self.fine_tune_loss = nn.MSELoss()

    def forward(self, y_pred, y_truth):
        return  self.reconstruction_weight * self.reconstruction_loss(y_pred, y_truth) + \
                self.fine_tune_weight * self.fine_tune_loss(y_pred, y_truth)