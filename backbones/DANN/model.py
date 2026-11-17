"""Original full-batch MLP-DANN modules for citation experiments."""

import torch.nn as nn
from torch.autograd import Function


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, inputs, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


def grad_reverse(inputs, lambda_grl):
    return GradientReverseFunction.apply(inputs, lambda_grl)


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, features):
        return self.network(features)


class LabelClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, hidden):
        return self.classifier(hidden)


class DomainClassifier(nn.Module):
    def __init__(self, hidden_dim, dropout=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, hidden, lambda_grl=1.0):
        return self.classifier(grad_reverse(hidden, lambda_grl))
