"""Optimizer and loss helpers required by the Citation release."""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn


def build_optimizer(args, params):
    trainable = filter(lambda parameter: parameter.requires_grad, params)
    if args.opt == "adam":
        optimizer = optim.Adam(
            trainable, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.opt == "sgd":
        optimizer = optim.SGD(
            trainable,
            lr=args.lr,
            momentum=0.95,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "rmsprop":
        optimizer = optim.RMSprop(
            trainable, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.opt == "adagrad":
        optimizer = optim.Adagrad(
            trainable, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.opt}")

    if args.opt_scheduler == "none":
        scheduler = None
    elif args.opt_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.opt_decay_step,
            gamma=args.opt_decay_rate,
        )
    elif args.opt_scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[100, 300, 500, 700, 900],
            gamma=args.opt_decay_rate,
        )
    elif args.opt_scheduler == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.opt_restart
        )
    else:
        raise ValueError(f"Unsupported scheduler: {args.opt_scheduler}")
    return scheduler, optimizer


def CE_loss(pred, label):
    return nn.CrossEntropyLoss()(pred, label.to(torch.int64))


def _source_class_weights(pred, label):
    one_hot = F.one_hot(label, num_classes=pred.shape[1]).float()
    proportions = torch.sum(one_hot, dim=0) / len(one_hot)
    return torch.where(
        proportions > 0,
        1.0 / proportions,
        torch.ones_like(proportions),
    ).to(pred.device)


def CE_srcweight_loss(pred, label):
    label = label.to(torch.int64)
    class_weights = _source_class_weights(pred, label)
    return nn.CrossEntropyLoss(weight=class_weights)(pred, label)


def CE_loss_weight(pred, label, label_weight, weight_src):
    label = label.to(torch.int64)
    num_classes = int(label_weight.numel())
    one_hot = F.one_hot(label, num_classes=num_classes).float()
    if weight_src:
        loss_fn = nn.CrossEntropyLoss(
            reduction="none",
            weight=_source_class_weights(pred, label),
        )
    else:
        loss_fn = nn.CrossEntropyLoss(reduction="none")
    sample_weight = torch.mm(one_hot, label_weight.view(-1, 1).float())
    return torch.mean(loss_fn(pred, label).view(-1, 1) * sample_weight) / num_classes
