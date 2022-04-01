import shutil

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lib.util.average_meter import AverageMeter


def train_model(
    model: nn.Module,
    dataset: Dataset,
    neptune_run,
    params: dict,
    validation_set: Dataset = None,
    validation_metric=None,
    criterion=None,
    optimizer=None,
    finish_neptune=False,
):
    """
    Trains a PyTorch model and logs it to neptune (https://neptune.ai/)
    :param model: torch model, probably a Perceiver
    :param neptune_run: instance of a started neptune run
    :param dataset: torch dataset
    :param params: dict including values: batch size, epochs, learning rate
    :param validation_set: dataset used to determine model performance. no set is used by default
    :param validation_metric: metric to determine performance (greater is better).
        Default - use loss (smaller is better)
    :param criterion: for training, on cuda. Defaults to MSE
    :param optimizer: for training. Defaults to Adam with lr from params dict
    :param finish_neptune: whether to stop neptune run
    :return: None
    """
    losses = AverageMeter()
    best_score = None

    # model essentials
    if criterion is None:
        criterion = torch.nn.MSELoss().cuda()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=params["learning rate"], amsgrad=True)

    # data transformations
    dataloader = DataLoader(dataset, batch_size=params["batch size"], shuffle=True)
    val_loader = None
    if validation_set is not None:
        val_loader = DataLoader(validation_set, batch_size=params["batch size"], shuffle=True)

    model.cuda()
    model.train()

    # log to neptune
    neptune_run["parameters"] = params

    for epoch in range(params["epochs"]):
        losses.reset()
        for batch in dataloader:
            data, target = batch
            data = data.cuda(non_blocking=True)  # asynchronous data transfer
            target = target.cuda(non_blocking=True)

            optimizer.zero_grad()

            out = model(data)

            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), data.size(0))

        print(f"train loss at epoch {epoch + 1}: {losses.avg}")
        neptune_run["train/loss"].log(losses.avg)

        # run on validation set
        if val_loader is not None:
            val_loss, val_score = validate(model, val_loader, criterion, validation_metric)
            model.train()

            neptune_run["val/loss"].log(val_loss)
            neptune_run["val/score"].log(val_score)
            print(f"val loss at epoch {epoch + 1}: {val_loss}")
            if validation_metric is not None:
                if best_score is None:
                    best_score = val_score
                is_best = val_score > best_score
                best_score = max(val_score, best_score)
            else:
                # fallback to loss
                if best_score is None:
                    best_score = val_loss
                is_best = val_loss < best_score
                best_score = min(val_loss, best_score)
            neptune_run["best score"] = best_score
            # save model
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
            )

    if finish_neptune:
        neptune_run.stop()


def validate(model: nn.Module, val_dataloader: DataLoader, criterion, metric=None):
    """

    :param model: model to validate
    :param val_dataloader: validation data
    :param criterion: loss function
    :param metric: answer assessment function
    :return: average loss and metric score
    """
    losses = AverageMeter()
    metric_scores = AverageMeter()
    model.eval()

    with torch.no_grad():
        for batch in val_dataloader:
            data, target = batch
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            out = model(data)

            loss = criterion(out, target)
            losses.update(loss.item(), data.size(0))
            if metric is not None:
                score = metric(out, target)
                metric_scores.update(score, data.size(0))
    return losses.avg, metric_scores.avg


def save_checkpoint(state: dict, is_best: bool, filename="model_checkpoint.pth.tar"):
    """
    saves model to a file
    :param state: state dict
    :param is_best:
    :param filename:
    :return: None
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")
