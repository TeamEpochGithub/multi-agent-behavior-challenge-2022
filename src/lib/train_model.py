import shutil


from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR
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
    optimizer: Optimizer = None,
    lr_scheduler=None,
    lr_gamma=0.9,
):
    """
    Trains a PyTorch model and logs it to neptune (https://neptune.ai/)
    :param model: torch model, probably a Perceiver
    :param neptune_run: instance of a started neptune run
    :param dataset: torch dataset
    :param params: dict including values: batch size, epochs, learning rate
        And optionally: criterion, optimizer, lr_scheduler, lr_gamma
            (these will not override provided args)
    :param validation_set: dataset used to determine model performance. No set is used by default
        When provided, saves 2 model checkpoints in the current folder: last and best
    :param validation_metric: metric to determine performance (ONLY greater is better).
        Default - use loss (smaller is better)
    :param criterion: for training, on cuda. Defaults to MSE
    :param optimizer: for training. Defaults to Adam with lr from params dict.
        When providing optimizer, ensure you call model.cuda() before initializing the optim
    :param lr_scheduler: class to change lr between epochs. Defaults to Exponential with lr_gamma
        When providing a scheduler (instance), also provide the optimizer it is configured for
    :param lr_gamma: gamma parameter for default ExponentialLR scheduler
    :return: None
    """
    model.cuda()
    model.train()

    losses = AverageMeter()
    best_score = None

    # set model essentials according to provided parameters
    criterion, optimizer, lr_scheduler = _init_model_essentials(
        model, params, criterion, optimizer, lr_scheduler, lr_gamma
    )

    # data transformations
    dataloader = DataLoader(dataset, batch_size=params["batch size"], shuffle=True)
    val_loader = None
    if validation_set is not None:
        val_loader = DataLoader(validation_set, batch_size=params["batch size"], shuffle=True)

    # log to neptune
    neptune_run["parameters"] = params

    for epoch in range(params["epochs"]):
        losses.reset()
        neptune_run["train/lr"].log(lr_scheduler.get_last_lr())
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
        # change learning rate
        lr_scheduler.step()

        neptune_run["train/loss"].log(losses.avg)
        print(f"train loss at epoch {epoch + 1}: {losses.avg}")

        # run on validation set
        if val_loader is not None:
            val_loss, val_score = validate(model, val_loader, criterion, validation_metric)
            model.train()

            best_score, is_best = _compute_best_val_score(
                best_score, val_loss, val_score, validation_metric
            )
            neptune_run["val/loss"].log(val_loss)
            neptune_run["val/score"].log(val_score)
            neptune_run["best score"] = best_score
            print(f"val loss at epoch {epoch + 1}: {val_loss}")

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
    return best_score


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


def load_checkpoint(model, optimizer, path):
    """
    restores model state from checkpoint
    :param model: initialized model of the same class as the saved one
    :param optimizer: initialized optimizer
    :param path: to file
    :return: model, optimizer, loss, epoch
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, loss, epoch


def _init_model_essentials(model, params, criterion, optimizer, lr_scheduler, lr_gamma):
    if criterion is None:
        if "criterion" in params:
            criterion = params["criterion"]
        else:
            criterion = torch.nn.MSELoss().cuda()
    if optimizer is None:
        if "optimizer" in params:
            optimizer = params["optimizer"]
        else:
            optimizer = Adam(model.parameters(), lr=params["learning rate"], amsgrad=True)
    if lr_scheduler is None:
        if "lr_gamma" in params:
            lr_gamma = params["lr_gamma"]
        if "lr_scheduler" in params:
            lr_scheduler = params["lr_scheduler"]
        else:
            lr_scheduler = ExponentialLR(optimizer, gamma=lr_gamma)
    return criterion, optimizer, lr_scheduler


def _compute_best_val_score(best_score, val_loss, val_score, validation_metric) -> (int, bool):
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
    return best_score, is_best

def train_epoch(epoch: int, train_loader, model: nn.Module, criterion, optimizer, config: dict):
    

    loss_epoch = 0
    # tqdm_iter = tqdm(train_loader, total=len(train_loader)) 
	# Total train loader is huge, he're we limiting to steps per epoch
    tqdm_iter = tqdm(train_loader, total=config["steps_per_epoch"])

    tqdm_iter.set_description(f"Epoch {epoch}")
    for step, batch in enumerate(tqdm_iter):
        optimizer.zero_grad()
        x_i = batch['image'][0].cuda(non_blocking=True)
        x_j = batch['image'][1].cuda(non_blocking=True)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)


        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        tqdm_iter.set_postfix(iter_loss=loss.item())

        loss_epoch += loss.item()
        if step >= config["steps_per_epoch"]:
            break

    return loss_epoch


def train(model, config: dict, optimizer, scheduler, criterion, train_loader, neptune_run):

    neptune_run["parameters"] = config

    for epoch in range(config["epochs"]):
        lr = optimizer.param_groups[0]['lr']
        loss_epoch = train_epoch(epoch, train_loader, model, criterion, optimizer, config)
        neptune_run["train/loss"].log(loss_epoch)
        print(f"Loss on epoch {epoch}: {loss_epoch}")

        if scheduler:
            scheduler.step()

        save_checkpoint(model.state_dict(), True)

    save_checkpoint(model.state_dict(), True)




