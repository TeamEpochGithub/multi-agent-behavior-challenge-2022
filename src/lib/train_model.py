import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset


def train_model(
    model: nn.Module,
    dataset: Dataset,
    neptune_run,
    params: dict,
    dataset_config: dict = None,
    criterion=None,
    optimizer=None,
    finish_neptune=False,
):
    """
    Trains a PyTorch model and logs it to neptune (https://neptune.ai/)
    :param model: torch model, probably a Perceiver
    :param neptune_run: instance of a started neptune run
    :param dataset: torch dataset
    :param params: dict including values: batch size, epochs
    :param dataset_config: dict that was used for dataset creation. some values may be added
    :param criterion: for training, on cuda. Defaults to MSE
    :param optimizer: for training. Defaults to Adam with lr from params dict
    :param finish_neptune: whether to stop neptune run
    :return: None
    """
    # model essentials
    if not criterion:
        criterion = torch.nn.MSELoss().cuda()
    if not optimizer:
        optimizer = optim.Adam(model.parameters(), lr=params["learning rate"], amsgrad=True)

    # data transformations
    dataloader = DataLoader(dataset, batch_size=params["batch size"], shuffle=True)

    model.cuda()
    model.train()

    # log to neptune
    neptune_run["parameters"] = params
    if dataset_config:
        # complete Datasets have __len__ implementation, so the warning is wrong
        # (also read https://github.com/pytorch/pytorch/blob/master/torch/utils/data/sampler.py)
        # noinspection PyTypeChecker
        dataset_config["dataset size"] = len(dataset)
        neptune_run["dataset config"] = dataset_config

    for epoch in range(params["epochs"]):
        running_loss = 0.0
        for batch in dataloader:
            data, answers = batch
            data = data.cuda()
            answers = answers.cuda()

            optimizer.zero_grad()
            out = model(data)

            loss = criterion(out, answers)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        # noinspection PyTypeChecker
        avg_loss = running_loss / len(dataset)
        print(f"loss at epoch {epoch}: {avg_loss}")
        neptune_run["train/loss"].log(avg_loss)

    if finish_neptune:
        neptune_run.stop()
