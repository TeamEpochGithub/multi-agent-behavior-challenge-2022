import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.mice_dataset import MiceDataset


def train_model(
    model,
    neptune_run,
    X: torch.tensor,
    y: torch.tensor,
    params: dict,
    dataset_config: dict,
    batch_size=16,
):
    """
    TODO
    :param model:
    :param neptune_run:
    :param X:
    :param y:
    :param params:
    :param dataset_config:
    :param batch_size:
    :return:
    """
    # model essentials
    criterion = torch.nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)

    # data transformations
    dataset = MiceDataset(torch.Tensor(X), torch.Tensor(y))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.cuda()
    model.train()

    # log to neptune
    params["batch size"] = batch_size
    dataset_config["dataset size"] = len(dataset)
    neptune_run["parameters"] = params
    neptune_run["dataset config"] = dataset_config

    for epoch in range(20):
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

            # print statistics
            running_loss += loss.item()
        avg_loss = running_loss / len(dataset)
        print(f"loss at epoch {epoch}: {avg_loss}")
        neptune_run["train/loss"].log(avg_loss)

    # submit neptune
    neptune_run.stop()
