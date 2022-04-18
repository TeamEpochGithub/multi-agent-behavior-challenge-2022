from collections.abc import Iterable

import neptune.new as neptune
from sklearn.model_selection._search import ParameterGrid
from torch import nn
from torch.utils.data import Dataset

from lib.train_model import train_model


def grid_search(
    model_class: nn.Module,
    model_params: dict,
    train_params: dict,
    dataset: Dataset,
    val_dataset: Dataset,
    neptune_project: str,
    neptune_token: str,
    validation_metric=None,
    model_args_as_str="",
):
    """
    Performs grid search for the sets of given parameters (model_params and train_params) and
    model_class.

    Options for optimizers, criterions, and lr schedulers should be included in train_params
    :param model_class:
    :param model_params:
    :param train_params:
    :param dataset:
    :param val_dataset:
    :param neptune_project:
    :param neptune_token:
    :param validation_metric:
    :param model_args_as_str: (with a trailing comma)
    :return:
    """
    param_options = ParameterGrid(model_params)

    for (key, value) in train_params.items():
        if not isinstance(value, Iterable):
            train_params[key] = [value]
            print("A train parameter is converted to a list")
    train_param_options = ParameterGrid(train_params)

    print("Performing grid search in:", train_param_options.param_grid)
    print("And", param_options.param_grid)

    for tr_params in train_param_options:
        for params in param_options:
            model_kwargs = ""

            for (key, value) in params.items():
                model_kwargs += f"{key} = {value}, "
            model_kwargs = model_kwargs[:-2]  # remove last comma and space

            model = nn.Module()  # will actually be set inside exec
            # In order to handle arbitrary argument names in constructor, the model class is
            # initialized through exec function and the parameters are given as a string
            exec(f"model = model_class({model_args_as_str}{model_kwargs})")

            neptune_run = neptune.init(
                project=neptune_project,
                api_token=neptune_token,
            )
            train_model(
                model,
                dataset,
                neptune_run,
                tr_params,
                validation_set=val_dataset,
                validation_metric=validation_metric,
            )
            neptune_run.stop()
