from collections.abc import Iterable

import neptune.new as neptune
from sklearn.model_selection._search import ParameterGrid
from torch import nn
from torch.utils.data import Dataset

from lib.train_model import train_model
from lib.util.hash_dict import hash_dict


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
    run_tag=None,
):
    """
    Performs grid search for the sets of given parameters (model_params and train_params) and
    model_class.

    Options for optimizers, criteria, and lr schedulers should be included in train_params
    :param model_class: class itself, not its instance
    :param model_params: dict with model args that have to searched through
    :param train_params: dict with training parameters that have to searched through
        only values that are used by `train_model` are supported
    :param dataset: torch dataset
    :param val_dataset: dataset used to determine model performance.
    :param neptune_project: string like TeamEpoch/MABe-Perceiver
    :param neptune_token: your personal token for neptune
    :param validation_metric: metric to determine performance (ONLY greater is better).
        Default - use loss (smaller is better)
    :param model_args_as_str: a string with parameter values that will NOT be searched through
        for example: input_axis = 1, fourier_encode_data = True
    :param run_tag: tag that will be used for labeling all neptune runs
        Default - hash of parameter dictionaries
    :return:
    """
    if model_args_as_str != "" and model_args_as_str[-2:] != ", " and model_args_as_str[-1] != ",":
        model_args_as_str += ", "

    param_options = ParameterGrid(model_params)

    for (key, value) in train_params.items():
        if not isinstance(value, Iterable):
            train_params[key] = [value]
            print("A train parameter is converted to a list")
    train_param_options = ParameterGrid(train_params)

    if run_tag is None:
        full_dict = dict()
        full_dict.update(model_params)
        full_dict.update(train_params)
        run_tag = hash_dict(full_dict) % 1000000

    print("Performing grid search in:", train_param_options.param_grid)
    print("And", param_options.param_grid)

    # iterate over all possible combinations
    for tr_params in train_param_options:
        for params in param_options:
            print("Training with", tr_params, params)
            # make a string with keyword arguments
            model_kwargs = ""
            for (key, value) in params.items():
                model_kwargs += f"{key} = {value}, "
            model_kwargs = model_kwargs[:-2]  # remove last comma and space

            # In order to handle arbitrary argument names in constructor, the model class is
            # initialized through exec function and the parameters are given as a string
            exec_locals = locals()
            exec(f"model = model_class({model_args_as_str}{model_kwargs})", globals(), exec_locals)
            # after the exec sets the model variable, we need to get it into the current scope
            model: nn.Module = exec_locals["model"]

            neptune_run = neptune.init(
                project=neptune_project,
                api_token=neptune_token,
            )
            neptune_run["sys/tags"].add(f"GS {run_tag}")
            neptune_run["gs tr params"] = tr_params
            neptune_run["gs model params"] = params
            score = train_model(
                model,
                dataset,
                neptune_run,
                tr_params,
                validation_set=val_dataset,
                validation_metric=validation_metric,
            )
            neptune_run.stop()
            print("Best score:", score)
