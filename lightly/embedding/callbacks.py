import os
from typing import Optional

from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from lightly.utils.hipify import print_as_warning


def create_checkpoint_callback(
    save_last: bool = False,
    save_top_k: int = 0,
    monitor: str = "loss",
    dirpath: Optional[str] = None,
) -> ModelCheckpoint:
    """Initializes the checkpoint callback.

    Args:
        save_last:
            Whether or not to save the checkpoint of the last epoch.
        save_top_k:
            Save the top_k model checkpoints.
        monitor:
            Which quantity to monitor.
        dirpath:
            Where to save the checkpoint.

    Returns:
        ModelCheckpoint: The initialized checkpoint callback.

    """
    return ModelCheckpoint(
        dirpath=os.getcwd() if dirpath is None else dirpath,
        filename="lightly_epoch_{epoch:d}",
        save_last=save_last,
        save_top_k=save_top_k,
        monitor=monitor,
        auto_insert_metric_name=False,
    )


def create_summary_callback(
    summary_callback_config: DictConfig, trainer_config: DictConfig
) -> ModelSummary:
    """Creates a model summary callback based on the configuration.

    Args:
        summary_callback_config:
            Configuration dictionary for the summary callback.
        trainer_config:
            Trainer configuration dictionary, which may include deprecated `weights_summary`.

    Returns:
        ModelSummary: The model summary callback.

    """

    # Use .get once and store result
    weights_summary = trainer_config.get("weights_summary", None)

    if weights_summary not in (None, "None"):
        # Deprecated path, return early
        return _create_summary_callback_deprecated(weights_summary)
    # Non-deprecated path
    return ModelSummary(max_depth=summary_callback_config["max_depth"])


def _create_summary_callback(max_depth: int) -> ModelSummary:
    """Initializes the model summary callback.
    See `ModelSummary reference documentation
    <https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelSummary.html?highlight=ModelSummary>`.

    Args:
        max_depth:
            The maximum depth of layer nesting that the summary will include.

    Returns:
        ModelSummary: The initialized model summary callback.

    """
    return ModelSummary(max_depth=max_depth)


def _create_summary_callback_deprecated(weights_summary: str) -> ModelSummary:
    """Constructs summary callback from the deprecated ``weights_summary`` argument.

    The ``weights_summary`` trainer argument was deprecated with the release
    of pytorch lightning 1.7 in 08/2022. Support for this will be removed
    in the future.

    Args:
        weights_summary:
            The deprecated `weights_summary` argument value ("top" or "full").

    Returns:
        ModelSummary: The initialized model summary callback based on the `weights_summary` argument.

    Raises:
        ValueError: If an invalid value is provided for `weights_summary`.

    """
    print_as_warning(
        "The configuration parameter 'trainer.weights_summary' is deprecated."
        " Please use 'trainer.weights_summary: True' and set"
        " 'checkpoint_callback.max_depth' to value 1 for the option 'top'"
        " or -1 for the option 'full'."
    )
    # direct conditional assignment, else early raise
    if weights_summary == "top":
        return ModelSummary(max_depth=1)
    if weights_summary == "full":
        return ModelSummary(max_depth=-1)
    raise ValueError(
        "Invalid value for the deprecated trainer.weights_summary"
        " configuration parameter."
    )
