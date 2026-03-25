# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import logging
from packaging import version
from .utils import load_config_file, loading_to_local, loading_customized_predictor
from .prediction.predict_by_kernel import nn_predict
from nn_meter.kernel_detector import KernelDetector
from nn_meter.ir_converter import model_file_to_graph, model_to_graph
from nn_meter.utils import get_user_data_folder

logging = logging.getLogger("nn-Meter")

__power_predictors_cfg_filename__ = "power_predictors.yaml"


def list_power_predictors():
    """Return power predictor entries from nn_meter/configs/power_predictors.yaml (copied to ~/.nn_meter/config)."""
    cfg = load_config_file(__power_predictors_cfg_filename__)
    return cfg if cfg is not None else []


def _load_power_predictor_config(predictor_name: str, predictor_version: float = None):
    config = load_config_file(__power_predictors_cfg_filename__) or []
    if len(config) == 0:
        raise NotImplementedError(
            "No power predictors are registered; add entries to power_predictors.yaml."
        )
    preds_info = [
        p
        for p in config
        if p["name"] == predictor_name
        and (predictor_version is None or p["version"] == predictor_version)
    ]
    n_preds = len(preds_info)
    if n_preds == 1:
        return preds_info[0]
    if n_preds > 1:
        latest_version, latest_version_idx = version.parse(str(preds_info[0]["version"])), 0
        for i in range(1, n_preds):
            if version.parse(str(preds_info[i]["version"])) > latest_version:
                latest_version = version.parse(str(preds_info[i]["version"]))
                latest_version_idx = i
        print(
            f"WARNING: There are multiple versions for {predictor_name}, use the latest one ({str(latest_version)})"
        )
        return preds_info[latest_version_idx]
    raise NotImplementedError(
        "No power predictor that meets the required name and version; check power_predictors.yaml."
    )


def load_power_predictor(predictor_name: str, predictor_version: float = None):
    """
    Load a trained power (average watts) predictor by name and optional version.

    predictor_name: Registered name in power_predictors.yaml (see list_power_predictors()).
    predictor_version: If None, the latest matching version is used when multiple exist.
    """
    user_data_folder = get_user_data_folder()
    pred_info = _load_power_predictor_config(predictor_name, predictor_version)
    if "download" in pred_info:
        kernel_predictors, fusionrule = loading_to_local(
            pred_info, os.path.join(user_data_folder, "power_predictor")
        )
    else:
        kernel_predictors, fusionrule = loading_customized_predictor(pred_info)

    return nnMeterPowerPredictor(kernel_predictors, fusionrule)


class nnMeterPowerPredictor:
    """Model-level power estimator using the same kernel graph pipeline as nnMeterPredictor."""

    def __init__(self, predictors, fusionrule):
        self.kernel_predictors = predictors
        self.fusionrule = fusionrule
        self.kd = KernelDetector(self.fusionrule)

    def predict(
        self,
        model,
        model_type,
        input_shape=(1, 3, 224, 224),
        apply_nni=False,
    ):
        """
        Predict average power in watts (W) as the sum of kernel-level power predictions.

        model, model_type, input_shape, apply_nni: same as nnMeterPredictor.predict().
        """
        logging.info("Start power prediction ...")
        if isinstance(model, str):
            graph = model_file_to_graph(
                model, model_type, input_shape, apply_nni=apply_nni
            )
        else:
            graph = model_to_graph(
                model, model_type, input_shape=input_shape, apply_nni=apply_nni
            )

        self.kd.load_graph(graph)
        py = nn_predict(self.kernel_predictors, self.kd.get_kernels())
        logging.info(f"Predict power: {py} W")
        return py
