# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import logging
from glob import glob
from nn_meter import list_power_predictors, load_power_predictor


def list_power_predictors_cli():
    preds = list_power_predictors()
    logging.keyinfo("Supported power predictors:")
    for p in preds:
        logging.result(f"[Power predictor] {p['name']}: version={p['version']}")
    return


def apply_power_predictor_cli(args):
    """Apply power predictor to predict model average power (W) from CLI arguments."""

    if args.tensorflow:
        input_model, model_type, model_suffix = args.tensorflow, "pb", ".pb"
    elif args.onnx:
        input_model, model_type, model_suffix = args.onnx, "onnx", ".onnx"
    elif args.nn_meter_ir:
        input_model, model_type, model_suffix = args.nn_meter_ir, "nnmeter-ir", ".json"
    elif args.torchvision:
        input_model_list, model_type = args.torchvision, "torch"
    else:
        logging.keyinfo('please run "nn-meter predict_power --help" to see guidance.')
        return

    if not args.predictor:
        logging.keyinfo(
            'You must specify a predictor. Use "nn-meter --list-power-predictors" to see all supporting power predictors.'
        )
        return

    predictor = load_power_predictor(args.predictor, args.predictor_version)

    if not args.torchvision:
        input_model_list = []
        if os.path.isfile(input_model):
            input_model_list = [input_model]
        elif os.path.isdir(input_model):
            input_model_list = glob(os.path.join(input_model, "**" + model_suffix))
            input_model_list.sort()
            logging.info(f"Found {len(input_model_list)} model in {input_model}. Start prediction ...")
        else:
            logging.error("Cannot find any model satisfying the arguments.")

    result = {}
    for model in input_model_list:
        power_w = predictor.predict(model, model_type)
        result[os.path.basename(model)] = power_w
        logging.result(f"[RESULT] predict power for {os.path.basename(model)}: {power_w} W")

    return result
