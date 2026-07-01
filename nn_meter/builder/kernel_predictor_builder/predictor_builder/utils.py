# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
from nn_meter.utils import get_conv_flop_params, get_dwconv_flop_params, get_separable_dwconv_flop_params, get_fc_flop_params


def get_flops_params(kernel_type, config):
    if "separable-dwconv" in kernel_type:
        input_h = config["INPUT_H"] if "INPUT_H" in config else config["HW"]
        input_w = config["INPUT_W"] if "INPUT_W" in config else config["HW"]
        cin = config["CIN"]
        kh = config.get("KERNEL_H", config.get("KERNEL_SIZE", 3))
        kw = config.get("KERNEL_W", kh)
        sh = config.get("STRIDE_H", config.get("STRIDES", 1))
        sw = config.get("STRIDE_W", sh)
        return get_separable_dwconv_flop_params(input_h, input_w, cin, kh, kw, sh, sw)
    elif "dwconv" in kernel_type:
        hw, cin, kernel_size, stride = config["HW"], config["CIN"], \
            config["KERNEL_SIZE"], config["STRIDES"]
        return get_dwconv_flop_params(hw, cin, kernel_size, stride)
    elif "conv" in kernel_type:
        hw, cin, cout, kernel_size, stride = config["HW"], config["CIN"], \
            config["COUT"], config["KERNEL_SIZE"], config["STRIDES"]
        return get_conv_flop_params(hw, cin, cout, kernel_size, stride)
    elif "fc" in kernel_type:
        cin, cout = config["CIN"], config["COUT"]
        return get_fc_flop_params(cin, cout)


def collect_kernel_data(kernel_data, predict_label = 'latency'):
    if isinstance(kernel_data, dict):
        return kernel_data

    config, label = kernel_data
    if isinstance(config, list):
        config = collect_data(config)
    else:
        with open(config, 'r') as fp:
            config = json.load(fp)

    if isinstance(label, list):
        label = collect_data(label)
    else:
        with open(label, 'r') as fp:
            label = json.load(fp)
    if predict_label == 'latency':
        from nn_meter.builder.backend_meta.utils import read_profiled_results
        label = read_profiled_results(label)

    for modules in config.keys():
        for model_id in config[modules].keys():
            try:
                config[modules][model_id][predict_label] = label[modules][model_id][predict_label]
            except:
                pass

    return config


def collect_data(file_list):
    file_list_copy = file_list[:]

    from ...utils import merge_info
    data = file_list_copy.pop(0)
    with open(data, 'r') as fp:
        data = json.load(fp)
    for file in file_list_copy:
        data = merge_info(new_info=file, prev_info=data)
    return data
