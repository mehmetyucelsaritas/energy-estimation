# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from torch import nn
from .interface import BaseTestCase


class SingleOpModel(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, inputs):
        return self.op(inputs)


class TwoOpModel(nn.Module):
    def __init__(self, op1, op2, op1_is_two_inputs, op2_is_two_inputs):
        super().__init__()
        self.op1 = op1
        self.op2 = op2
        self.op1_is_two_inputs = op1_is_two_inputs
        self.op2_is_two_inputs = op2_is_two_inputs

    def forward(self, inputs):
        if self.op1_is_two_inputs:
            x = self.op1([inputs[0], inputs[1]])
        else:
            if self.op2_is_two_inputs:
                x = self.op1(inputs[0])
            else:
                x = self.op1(inputs)
        if self.op2_is_two_inputs:
            x = self.op2([x, inputs[-1]])
        else:
            x = self.op2(x)
        return x


class MultipleOutNodes(BaseTestCase):
    name = 'MON'
    cases = {
        'case1': ['relu_relu', 'relu_dwconv', 'dwconv'],
        'case2': ['dwconv_relu_relu', 'relu_dwconv'],
        'case3': ['dwconv_relu', 'dwconv', 'relu_relu']
    }
    true_case = 'case1'
    deps = {
        'BF_dwconv_relu': True,
    }
    implement = 'torch'

    def load_config(self):
        super().load_config()
        c = self.config
        # torch uses NCHW; BaseTestCase defaults to [HW, HW, CIN] for Keras
        self.input_shape = [c['CIN'], c['HW'], c['HW']]

    def _dw_layer(self):
        from nn_meter.builder.nn_modules.torch_networks.utils import get_padding

        cin = self.input_shape[0]
        pad = get_padding(
            self.kernel_size, self.config['STRIDES'], self.input_shape[1]
        )
        return nn.Conv2d(
            cin,
            cin,
            kernel_size=self.kernel_size,
            stride=self.config['STRIDES'],
            padding=pad,
            groups=cin,
        )

    def _model_block(self):
        class Block(nn.Module):
            def __init__(self, dw1, dw2):
                super().__init__()
                self.dw1 = dw1
                self.relu0 = nn.ReLU()
                self.relu1 = nn.ReLU()
                self.relu_slope = nn.LeakyReLU(negative_slope=2.0)
                self.dw2 = dw2

            def forward(self, x):
                t = self.dw1(x)
                branch_1 = self.relu1(self.relu0(t))
                branch_2 = self.dw2(self.relu_slope(t))
                return branch_1, branch_2

        return Block(self._dw_layer(), self._dw_layer()), [self.input_shape]

    def _model_relu_relu(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.r1 = nn.ReLU()
                self.r2 = nn.ReLU()

            def forward(self, x):
                return self.r2(self.r1(x))

        return Net(), [self.input_shape]

    def _model_dwconv_relu_relu(self):
        class Net(nn.Module):
            def __init__(self, dw):
                super().__init__()
                self.dw = dw
                self.r1 = nn.ReLU()
                self.r2 = nn.ReLU()

            def forward(self, x):
                return self.r2(self.r1(self.dw(x)))

        return Net(self._dw_layer()), [self.input_shape]

    def _model_relu_dwconv(self):
        class Net(nn.Module):
            def __init__(self, dw):
                super().__init__()
                self.r = nn.ReLU()
                self.dw = dw

            def forward(self, x):
                return self.dw(self.r(x))

        return Net(self._dw_layer()), [self.input_shape]

    def _model_dwconv_relu(self):
        class Net(nn.Module):
            def __init__(self, dw):
                super().__init__()
                self.dw = dw
                self.r = nn.ReLU()

            def forward(self, x):
                return self.r(self.dw(x))

        return Net(self._dw_layer()), [self.input_shape]

    def _model_dwconv(self):
        class Net(nn.Module):
            def __init__(self, dw):
                super().__init__()
                self.dw = dw

            def forward(self, x):
                return self.dw(x)

        return Net(self._dw_layer()), [self.input_shape]

