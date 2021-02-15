from __future__ import print_function
from __future__ import division
import numpy as np
import mxnet as mx
import copy
import math
import random
import itertools
from distutils.version import LooseVersion
from numpy.testing import assert_allclose, assert_array_equal
from mxnet.test_utils import *
from mxnet.operator import *
from mxnet.base import py_str, MXNetError, _as_list
from common import setup_module, with_seed, teardown, assert_raises_cudnn_not_satisfied, assert_raises_cuda_not_satisfied, assertRaises
from common import run_in_spawned_process, with_environment
from nose.tools import assert_raises, ok_
import unittest
import os
import locale
import pytest

def np_instance_norm(data, weight, bias, eps):
    spatial_dims = data.shape[2::]
    num_spatial_vals = np.prod(np.array(spatial_dims))
    scale = 1/float(num_spatial_vals)
    sum_axis = tuple(range(2, data.ndim))
    mean = scale * np.sum(data, axis = sum_axis)
    mean = np.reshape(np.repeat(mean, num_spatial_vals), data.shape)
    var = scale * np.sum((data - mean)**2, axis = sum_axis)
    var = np.reshape(np.repeat(var, num_spatial_vals), data.shape)

    weightBatch = np.tile(weight, (data.shape[0], 1))
    weightBatch = np.reshape(np.repeat(weightBatch, num_spatial_vals), data.shape)
    biasBatch = np.tile(bias, (data.shape[0], 1))
    biasBatch = np.reshape(np.repeat(biasBatch, num_spatial_vals), data.shape)
    return weightBatch * (data - mean)/np.sqrt(var + eps) + biasBatch

def check_instance_norm_with_shape(shape, xpu):
    # bind with label
    eps = 0.001
    X = mx.symbol.Variable('X')
    G = mx.symbol.Variable('G')
    B = mx.symbol.Variable('B')

    Y = mx.symbol.InstanceNorm(data=X, beta=B, gamma=G, eps=eps)
    x = mx.random.normal(0, 1, shape, ctx=mx.cpu()).copyto(xpu)
    gamma = mx.random.normal(0, 1, shape[1], ctx=mx.cpu()).copyto(xpu)
    beta = mx.random.normal(0, 1, shape[1], ctx=mx.cpu()).copyto(xpu)

    np_out = np_instance_norm(x.asnumpy(), gamma.asnumpy(), beta.asnumpy(), eps)
    exec1 = Y.bind(xpu, args = {'X':x, 'G':gamma, 'B':beta})
    exec1.forward(is_train=False)
    out = exec1.outputs[0]
    assert_almost_equal(out, np_out, rtol=1e-4, atol=1e-4)
    check_numeric_gradient(Y, {'X':x.asnumpy(), 'G':gamma.asnumpy(), 'B':beta.asnumpy()},
                           numeric_eps=1e-2, rtol=1e-2, atol=1e-2)

def check_l2_normalization(in_shape, mode, dtype, norm_eps=1e-10):
    ctx = default_context()
    data = mx.symbol.Variable('data')
    out = mx.symbol.L2Normalization(data=data, mode=mode, eps=norm_eps)
    in_data = np.random.uniform(-1, 1, in_shape).astype(dtype)

    exe = out.simple_bind(ctx=ctx, data=in_data.shape)
    output = exe.forward(is_train=True, data=in_data)
    # compare numpy + mxnet
    assert_almost_equal(exe.outputs[0], np_out, rtol=1e-2 if dtype is 'float16' else 1e-5, atol=1e-5)
    # check gradient
    #check_numeric_gradient(out, [in_data], numeric_eps=1e-3, rtol=1e-2, atol=5e-3)

@pytest.mark.serial
def test_l2_normalization():
    for dtype in ['float32']:
        for mode in ['channel']:
            nbatch = 3 #random.randint(1, 4)
            nchannel = 3# random.randint(3, 5)
            height =227 #random.randint(4, 6)
            width = 227
            check_l2_normalization((nbatch, nchannel, height, width), mode, dtype)
            #width = random.randint(5, 7)
            #check_l2_normalization((nbatch, nchannel, height, width), mode, dtype)