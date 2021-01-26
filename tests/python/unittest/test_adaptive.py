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
from nose.tools import assert_raises, ok_
import unittest
import os
import pytest
import locale

@pytest.mark.serial
def test_adaptive_avg_pool_op():
    def py_adaptive_avg_pool(x, height, width):
        # 2D per frame adaptive avg pool
        def adaptive_avg_pool_frame(x, y):
            isizeH, isizeW = x.shape
            osizeH, osizeW = y.shape
            for oh in range(osizeH):
                istartH = int(np.floor(1.0 * (oh * isizeH) / osizeH))
                iendH = int(np.ceil(1.0 * (oh + 1) * isizeH / osizeH))
                kH = iendH - istartH
                for ow in range(osizeW):
                    istartW = int(np.floor(1.0 * (ow * isizeW) / osizeW))
                    iendW = int(np.ceil(1.0 * (ow + 1) * isizeW / osizeW))
                    kW = iendW - istartW
                    xsum = 0
                    for ih in range(kH):
                        for iw in range(kW):
                            xsum += x[istartH+ih][istartW+iw]
                    y[oh][ow] = xsum / kH / kW

        B,C,_,_ = x.shape
        y = np.empty([B,C,height, width], dtype=x.dtype)
        for b in range(B):
            for c in range(C):
                adaptive_avg_pool_frame(x[b][c], y[b][c])
        return y
    def check_adaptive_avg_pool_op(shape, output_height, output_width=None):
        x = mx.nd.random.uniform(shape=shape)
        if output_width is None:
            y = mx.nd.contrib.AdaptiveAvgPooling2D(x, output_size=output_height)
            npy = py_adaptive_avg_pool(x.asnumpy(), output_height, output_height)
        else:
            y = mx.nd.contrib.AdaptiveAvgPooling2D(x, output_size=(output_height, output_width))
            npy = py_adaptive_avg_pool(x.asnumpy(), output_height, output_width)
        assert_almost_equal(y.asnumpy(), npy)
    shape = (2, 2, 10, 10)
    for i in range(1, 11):
        check_adaptive_avg_pool_op(shape, i)
        for j in range(1, 11):
            check_adaptive_avg_pool_op(shape, i, j)

