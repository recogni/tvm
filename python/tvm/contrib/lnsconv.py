"""External function interface to lnsconv libraries."""
from __future__ import absolute_import as _abs

from .. import api as _api
from .. import intrin as _intrin
from .._ffi.function import _init_api

def conv3x3(data,weights):
    return _api.extern( (data.shape[0]), [data, weights], 
        lambda ins, outs: _intrin.call_packed("tvm.contrib.lnsconv.conv3x3", ins[0], ins[1], outs[0]), dtype='float32')

