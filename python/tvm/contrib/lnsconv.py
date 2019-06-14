"""External function interface to lnsconv libraries."""
from __future__ import absolute_import as _abs

from .. import api as _api
from .. import intrin as _intrin
from .._ffi.function import _init_api

def conv3x3(data,weights):
    pass


_init_api("tvm.contrib.lnsconv")
