import tvm
import numpy as np
from tvm.contrib import lnsconv

def test_conv():
    print("Testing conv3x3")

    ctx = tvm.gpu(0)

    A = tvm.placeholder((3,3), name="data")
    B = tvm.placeholder((3,3), name="weights")
    C = lnsconv.conv3x3(A,B)
    D = tvm.compute(C.shape, lambda i: C(i), name='D')
    s = tvm.create_schedule(D.op)
   
    block_y = tvm.thread_axis("blockIdx.y")
   
    s[D].bind(D.op.axis[0], block_y)
    s[D].storage_align(block_y, 64, 0)

    f = tvm.build(s, [A,B,D], "cuda", target_host="llvm", name="function_wrapper")
    print(f.imported_modules[0].get_source())


if __name__ == "__main__":
    test_conv()


