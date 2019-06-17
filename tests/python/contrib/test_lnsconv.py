import tvm
import numpy as np
from tvm.contrib import lnsconv

def test_conv():
    print("Testing conv3x3")

    ctx = tvm.gpu(0)

    # Number of channels
    n = tvm.var('n')

    A = tvm.placeholder((n,3,3), name="data")
    B = tvm.placeholder((n,3,3), name="weights")
    rx = tvm.reduce_axis((0,3))
    ry = tvm.reduce_axis((0,3))

    As = A[n,rx,ry]
    Bs = B[n,rx,ry]

    D = lnsconv.conv3x3(A,B)
    
    E = tvm.compute( (n,0,0), lambda x: D[x], name='E')

    s = tvm.create_schedule(E.op)
   
    block_y = tvm.thread_axis("blockIdx.y")
    
    s[E].bind(E.op.axis[0], block_y)
    s[E].storage_align(block_y, 64, 0)

    # Show IR 
    print('-------------------- IR -----------------------')
    print(tvm.lower(s,[A,B,E],simple_mode=True)) 

    f = tvm.build(s, [A,B,E], "cuda", target_host="llvm", name="function_wrapper")
    
    # Show cuda
    print('-------------------- CUDA -----------------------')
    print(f.imported_modules[0].get_source())

    # Dump out generated code
    f.export_library("foo.so")
    f.save("foo.o")

if __name__ == "__main__":
    test_conv()


