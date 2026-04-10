# cutlass_int8_convolution
High Performance INT8 Convolution implementing with CUTLASS


# Environment

* CUDA 11.4
* CUTLASS v4.2.0


# Quick Start

```shell
git clone git@github.com:JJXiangJiaoJun/cutlass_int8_convolution.git
cd cutlass_int8_convolution
git clone git@github.com:NVIDIA/cutlass.git
cd cutlass && git checkout v4.2.1 && cd ..
nvcc --std=c++17 -arch=sm_86 --expt-relaxed-constexpr -O2 -I ./ -I ./cutlass/include -I ./cutlass/tools/util/include -DHOST_CHECK int8_convolution_perchannel_residual.cu -o int8_convolution_perchannel_residual
```