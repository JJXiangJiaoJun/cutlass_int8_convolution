#include <random>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/epilogue/thread/activation.h"

#include "utils/initializer.h"
#include "conv/kernel/default_conv2d_fprop_quant_with_broadcast.h"
#include "epilogue/thread/linear_combination_scale_add_bias_elementwise_quant_perchannel.h"
#include "host/operation/conv2dbroadcast.h"


#define CHECK_CUDA_ERROR(call)                                                           \
  {                                                                                      \
    cudaError_t err = call;                                                              \
    if (err != cudaSuccess) {                                                            \
      printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(-1);                                                                          \
    }                                                                                    \
  }

using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = int8_t;
using ElementOutput = ElementC;
using ElementResidual = ElementC;
using ElementScaleBias = float;
using ElementCompute = ElementScaleBias;
using ElementAccumulator = int32_t;

using LayoutA = cutlass::layout::TensorNHWC;
using LayoutB = cutlass::layout::TensorNHWC;
using LayoutC = cutlass::layout::TensorNHWC;

// Epilogue: D = Identity(plus(scale * accum + bias, beta * residual))
//         = scale * accum + bias + beta * residual
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationAddBiasElementwiseQuantPerChannel<
    ElementC,                                                   // ElementOutput = int8_t
    8,                                                          // Count = 8
    ElementAccumulator,                                         // int32_t
    ElementCompute,                                             // float
    ElementResidual,                                            // Residual element type = int8_t
    cutlass::epilogue::thread::ReLU<ElementCompute>,            // Identity activation
    cutlass::plus<ElementCompute>                               // BinaryOp = plus
>;

using InnerKernelAmpere = typename cutlass::conv::kernel::DefaultConv2dFpropQuantPerChannelWithBroadcast<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementScaleBias,
  LayoutC,
  ElementC,
  LayoutC,
  ElementAccumulator,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128, 128, 64>,
  cutlass::gemm::GemmShape<64, 64, 64>,
  cutlass::gemm::GemmShape<16, 8, 32>,
  // Output = alpha * (A@B) + beta * C
  EpilogueOp,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3,
  cutlass::arch::OpMultiplyAddSaturate,
  cutlass::conv::IteratorAlgorithm::kOptimized,
  cutlass::conv::StrideSupport::kStrided,
  16,
  16
>::Kernel;


using TestKernel = cutlass::conv::device::ImplicitGemmConvolutionAdapter<InnerKernelAmpere>;

// Host reference kernel: (int8 specialization)
// Formula: output = act(round(binary_op(scale * accum + bias, beta * residual)))


void device_kernel(const ElementScaleBias* scale,
                   const ElementA* input,
                   const ElementB* weight,
                   const ElementScaleBias* bias,
                   const ElementC* residual,
                   ElementC* output,
                   int n,
                   int h,
                   int w,
                   int c,
                   int k,
                   int r,
                   int s,
                   int p,
                   int q,
                   int padding_h,
                   int padding_w,
                   int stride_h,
                   int stride_w,
                   int dilation_h,
                   int dilation_w,
                   float alpha,
                   float beta) {
  cutlass::conv::Conv2dProblemSize problem_size =
      cutlass::conv::Conv2dProblemSize(n,
                                       h,
                                       w,
                                       c,
                                       k,
                                       r,
                                       s,
                                       p,
                                       q,
                                       padding_h,
                                       padding_w,
                                       stride_h,
                                       stride_w,
                                       dilation_h,
                                       dilation_w,
                                       cutlass::conv::Mode::kCrossCorrelation);

  typename TestKernel::Arguments args{
      problem_size,
      cutlass::make_TensorRef(const_cast<ElementA*>(input), LayoutA::packed(cutlass::conv::implicit_gemm_tensor_a_extent(TestKernel::kConvolutionalOperator, problem_size))),  // nhwc
      cutlass::make_TensorRef(const_cast<ElementB*>(weight), LayoutB::packed(cutlass::conv::implicit_gemm_tensor_b_extent(TestKernel::kConvolutionalOperator, problem_size))),  // nhwc
      cutlass::TensorRef<ElementScaleBias, LayoutC>(const_cast<ElementScaleBias*>(scale), LayoutC::Stride(0)),  // no-use
      cutlass::TensorRef<ElementScaleBias, LayoutC>(const_cast<ElementScaleBias*>(bias), LayoutC::Stride(0)),
      cutlass::make_TensorRef(const_cast<ElementResidual*>(residual), LayoutC::packed(cutlass::conv::implicit_gemm_tensor_c_extent(TestKernel::kConvolutionalOperator, problem_size))),  // npqk
      cutlass::make_TensorRef(output, LayoutC::packed(cutlass::conv::implicit_gemm_tensor_c_extent(TestKernel::kConvolutionalOperator, problem_size))),  // npqk
      typename TestKernel::EpilogueOutputOp::Params(beta)};

  TestKernel op;
  op.initialize(args);
  op.run();
}

using HostConv2dBroadcast = reference::host::Conv2dBroadcast<
    ElementA, ElementB, ElementC, ElementOutput,
    ElementAccumulator, ElementCompute,
    reference::host::ReLU<ElementCompute>,
    std::plus<ElementCompute>>;

void host_kernel(const ElementScaleBias* scale,
                 const ElementA* input,
                 const ElementB* weight,
                 const ElementScaleBias* bias,
                 const ElementC* residual,
                 ElementC* output,
                 int N, int H, int W, int C,
                 int K, int R, int S, int P, int Q,
                 int padding_h, int padding_w,
                 int stride_h,  int stride_w,
                 int dilation_h, int dilation_w,
                 float alpha, float beta) {
  HostConv2dBroadcast::run(
      scale, input, weight, bias, residual, output,
      N, H, W, C, K, R, S, P, Q,
      padding_h, padding_w, stride_h, stride_w,
      dilation_h, dilation_w, alpha, beta);
}

int main() {
  int N = 10, C = 128, H = 72, W = 120;
  int K = 128, R = 3, S = 3;
  int pad_h = 1, pad_w = 1;
  int stride_h = 1, stride_w = 1;
  int dilation_h = 1, dilation_w = 1;

  float alpha = 1.f, beta = 1.f;

  int P = (H + pad_h * 2 - (R - 1) * dilation_h - 1) / stride_h + 1;
  int Q = (W + pad_w * 2 - (S - 1) * dilation_w - 1) / stride_w + 1;

  ElementA* h_input = new ElementA[N * C * H * W];
  ElementB* h_weight = new ElementB[K * C * R * S];
  ElementScaleBias* h_scale = new ElementScaleBias[K];
  ElementScaleBias* h_bias = new ElementScaleBias[K];
  ElementC * h_residual = new ElementC[N * P * Q * K];

  reference::random_initializer<ElementA>::init(h_input, N * C * H * W, -10, 10);
  reference::random_initializer<ElementB>::init(h_weight, K * C * R * S, -10, 10);
  reference::random_initializer<ElementScaleBias>::init(h_scale, K, 0.02f, 0.05f);
  reference::random_initializer<ElementScaleBias>::init(h_bias, K, 0.02f, 2.0f);
  reference::random_initializer<ElementResidual>::init(h_residual, N * P * Q * K, -10, 10);


  ElementOutput* h_output = new ElementOutput[N * P * Q * K];

  ElementA *d_input, *d_weight;
  ElementScaleBias *d_scale, *d_bias;
  ElementResidual *d_residual;
  ElementOutput* d_output;

  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, sizeof(ElementA) * N * C * H * W));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_weight, sizeof(ElementB) * K * C * R * S));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_bias, sizeof(ElementScaleBias) * K));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_scale, sizeof(ElementScaleBias) * K));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_residual, sizeof(ElementResidual) * N * P * Q * K));

  CHECK_CUDA_ERROR(
      cudaMemcpy(d_input, h_input, sizeof(ElementA) * N * C * H * W, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(d_weight, h_weight, sizeof(ElementB) * K * C * R * S, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(d_scale, h_scale, sizeof(ElementScaleBias) * K, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(d_bias, h_bias, sizeof(ElementScaleBias) * K, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(d_residual, h_residual, sizeof(ElementResidual) * N * P * Q * K, cudaMemcpyHostToDevice));


  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, sizeof(ElementOutput) * N * K * P * Q));

  cudaEvent_t start_event;
  cudaEvent_t stop_event;

  CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop_event));

  int warmup_iter = 10;
  int prof_iter = 20;
  ///< warm up
  for (int i = 0; i < warmup_iter; i++) {
    device_kernel(d_scale,
                  d_input,
                  d_weight,
                  d_bias,
                  d_residual,
                  d_output,
                  N,
                  H,
                  W,
                  C,
                  K,
                  R,
                  S,
                  P,
                  Q,
                  pad_h,
                  pad_w,
                  stride_h,
                  stride_w,
                  dilation_h,
                  dilation_w,
                  alpha,
                  beta);
  }
  // cudaDeviceSynchronize();

  cudaEventRecord(start_event);
  for (int i = 0; i < prof_iter; i++) {
    device_kernel(d_scale,
                  d_input,
                  d_weight,
                  d_bias,
                  d_residual,
                  d_output,
                  N,
                  H,
                  W,
                  C,
                  K,
                  R,
                  S,
                  P,
                  Q,
                  pad_h,
                  pad_w,
                  stride_h,
                  stride_w,
                  dilation_h,
                  dilation_w,
                  alpha,
                  beta);
  }

  CHECK_CUDA_ERROR(cudaEventRecord(stop_event));
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  float total_ms;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&total_ms, start_event, stop_event));

  double avg_ms = double(total_ms) / double(prof_iter);

  std::cout << "Run " << prof_iter << " iter, total duration " << total_ms << " ms, avg " << avg_ms
            << " ms." << std::endl;

  ElementOutput* h_result = new ElementOutput[N * P * Q * K];

  cudaMemcpy(h_result, d_output, sizeof(ElementOutput) * N * K * P * Q, cudaMemcpyDeviceToHost);

#ifdef HOST_CHECK

  host_kernel(h_scale,
              h_input,
              h_weight,
              h_bias,
              h_residual,
              h_output,
              N,
              H,
              W,
              C,
              K,
              R,
              S,
              P,
              Q,
              pad_h,
              pad_w,
              stride_h,
              stride_w,
              dilation_h,
              dilation_w,
              alpha, beta);

  for (int i = 0; i < N * P * Q * K; ++i) {
    float diff = fabs(float(h_output[i]) - float(h_result[i]));
    if (diff > 5e-4) {
      std::cout << "row: (" << i << "), cpu: " << float(h_output[i])
                << ", gpu: " << float(h_result[i]) << ", diff: " << diff << std::endl;
    }
  }

  std::cout << "Finished host check." << std::endl;

#endif


  delete[] h_input;
  delete[] h_weight;
  delete[] h_scale;
  delete[] h_bias;
  delete[] h_residual;
  delete[] h_output;
  delete[] h_result;

  CHECK_CUDA_ERROR(cudaFree(d_input));
  CHECK_CUDA_ERROR(cudaFree(d_weight));
  CHECK_CUDA_ERROR(cudaFree(d_bias));
  CHECK_CUDA_ERROR(cudaFree(d_scale));
  CHECK_CUDA_ERROR(cudaFree(d_residual));
  CHECK_CUDA_ERROR(cudaFree(d_output));

  CHECK_CUDA_ERROR(cudaEventDestroy(start_event));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop_event));

  return 0;
}