/*! \file
  \brief Host reference implementation for quantized Conv2d fprop
         with per-channel scale, bias, residual broadcast and activation.
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>

namespace reference {
namespace host {

/// Host-side activation functors (mirroring cutlass::epilogue::thread)

template <typename T>
struct Identity {
  T operator()(T val) const { return val; }
};

template <typename T>
struct ReLU {
  T operator()(T val) const { return val > T(0) ? val : T(0); }
};

template <typename T>
struct Clamp {
  T lower;
  T upper;
  Clamp(T lo = T(0), T hi = T(6)) : lower(lo), upper(hi) {}
  T operator()(T val) const { return std::min(std::max(val, lower), upper); }
};

template <typename T>
struct SiLU {
  T operator()(T val) const { return val / (T(1) + std::exp(-val)); }
};

/// Saturating cast from float to integer output type with round-to-nearest.
template <typename OutputType, typename ComputeType>
struct SaturatingCast {
  static OutputType apply(ComputeType val) {
    using limits = std::numeric_limits<OutputType>;
    int32_t rounded = static_cast<int32_t>(std::roundf(static_cast<float>(val)));
    rounded = std::max(static_cast<int32_t>(limits::min()),
                       std::min(static_cast<int32_t>(limits::max()), rounded));
    return static_cast<OutputType>(rounded);
  }
};

/**
 * Conv2dBroadcast: host reference for quantized 2-D fprop with broadcast epilogue.
 *
 * Formula:
 *   accum = Conv2d(input, weight)                            // int32
 *   result = activation(scale[k] * float(accum)
 *            + beta * float(residual[n][p][q][k])
 *            + bias[k])
 *   output = saturating_cast<ElementOutput>(result)
 *
 * Template params follow the CUTLASS epilogue convention:
 *   ElementA           - input element type    (e.g. int8_t)
 *   ElementB           - weight element type   (e.g. int8_t)
 *   ElementC           - residual element type  (e.g. int8_t)
 *   ElementOutput      - output element type    (e.g. int8_t)
 *   ElementAccumulator - accumulator type       (e.g. int32_t)
 *   ElementCompute     - compute (scale/bias) type (e.g. float)
 *   ElementwiseOp      - activation functor     (e.g. ReLU<float>)
 *   BinaryOp           - binary op to combine scaled-accum+bias with residual
 */
template <
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename ElementOutput,
  typename ElementAccumulator,
  typename ElementCompute,
  typename ElementwiseOp = Identity<ElementCompute>,
  typename BinaryOp      = std::plus<ElementCompute>
>
class Conv2dBroadcast {
public:

  static void run(const ElementCompute* scale,
                  const ElementA*       input,
                  const ElementB*       weight,
                  const ElementCompute* bias,
                  const ElementC*       residual,
                  ElementOutput*        output,
                  int N, int H, int W, int C,
                  int K, int R, int S,
                  int P, int Q,
                  int padding_h,  int padding_w,
                  int stride_h,   int stride_w,
                  int dilation_h, int dilation_w,
                  ElementCompute alpha,
                  ElementCompute beta,
                  ElementwiseOp activation = ElementwiseOp(),
                  BinaryOp      binary_op  = BinaryOp()) {

    for (int nn = 0; nn < N; ++nn) {
      for (int pp = 0; pp < P; ++pp) {
        for (int qq = 0; qq < Q; ++qq) {
          for (int kk = 0; kk < K; ++kk) {

            // --- 1. Convolution accumulation (NHWC cross-correlation) ---
            ElementAccumulator accum = 0;

            for (int rr = 0; rr < R; ++rr) {
              for (int ss = 0; ss < S; ++ss) {
                int ih = pp * stride_h - padding_h + rr * dilation_h;
                int iw = qq * stride_w - padding_w + ss * dilation_w;

                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                  for (int cc = 0; cc < C; ++cc) {
                    int64_t input_idx  = (int64_t)nn * (H * W * C) + ih * (W * C) + iw * C + cc;
                    int64_t weight_idx = (int64_t)kk * (R * S * C) + rr * (S * C) + ss * C + cc;
                    accum += static_cast<ElementAccumulator>(input[input_idx]) *
                             static_cast<ElementAccumulator>(weight[weight_idx]);
                  }
                }
              }
            }

            // --- 2. Epilogue: scale * accum (+ residual) + bias -> activation -> quant ---
            int64_t out_idx = (int64_t)nn * (P * Q * K) + pp * (Q * K) + qq * K + kk;

            ElementCompute scaled = scale[kk] * static_cast<ElementCompute>(accum);
            ElementCompute res_term = beta * static_cast<ElementCompute>(residual[out_idx]);
            ElementCompute result = binary_op(scaled, res_term) + bias[kk];

            result = activation(result);

            output[out_idx] =
                SaturatingCast<ElementOutput, ElementCompute>::apply(result);
          }
        }
      }
    }
  }
};

} // namespace host
} // namespace reference
