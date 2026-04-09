#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/array_subbyte.h"

#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/scale_type.h"


namespace cutlass {
namespace epilogue {
namespace thread {

template<
  typename ElementOutput_,       ///< Data type of final result
  int Count,
  typename ElementAccumulator_,  ///< Data type of AB
  typename ElementCompute_,      ///< Data type of scale and bias
  typename ElementResidual_ = ElementOutput_,
  typename ElementwiseOp_ = Identity<ElementCompute_>,  ///< Activation operation
  typename BinaryOp_ = plus<ElementCompute_>,
  ScaleType::Kind Scale = ScaleType::Default,
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
class LinearCombinationAddBiasElementwiseQuantPerChannel {
public:

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementC = ElementOutput;
  using ElementCompute = ElementCompute_;
  using ElementResidual = ElementResidual_;

  using ElementwiseOp = ElementwiseOp_;
  using BinaryOp = BinaryOp_;

  static const int kCount = Count;
  static const int kElementsPerAccess = kCount;
  static const ScaleType::Kind kScale = Scale;
  static const FloatRoundStyle kRound = Round;

  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentCompute = Array<ElementCompute, kCount>;
  using FragmentScaleBias = Array<ElementCompute, kCount>;
  using FragmentResidual = Array<ElementResidual, kCount>;

  static const bool kIsHeavy = false;

  ///< Relu0: threshold for relu is constantly zero.
  struct Params {
    ElementCompute alpha;
    ElementCompute beta;
    const ElementCompute* alpha_ptr;
    const ElementCompute* beta_ptr;

    CUTLASS_HOST_DEVICE
    Params() : alpha(1), beta(0), alpha_ptr(nullptr), beta_ptr(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(ElementCompute beta) :
        alpha(1), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha,
           ElementCompute beta) :
        alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(const ElementCompute* alpha_ptr) :
        alpha(1), beta(0), alpha_ptr(alpha_ptr), beta_ptr(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(const ElementCompute* alpha_ptr,
           const ElementCompute* beta_ptr) :
      alpha(1), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) { }
  };

private:

  ElementCompute alpha_;
  ElementCompute beta_;

public:

  CUTLASS_HOST_DEVICE
  LinearCombinationAddBiasElementwiseQuantPerChannel(const Params& params) {
    alpha_ = params.alpha_ptr ? *params.alpha_ptr : params.alpha;
    beta_ = params.beta_ptr ? *params.beta_ptr : params.beta;
  }

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    ///< Output = scale_per_channel * accum + beta * residual +  bias
    if (Scale == ScaleType::NoBetaScaling) return true;
    ///< Output = Scale_per_channel * accum + beta * residual
    if (Scale == ScaleType::OnlyAlphaPerChannelScaling) return false;
    ///< Output = scale_per_channel * accum + beta * output
    // return beta_ != ElementCompute(0);

    ///< default is true
    return true;
  }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition != 0) {
      beta_ = ElementCompute(1);
    }
  }

  ///< Scale == OnlyAlphaPerChannelScaling.
  ///< Output = scale * accum + beta * residual.
  ///< \todo: (jhd) compatible with output type half.
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(const FragmentCompute& scales,
                            const FragmentAccumulator& accum,
                            const FragmentResidual& residual) const {
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, kRound> accum_convert;
    FragmentCompute converted_accum = accum_convert(accum);

    FragmentCompute float_res;

    ElementwiseOp elementwise_op;
    BinaryOp binary_op;

    ///< Only scale no bias.
    // float_res = op(scales, converted_accum);

    NumericArrayConverter<ElementCompute, ElementResidual, kCount, kRound> residual_converter;
    FragmentCompute float_residual = residual_converter(residual);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      ///< Forbid to use {alpha, beta} scalar coefficients.
      ElementCompute res = binary_op(scales[i] * converted_accum[i], beta_ * float_residual[i]);
      float_res[i] = elementwise_op(res);
    }


    NumericConverter<ElementOutput, ElementCompute, kRound> result_convert;

    FragmentOutput output;

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < kCount; ++i) {
      output[i] = result_convert(float_res[i]);
    }

    return output;
  }

  ///< Scale == NoBetaScaling.
  ///< Output = scale * accum + bias
  ///< \todo: (jhd) compatible with output type half.
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(const FragmentCompute& scales,
                            const FragmentAccumulator& accum,
                            const FragmentResidual& residual,
                            const FragmentCompute& bias) const {

    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, kRound> accum_convert;
    FragmentCompute converted_accum = accum_convert(accum);

    FragmentCompute float_res;

    ElementwiseOp elementwise_op;
    BinaryOp binary_op;

    ///< Only scale no bias.
    // float_res = op(scales, converted_accum);

    NumericArrayConverter<ElementCompute, ElementResidual, kCount, kRound> residual_converter;
    FragmentCompute float_residual = residual_converter(residual);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      ///< Forbid to use {alpha, beta} scalar coefficients.
      ElementCompute res = binary_op(scales[i] * converted_accum[i], beta_ * float_residual[i]) + bias[i];
      float_res[i] = elementwise_op(res);
    }

    NumericConverter<ElementOutput, ElementCompute, kRound> result_convert;

    FragmentOutput output;

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < kCount; ++i) {
      output[i] = result_convert(float_res[i]);
    }

    return output;
  }
};


///<===============================================================================================

} // End of namespace thread
} // End of namespace epilogue
} // End of namespace cutlass
