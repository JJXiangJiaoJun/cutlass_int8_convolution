#pragma once

#include "cutlass/cutlass.h"


#include "conv/kernel/default_conv2d_fprop_quant.h"
#include "conv/kernel/implicit_gemm_convolution_quant_with_fused_epilogue.h"

#include "epilogue/threadblock/default_epilogue_quant_with_broadcast.h"
#include "epilogue/threadblock/epilogue_quant_with_broadcast.h"

namespace cutlass {
namespace conv {
namespace kernel {

////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementScale,
  typename LayoutScale,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename OperatorClass,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::IteratorAlgorithm IteratorAlgorithm = IteratorAlgorithm::kOptimized,
  conv::StrideSupport StrideSupport = StrideSupport::kStrided,
  int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value,
  int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value
> struct DefaultConv2dFpropQuantPerChannelWithBroadcast;

////////////////////////////////////////////////////////////////////////////////

/**
 * quantized convolution residual. (per channel)
 */
template <
  typename ElementA,
  typename ElementB,
  typename ElementScale,
  typename LayoutScale,
  typename ElementC,
  typename ElementAccumulator,
  typename OperatorClass,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::IteratorAlgorithm IteratorAlgorithm,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFpropQuantPerChannelWithBroadcast<
  ElementA, layout::TensorNHWC,
  ElementB, layout::TensorNHWC,
  ElementScale, LayoutScale,
  ElementC, layout::TensorNHWC,
  ElementAccumulator,
  OperatorClass,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  using ImplicitGemmBase = typename DefaultConv2dFpropQuantPerChannel<
    ElementA, layout::TensorNHWC,
    ElementB, layout::TensorNHWC,
    ElementScale, layout::TensorNHWC,
    ElementC, layout::TensorNHWC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    MathOperatorTag,
    IteratorAlgorithm,
    StrideSupport,
    AlignmentA,
    AlignmentB
  >::Kernel;

  // Define epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueQuantPerChannelWithBroadcastTensorOp<
    typename ImplicitGemmBase::Epilogue::Shape,
    typename ImplicitGemmBase::Epilogue::WarpMmaOperator,
    ImplicitGemmBase::Epilogue::kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionQuantPerChannelWithFusedEpilogue<
    typename ImplicitGemmBase::Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace conv
}  // namespace cutlass
