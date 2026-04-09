#pragma once

#include "cutlass/array.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/conv/threadblock/implicit_gemm_multistage.h"
#include "cutlass/conv/threadblock/implicit_gemm_pipelined.h"
#include "cutlass/conv/threadblock/conv2d_tile_iterator.h"
#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_optimized.h"
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_optimized.h"
#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_fixed_channels.h"
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_fixed_channels.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "conv/kernel/implicit_gemm_convolution_quant.h"
#include "epilogue/threadblock/default_epilogue_quant_tensor_op.h"


namespace cutlass {
namespace conv {
namespace kernel {

template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementScaleBias,
  typename LayoutScaleBias,
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
> struct DefaultConv2dFpropQuantPerChannel;

/////////////////////////////////////////////////////////////////////////////////////////

/**
 * Defines a kernel for Conv2dFprop Per-Channel Quantinization specialization for
 * optimzed IteratorAlgorithm and multistage pipeline (Sm80).
 */
template <
  typename ElementA,
  typename ElementB,
  typename ElementScaleBias,
  typename LayoutScaleBias,
  typename ElementC,
  typename ElementAccumulator,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFpropQuantPerChannel <
  ElementA,
  layout::TensorNHWC,
  ElementB,
  layout::TensorNHWC,
  ElementScaleBias,
  LayoutScaleBias,
  ElementC,
  layout::TensorNHWC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  arch::Sm80,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport::kStrided,
  AlignmentA,
  AlignmentB
> {

  using LayoutA = layout::TensorNHWC;
  using LayoutB = layout::TensorNHWC;
  using LayoutC = layout::TensorNHWC;

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
    ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
    ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
    Stages, MathOperatorTag
  >;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA,
      LayoutA,
      ThreadMapA,
      AccessTypeA
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;

  using IteratorB =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB,
      LayoutB,
      ThreadMapB,
      AccessTypeB
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * AlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmMultistage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    arch::CacheOperation::Always,
    IteratorB,
    SmemIteratorB,
    CacheOpB,
    MmaPolicy,
    Stages
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueQuantPerChannelTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionQuantPerChannel<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

///<===============================================================================================

/**
 * Defines a kernel for Conv2dFprop Per-Channel Quantinization specialization for
 * optimzed IteratorAlgorithm and pipeline (Sm75).
 */
template <
  typename ElementA,
  typename ElementB,
  typename ElementScaleBias,
  typename LayoutScaleBias,
  typename ElementC,
  typename ElementAccumulator,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFpropQuantPerChannel <
  ElementA,
  layout::TensorNHWC,
  ElementB,
  layout::TensorNHWC,
  ElementScaleBias,
  LayoutScaleBias,
  ElementC,
  layout::TensorNHWC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  arch::Sm75,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport::kStrided,
  AlignmentA,
  AlignmentB
> {

  using LayoutA = layout::TensorNHWC;
  using LayoutB = layout::TensorNHWC;
  using LayoutC = layout::TensorNHWC;

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        LayoutA,
        ThreadMapA,
        AccessTypeA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        LayoutB,
        ThreadMapB,
        AccessTypeB
      >
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueQuantPerChannelTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;


  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionQuantPerChannel<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

} // End of namespace kernel
} // End of namespace conv
} // End of namespace cutlass
