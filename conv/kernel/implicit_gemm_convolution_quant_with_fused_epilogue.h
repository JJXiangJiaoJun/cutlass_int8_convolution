#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/semaphore.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/epilogue/threadblock/output_iterator_parameter.h"

namespace cutlass {
namespace conv {
namespace kernel {
/**
 * quantized implicit-gemm convolution with extra epilogue kernel. (per channel)
 */
template<
  ///< Threadblock-level GeMM
  typename Mma_,
  ///< Epilogue with broadcast. (Concept: )
  typename Epilogue_,
  ///< Threadblock swizzling function. (Concept: gemm::threadblock::GemmThreadblockSwizzlor<>)
  typename ThreadblockSwizzle_,
  ///< Convolutional operator. (Fprop, Dgrad, Wgrad)
  conv::Operator ConvOperator,
  ///< Convolutional operator on 2D or 3D problem.
  typename ConvProblemSize_ = Conv2dProblemSize,
  conv::GroupMode GroupMode_ = conv::GroupMode::kNone    ///! Group mode
>
struct ImplicitGemmConvolutionQuantPerChannelWithFusedEpilogue {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static const Operator kConvolutionalOperator = ConvOperator;

  ///< Feature map tensor description.
  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  ///< Filter tensor description.
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  ///< Output data description.
  using ElementC = typename EpilogueOutputOp::ElementOutput;
  /// Set output tensor C layout
  ///< \attention LayoutA must equal to LayoutC.
  using LayoutC = LayoutA;

  ///< Scale/Bias layout
  using LayoutScaleBias = LayoutC;

  ///< Residual element and layout (same as output)
  using ElementResidual = typename Epilogue::TensorTileIterator::Element;
  using LayoutResidual = LayoutC;

  using ElementAccumulator = typename EpilogueOutputOp::ElementAccumulator;
  ///< Scale and bias data type.
  using ElementCompute = typename EpilogueOutputOp::ElementCompute;

  using WarpMmaOperator = typename Mma::Policy::Operator;

  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
  using MathOperator = typename ArchMmaOperator::Operator;

  using OperatorClass = typename WarpMmaOperator::OperatorClass;
  using ArchTag = typename WarpMmaOperator::ArchTag;

  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename WarpMmaOperator::Shape;
  using InstructionShape = typename ArchMmaOperator::Shape;

  static int const kStages = Mma::kStages;
  static int const kWarpGemmIterations = Mma::Base::kWarpGemmIterations;
  ///< if kWarpGemmIterations==1, it means Mma is type ImplicitGemmMultistageFewChannels,
  ///< ThreadBlockShapeK == WarpShapeK == InstructionShapeK
  static IteratorAlgorithm const kIteratorAlgorithm =
      kWarpGemmIterations > 1 ? Mma::IteratorA::kIteratorAlgorithm : IteratorAlgorithm::kFewChannels;
  static StrideSupport const kStrideSupport = Mma::IteratorA::kStrideSupport;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using TensorRefA = typename Mma::IteratorA::TensorRef;
  using TensorRefB = typename Mma::IteratorB::TensorRef;
  using TensorRefScaleBias = cutlass::TensorRef<ElementCompute, LayoutScaleBias>;
  using TensorRefResidual = cutlass::TensorRef<ElementResidual, LayoutResidual>;
  using TensorRefD = cutlass::TensorRef<ElementC, LayoutC>;

  /// Check iterator A and B convolution dimension are the same and
  // set device::ImplicitGemmConvolution::kConvDim
  static_assert(Mma::IteratorA::kConvDim == Mma::IteratorB::kConvDim,
    "Convolution on different A/B dimensions is not supported");
  static int const kConvDim = Mma::IteratorA::kConvDim;

  /// Conv dimension and problem size structure (Conv2d or Conv3d)
  using ConvProblemSize = ConvProblemSize_;

  static const conv::GroupMode kGroupMode = GroupMode_;

  /// Wgrad C stride idx for implicit gemm algorithm
  // Conv2d row-major matrix C (KxRSC)
  // Conv3d row-major matrix C (KxTRSC)
  static int const kWgradCStrideIdx = 2;

  /// This chooses the appropriate stride element of the C tensor.
  static int const kTensorCStrideIdx = 0;

  //
  // ConvOutputIteratorParameter for output, scale/bias, and residual
  //
  using ConvOutputIteratorParameter = epilogue::threadblock::ConvOutputIteratorParameter<
    LayoutC,
    typename Epilogue::OutputTileIterator::Layout,
    TensorRefD,
    ConvOperator,
    ConvProblemSize
    >;

  using ConvScaleBiasIteratorParameter = epilogue::threadblock::ConvOutputIteratorParameter<
    LayoutC,
    typename Epilogue::ScaleBiasTileIterator::Layout,
    TensorRefScaleBias,
    ConvOperator,
    ConvProblemSize
    >;

  using ConvResidualIteratorParameter = epilogue::threadblock::ConvOutputIteratorParameter<
    LayoutResidual,
    typename Epilogue::TensorTileIterator::Layout,
    TensorRefResidual,
    ConvOperator,
    ConvProblemSize
    >;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    ConvProblemSize problem_size;
    TensorRefA ref_A;
    TensorRefB ref_B;
    TensorRefScaleBias ref_scale;
    TensorRefScaleBias ref_bias;
    TensorRefResidual ref_residual;
    TensorRefD ref_D;
    typename EpilogueOutputOp::Params output_op;
    SplitKMode split_k_mode;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }

    CUTLASS_HOST_DEVICE
    Arguments(
      ConvProblemSize const & problem_size
    ):
      problem_size(problem_size) { }

    CUTLASS_HOST_DEVICE
    Arguments(
      ConvProblemSize const & problem_size,
      TensorRefA const & ref_A,
      TensorRefB const & ref_B,
      TensorRefScaleBias const & ref_scale,
      TensorRefScaleBias const & ref_bias,
      TensorRefResidual const & ref_residual,
      TensorRefD const & ref_D,
      typename EpilogueOutputOp::Params const & output_op,
      SplitKMode const & split_k_mode = SplitKMode::kSerial
    ):
      problem_size(problem_size),
      ref_A(ref_A),
      ref_B(ref_B),
      ref_scale(ref_scale),
      ref_bias(ref_bias),
      ref_residual(ref_residual),
      ref_D(ref_D),
      output_op(output_op),
      split_k_mode(split_k_mode)
    {
    }

  };

  /// Parameters structure
  struct Params {
    ConvProblemSize problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    gemm::GemmCoord implicit_gemm_problem_size;
    int swizzle_log_tile;

    int gemm_k_iterations;
    int gemm_k_iterations_per_channel;
    typename Mma::IteratorA::Params iterator_A;
    typename Mma::IteratorA::Element const *ptr_A;
    typename Mma::IteratorB::Params iterator_B;
    typename Mma::IteratorB::Element const *ptr_B;
    typename Epilogue::ScaleBiasTileIterator::Params iterator_scale;
    ElementCompute* ptr_scale;
    typename Epilogue::ScaleBiasTileIterator::Params iterator_bias;
    ElementCompute* ptr_bias;
    typename Epilogue::TensorTileIterator::Params iterator_residual;
    ElementResidual *ptr_residual;
    typename Epilogue::OutputTileIterator::Params iterator_D;
    typename Epilogue::OutputTileIterator::Element *ptr_D;
    typename EpilogueOutputOp::Params output_op;
    int *semaphore;
    SplitKMode split_k_mode;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): swizzle_log_tile(0), gemm_k_iterations(0) { }

    ///
    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      int *semaphore = nullptr
    ):
      problem_size(args.problem_size),
      implicit_gemm_problem_size(cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size)),
      iterator_A(Mma::IteratorA::getParams(args.problem_size, args.ref_A.layout())),
      ptr_A(args.ref_A.data()),
      iterator_B(args.problem_size, args.ref_B.layout()),
      ptr_B(args.ref_B.data()),
      iterator_scale(ConvScaleBiasIteratorParameter::layout(args.ref_scale)),
      ptr_scale(args.ref_scale.data()),
      iterator_bias(ConvScaleBiasIteratorParameter::layout(args.ref_bias)),
      ptr_bias(args.ref_bias.data()),
      iterator_residual(ConvResidualIteratorParameter::layout(args.ref_residual)),
      ptr_residual(const_cast<ElementResidual*>(args.ref_residual.data())),
      iterator_D(ConvOutputIteratorParameter::layout(args.ref_D)),
      ptr_D(args.ref_D.data()),
      output_op(args.output_op),
      semaphore(semaphore),
      split_k_mode(args.split_k_mode)
    {
      gemm_k_iterations = implicit_gemm_k_iterations(
        kConvolutionalOperator,
        ThreadblockShape::kK,
        args.problem_size,
        kIteratorAlgorithm,
        kGroupMode,
        ThreadblockShape::kN);

      gemm_k_iterations_per_channel = implicit_gemm_k_iterations_per_channel(
          kConvolutionalOperator, args.problem_size, kIteratorAlgorithm);

      ThreadblockSwizzle threadblock_swizzle;

      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        implicit_gemm_problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.problem_size.split_k_slices);

      swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape);
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

public:

  CUTLASS_HOST_DEVICE
  ImplicitGemmConvolutionQuantPerChannelWithFusedEpilogue() { }

  /// Executes one ImplicitGEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_idx.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_idx.n()) {

      return;
    }

    // Compute position within threadblock
    int thread_idx = threadIdx.x;
    int iterator_A_column_offset = threadblock_tile_idx.k() * Mma::Shape::kK;
    if (kGroupMode != GroupMode::kNone) {
      if (kGroupMode != GroupMode::kDepthwise) {
        int k_per_group = params.problem_size.K / params.problem_size.groups;
        int group_idx = threadblock_tile_idx.n() * Mma::Shape::kN / k_per_group;
        int channels_per_group = params.problem_size.C / params.problem_size.groups;
        iterator_A_column_offset += group_idx * channels_per_group;
      } else {
        iterator_A_column_offset += threadblock_tile_idx.n() * Mma::Shape::kN;
      }
    }

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.iterator_A,
      params.problem_size,
      params.ptr_A,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.m() * Mma::Shape::kM,
        iterator_A_column_offset
      )
    );

    typename Mma::IteratorB iterator_B(
      params.iterator_B,
      params.problem_size,
      params.ptr_B,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.k() * Mma::Shape::kK,
        threadblock_tile_idx.n() * Mma::Shape::kN
      )
    );

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // Compute threadblock-scoped matrix multiply-add
    mma(params.gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators, params.gemm_k_iterations_per_channel);

    //
    // Epilogue
    //

    EpilogueOutputOp output_op(params.output_op);

    // Construct the semaphore.
    int block_idx = threadblock_tile_idx.m() + threadblock_tile_idx.n() * params.grid_tiled_shape.m();

    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    // Compute logical position within grid
    threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) {

      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      output_op.set_k_partition(threadblock_tile_idx.k(), params.grid_tiled_shape.k());
    }

    MatrixCoord threadblock_offset(
      threadblock_tile_idx.m() * Mma::Shape::kM,
      threadblock_tile_idx.n() * Mma::Shape::kN
    );

    // Tile iterator writing to destination tensor
    typename Epilogue::OutputTileIterator iterator_D(
      params.iterator_D,
      params.ptr_D,
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset
    );

    // Tile iterator reading from scale tensor (per-channel)
    typename Epilogue::ScaleBiasTileIterator iterator_scale(
      params.iterator_scale,
      params.ptr_scale,
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset
    );

    // Tile iterator reading from bias tensor (per-channel)
    typename Epilogue::ScaleBiasTileIterator iterator_bias(
      params.iterator_bias,
      params.ptr_bias,
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset
    );

    // Tile iterator reading from residual tensor
    typename Epilogue::TensorTileIterator iterator_residual(
      params.iterator_residual,
      params.ptr_residual,
      ConvResidualIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset
    );

    // Construct the epilogue
    Epilogue epilogue(
      shared_storage.epilogue,
      thread_idx,
      warp_idx,
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) {

      semaphore.wait(threadblock_tile_idx.k());

    }
    // Each split-k-slice writes to a unique tensor location
    else if (params.split_k_mode == SplitKMode::kParallel) {
      iterator_D.add_pointer_offset(threadblock_tile_idx.k() *
        cutlass::conv::implicit_gemm_tensor_c_size(ConvOperator, params.problem_size));
    }

    // Run efficient epilogue with residual
    epilogue(output_op, iterator_D, iterator_scale, accumulators, iterator_residual, iterator_bias);

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass
