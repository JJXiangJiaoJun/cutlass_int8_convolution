#pragma once

#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/epilogue/threadblock/epilogue_base.h"

namespace cutlass {
namespace epilogue {
namespace threadblock {

/**
 * Epilogue Quantinization Per-Channel operator.
 * Do not support streamK at present.
 */
template <
  typename Shape_,                          ///< Shape of threadblock tile (concept: GemmShape)
  typename WarpMmaOperator_,                ///< Warp-level MMA operator (concept: gemm::warp::MmaTensorOp)
  int PartitionsK,                          ///< Number of partitions of the K dimension
  typename OutputTileIterator_,             ///< Tile iterator reading and writing output tensors
  typename AccumulatorFragmentIterator_,    ///< Fragment iterator selecting accumulators
  typename ScaleBiasFragmentIterator_,      ///< Fragment iterator selecting scale and bias
  /// Residual tensor tile iterator
  typename TensorTileIterator_,
  typename WarpTileIterator_,               ///< Warp-scoped tile iterator writing accumulators to SMEM
  typename SharedLoadIterator_,             ///< Threadblock-scoped tile iterator loading from SMEM
  typename OutputOp_,                       ///< Output operator
  typename Padding_,                        ///< Padding added to SMEM allocation to avoid bank conflicts (concept: MatrixShape)
  int FragmentsPerPartition = 1,            ///< Used to coarsten the epilogue granularity
  int IterationsUnroll = (!IsEpilogueFunctorHeavy<OutputOp_>::value) ///< Used to reduce binary size when epilogue op is large
>
class EpilogueQuantPerChannelWithBroadcast :
  public EpilogueBase<
    Shape_,
    typename WarpMmaOperator_::Shape,
    PartitionsK,
    AccumulatorFragmentIterator_,
    WarpTileIterator_,
    Padding_,
    FragmentsPerPartition> {
public:

  using Base = EpilogueBase<
    Shape_,
    typename WarpMmaOperator_::Shape,
    PartitionsK,
    AccumulatorFragmentIterator_,
    WarpTileIterator_,
    Padding_,
    FragmentsPerPartition>;

  using Shape = Shape_;
  using WarpMmaOperator = WarpMmaOperator_;
  static int const kPartitionsK = PartitionsK;
  using OutputTileIterator = OutputTileIterator_;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  ///< Scale && bias
  using ScaleBiasTileIterator = ScaleBiasFragmentIterator_;
  using ScaleBiasFragmentIterator = ScaleBiasFragmentIterator_;
  using TensorTileIterator = TensorTileIterator_;
  using WarpTileIterator = WarpTileIterator_;
  using SharedLoadIterator = SharedLoadIterator_;
  using OutputOp = OutputOp_;
  using Padding = Padding_;
  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;

  /// Number of warps per block
  using WarpCount = typename Base::WarpCount;

  /// Number of threads per block
  static int const kBlockThreads = 32 * WarpCount::kCount;

  /// Per-thread accumulator tile type
  using AccumulatorTile = typename Base::AccumulatorTile;

  /// Numerical accumulation element type
  using ElementAccumulator = typename WarpMmaOperator::ElementC;

  /// Fragment type used by the accumulator tile's fragment iterator
  using AccumulatorFragment = typename AccumulatorFragmentIterator::Fragment;

  //using ElementCompute = float;
  using ElementCompute = typename ScaleBiasFragmentIterator::Element;

  /// Output element
  using ElementOutput = typename OutputTileIterator::Element;

  /// Output access size
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  /// Tensor reference to destination tensor
  using TensorRef = typename OutputTileIterator::TensorRef;

  /// Tensor reference to sync tensor
  using SyncTensorRef = typename cutlass::TensorRef<int, cutlass::layout::PackedVectorLayout>;

  /// Const tensor reference to source tensor
  using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

  /// Vector type used by the global output iterator
  using OutputAccessType = Array<
    typename OutputTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  using ComputeAccessType = Array<
    ElementCompute, OutputTileIterator::kElementsPerAccess>;

  using ResidualAccessType = Array<typename TensorTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  /// Vector type used by the shared output iterator
  using AccumulatorAccessType = Array<typename WarpTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  static int constexpr kSmemTiles = Base::kFragmentsPerIteration > 1 ? Base::kFragmentsPerIteration : kPartitionsK;

  static int constexpr kSmemPointerOffset = Base::SharedStorage::StorageShape::kCount / kSmemTiles;

public:


  static_assert(SharedLoadIterator::Fragment::kElements == OutputTileIterator::Fragment::kElements,
    "Mismatch between shared load iterator and output tile iterator.");

  static_assert(OutputTileIterator::kElementsPerAccess, "OutputTileIterator::kElementsPerAccess must not be zero.");

  static_assert(!(OutputTileIterator::Fragment::kElements % OutputTileIterator::kElementsPerAccess),
    "Divisibility");

  static_assert(kPartitionsK == 1 || Base::kFragmentsPerIteration == 1, "One of these must be exactly 1.");

public:


  /// Aspect for when epilogue source is not needed
  struct SourceAspectNotNeeded
  {
    /// Constructor
    CUTLASS_DEVICE
    SourceAspectNotNeeded()
    {}

    // No-op
    CUTLASS_DEVICE
    void load() { }

    /// Invoke the output functor over each vector of output
    CUTLASS_DEVICE
    void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment,
      OutputOp const &output_op,
      const typename ScaleBiasFragmentIterator::Fragment& scale_fragment,
      typename SharedLoadIterator::Fragment const &aligned_accum_fragment,
      typename TensorTileIterator::Fragment& residual_tensor_fragment
      )
    {
      OutputAccessType *output_frag_ptr =
        reinterpret_cast<OutputAccessType *>(&output_fragment);

      ResidualAccessType* residual_tensor_frag_ptr = reinterpret_cast<ResidualAccessType*>(&residual_tensor_fragment);

      ComputeAccessType const *scale_frag_ptr =
        reinterpret_cast<const ComputeAccessType*>(&scale_fragment);

      AccumulatorAccessType const *compute_frag_ptr =
        reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment);

      int const kOutputOpIterations =
        OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kOutputOpIterations; ++i)
      {
        // Call the output operator
        output_frag_ptr[i] = output_op(scale_frag_ptr[i], compute_frag_ptr[i], residual_tensor_frag_ptr[i]);
      }
    }
  };

  /// Aspect for when epilogue source is needed
  struct SourceAspectNeeded
  {
    ScaleBiasFragmentIterator bias_iterator;

    typename ScaleBiasFragmentIterator::Fragment bias_fragment;

    /// Invoke the output functor over each vector of output
    CUTLASS_DEVICE
    static void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment,
      OutputOp const &output_op,
      const typename ScaleBiasFragmentIterator::Fragment& scale_fragment,
      typename SharedLoadIterator::Fragment const &aligned_accum_fragment,
      typename TensorTileIterator::Fragment& residual_tensor_fragment,
      const typename ScaleBiasFragmentIterator::Fragment& bias_fragment)
    {
      OutputAccessType *output_frag_ptr =
        reinterpret_cast<OutputAccessType *>(&output_fragment);

      ResidualAccessType* residual_tensor_frag_ptr = reinterpret_cast<ResidualAccessType*>(&residual_tensor_fragment);

      AccumulatorAccessType const *compute_frag_ptr =
        reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment);

      ComputeAccessType const *scale_frag_ptr =
        reinterpret_cast<const ComputeAccessType*>(&scale_fragment);

      ComputeAccessType const *bias_frag_ptr =
        reinterpret_cast<const ComputeAccessType*>(&bias_fragment);

      int const kOutputOpIterations =
        OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kOutputOpIterations; ++i)
      {
        // Call the output operator
        output_frag_ptr[i] = output_op(scale_frag_ptr[i], compute_frag_ptr[i], residual_tensor_frag_ptr[i], bias_frag_ptr[i]);
      }
    }

    /// Constructor
    CUTLASS_DEVICE
    SourceAspectNeeded(ScaleBiasFragmentIterator bias_iterator) :
      bias_iterator(bias_iterator)
    {
      bias_fragment.clear();
    }

    // Load addend source fragment from global memory
    CUTLASS_DEVICE
    void load() {
      bias_iterator.load(bias_fragment);
      ++bias_iterator;
    }

    /// Invoke the output functor over each vector of output
    CUTLASS_DEVICE
    void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment,
      OutputOp const &output_op,
      const typename ScaleBiasFragmentIterator::Fragment& scale_fragment,
      typename SharedLoadIterator::Fragment const &aligned_accum_fragment,
      typename TensorTileIterator::Fragment& residual_tensor_fragment)
    {
      apply_output_operator(output_fragment, output_op, scale_fragment, aligned_accum_fragment, residual_tensor_fragment, bias_fragment);
    }
  };

private:

  /// Loads fragment from shared memory aligned with output tensor
  SharedLoadIterator shared_load_iterator_;

  /// Thread index in the threadblock
  int thread_idx;

  /// Warp index in the threadblock
  int warp_idx;

public:

  /// Constructor
  CUTLASS_DEVICE
  EpilogueQuantPerChannelWithBroadcast(
      typename Base::SharedStorage &shared_storage,   ///< Shared storage object
      int thread_idx,                                 ///< ID of a thread within the threadblock
      int warp_idx,                                   ///< ID of warp within threadblock
      int lane_idx)                                   ///< Id of thread within warp
  :
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      shared_load_iterator_(shared_storage.reference(), thread_idx),
      thread_idx(thread_idx),
      warp_idx(warp_idx) { }


  /// Streams the result to global memory
  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    ScaleBiasFragmentIterator scale_iterator,       ///< Fragment iterator for scale
    AccumulatorTile const &accumulators,            ///< Complete warp-level accumulator tile
    TensorTileIterator residual_tensor_iterator,  ///< Fragment iterator for residual tensor
    ScaleBiasFragmentIterator bias_iterator)        ///< Fragment iterator for bias
  {
    if (output_op.is_source_needed())
    {
      operator()(output_op, destination_iterator, scale_iterator, accumulators, residual_tensor_iterator, SourceAspectNeeded(bias_iterator));
    }
    else
    {
      operator()(output_op, destination_iterator, scale_iterator, accumulators, residual_tensor_iterator, SourceAspectNotNeeded());
    }
  }


  template<typename SourceAspect>
  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    ScaleBiasFragmentIterator scale_iterator,       ///< Fragment iterator for scale
    AccumulatorTile const &accumulators,            ///< Complete warp-level accumulator tile
    TensorTileIterator residual_tensor_iterator,  ///< Fragment iterator for residual tensor
    SourceAspect source)                             ///< Fragment iterator for bias
  {
    typename ScaleBiasFragmentIterator::Fragment scale_fragment;
    scale_fragment.clear();

    source.load();

    scale_iterator.load(scale_fragment);

    // Iterator over warp-level accumulator fragment
    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    //
    // Double-buffered residual load pipeline
    //
    static int const kTotalIterations = OutputTileIterator::kIterations;

    typename TensorTileIterator::Fragment residual_frag[2];
    int cur_buf = 0;

    // Prologue: prefetch first residual tile
    residual_tensor_iterator.load(residual_frag[cur_buf]);
    ++residual_tensor_iterator;

    //
    // Iterate over accumulator tile
    //

    #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations / Base::kFragmentsPerIteration : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; iter += Base::kFragmentsPerIteration)
    {

      //
      // Convert and store accumulator fragment to shared memory
      //

      __syncthreads();

      CUTLASS_PRAGMA_UNROLL
      for (int p = 0; p < Base::kFragmentsPerIteration; ++p)
      {
        typename AccumulatorFragmentIterator::Fragment accum_fragment;

        accum_fragment_iterator.load(accum_fragment);
        ++accum_fragment_iterator;

        this->warp_tile_iterator_.store(accum_fragment);

        if (p < Base::kFragmentsPerIteration - 1) {
          this->warp_tile_iterator_.add_pointer_offset(kSmemPointerOffset);
        }
      }

      if (Base::kFragmentsPerIteration > 1) {
        this->warp_tile_iterator_.add_pointer_offset(kSmemPointerOffset * (1 - Base::kFragmentsPerIteration));
      }


      //
      // Load fragments from shared memory, compute epilogue, store output
      //

      __syncthreads();

      CUTLASS_PRAGMA_UNROLL
      for (int p = 0; p < Base::kFragmentsPerIteration; ++p)
      {
        typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];

        shared_load_iterator_.load(aligned_accum_fragment[0]);

        if (p < Base::kFragmentsPerIteration - 1)
        {
          shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
        }
        else if (kPartitionsK > 1)
        {
          plus <typename SharedLoadIterator::Fragment> add_fragments;

          CUTLASS_PRAGMA_UNROLL
          for ( int i = 1; i < kPartitionsK; ++i) {
            shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
            shared_load_iterator_.load(aligned_accum_fragment[i]);
            aligned_accum_fragment[0] = add_fragments(aligned_accum_fragment[0], aligned_accum_fragment[i]);
          }

          shared_load_iterator_.add_pointer_offset((1 - kPartitionsK) * kSmemPointerOffset);
        }

        //
        // Pipeline stage 1: Prefetch next residual (global load, high latency)
        //   Overlaps with compute + store below, and syncthreads + accum_store
        //   + syncthreads + smem_load of the next outer iteration.
        //

        if (iter + p + 1 < kTotalIterations) {
          residual_tensor_iterator.load(residual_frag[cur_buf ^ 1]);
          ++residual_tensor_iterator;
        }

        //
        // Pipeline stage 2: Compute with current residual (prefetched earlier)
        //

        typename OutputTileIterator::Fragment output_fragment;

        source.apply_output_operator(output_fragment, output_op, scale_fragment,
                                     aligned_accum_fragment[0], residual_frag[cur_buf]);

        //
        // Pipeline stage 3: Store the final result
        //

        destination_iterator.store(output_fragment);
        ++destination_iterator;

        //
        // Advance pipeline: swap buffer index
        //

        cur_buf ^= 1;
      }

      if (Base::kFragmentsPerIteration > 1) {
        shared_load_iterator_.add_pointer_offset(kSmemPointerOffset * (1 - Base::kFragmentsPerIteration));
      }
    }
  }

};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass