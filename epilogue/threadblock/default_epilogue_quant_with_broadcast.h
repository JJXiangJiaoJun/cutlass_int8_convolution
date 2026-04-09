#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"

#include "cutlass/gemm/gemm.h"

#include "epilogue/threadblock/default_epilogue_quant_tensor_op.h"
#include "epilogue/threadblock/epilogue_quant_with_broadcast.h"

#include "cutlass/layout/permute.h"

namespace cutlass {
namespace epilogue {
namespace threadblock {
/**
 * Defines sensible defaults for epilogues for TensorOps. (Quant per channel)
 */
template <
  typename Shape,
  typename WarpMmaTensorOp,
  int PartitionsK,
  typename OutputOp,
  int ElementsPerAccess,
  bool ScatterD = false,
  typename PermuteDLayout = layout::NoPermute
>
struct DefaultEpilogueQuantPerChannelWithBroadcastTensorOp {

  /// Use defaults related to the existing epilogue. (Quant epilogue per channel tensor op)
  using Base = DefaultEpilogueQuantPerChannelTensorOp<
    Shape,
    WarpMmaTensorOp,
    PartitionsK,
    OutputOp,
    ElementsPerAccess
  >;

  using ElementResidual = typename OutputOp::ElementResidual;


  //
  // Additional tensor tile iterator - stores t = Elementwise(z)
  //
  using TensorTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    typename Base::OutputTileThreadMap,
    ElementResidual
  >;

  /// Define the epilogue
  using Epilogue = EpilogueQuantPerChannelWithBroadcast<
    Shape,
    WarpMmaTensorOp,
    PartitionsK,
    typename Base::OutputTileIterator,
    typename Base::AccumulatorFragmentIterator,
    typename Base::ScaleBiasFragmentIterator,
    TensorTileIterator,
    typename Base::WarpTileIterator,
    typename Base::SharedLoadIterator,
    OutputOp,
    typename Base::Padding,
    Base::kFragmentsPerIteration
  >;
};


} // namespace threadblock
} // namespace epilogue
} // namespace cutlass