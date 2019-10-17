#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import numpy as np


from snpe.converters.tensorflow.layers.fullyconnected import (
    FullyConnectedLayerResolver,
    FullyConnectedLayerBuilder
)
from snpe.converters.tensorflow.layers.convolution import (
    ConvolutionLayerResolver,
    ConvolutionLayerBuilder,
    GroupedConvolutionLayerResolver,
    DilatedConvolutionLayerResolver,
    DepthwiseConvolutionLayerResolver,
    DilatedDepthwiseConvolutionLayerResolver
)
from snpe.converters.tensorflow.layers.concat import (
    ConcatLayerResolver,
    ConcatLayerBuilder
)
from snpe.converters.tensorflow.layers.relu import (
    ReluLayerResolver,
    ReluLayerBuilder
)
from snpe.converters.tensorflow.layers.relu_min_max import (
    ReluMinMaxLayerResolver,
    ReluMinMaxLayerBuilder
)
from snpe.converters.tensorflow.layers.relu6 import (
    Relu6LayerResolver
)
from snpe.converters.tensorflow.layers.sigmoid import (
    SigmoidLayerResolver,
    SigmoidLayerBuilder
)
from snpe.converters.tensorflow.layers.tanh import (
    TanhLayerResolver,
    TanhLayerBuilder
)
from snpe.converters.tensorflow.layers.softmax import (
    SoftmaxLayerResolver,
    SoftmaxLayerBuilder
)
from snpe.converters.tensorflow.layers.lrn import (
    LrnLayerResolver,
    LrnLayerBuilder
)
from snpe.converters.tensorflow.layers.embedding import (
    EmbeddingLayerResolver,
    EmbeddingLayerBuilder
)
from snpe.converters.tensorflow.layers.deconvolution import (
    DeConvolutionOptimizedLayerResolver,
    DeConvolutionLayerBuilder
)
from snpe.converters.tensorflow.layers.batchnorm import (
    BatchNormLayerResolver,
    UnscaledBatchNormLayerResolver,
    ScaledBatchNormLayerResolver,
    BatchNormWithGlobalNormLayerResolver,
    BatchNormLayerBuilder,
    FusedBatchNormNormLayerResolver,
    GenericBatchNormLayerResolver
)

from snpe.converters.tensorflow.layers.instance_norm import (
    InstanceNormRMSLayerBuilder,
    InstanceNormRMSLayerResolver,
    InstanceNormLayerBuilder,
    InstanceNormLayerResolver
)

from snpe.converters.tensorflow.layers.pooling import (
    AvgPoolingLayerResolver,
    MaxPoolingLayerResolver,
    PoolingLayerBuilder
)
from snpe.converters.tensorflow.layers.eltwise import (
    EltWiseSumLayerResolver,
    EltWiseSumLayerBuilder,
    EltWiseSubLayerResolver,
    EltWiseSubLayerBuilder,
    EltWiseMulLayerResolver,
    EltWiseMulLayerBuilder,
    EltWiseMaxLayerResolver,
    EltWiseMaxLayerBuilder,
    EltWiseDivLayerResolver,
    EltWiseDivLayerBuilder
)

from snpe.converters.tensorflow.layers.add_n import (
    AddNLayerResolver,
    AddNLayerBuilder
)

from snpe.converters.tensorflow.layers.slice import (
    SliceLayerResolver,
    SliceLayerBuilder
)

from snpe.converters.tensorflow.layers.prelu import (
    PReLuLayerResolver,
    PReLuLayerBuilder
)

from snpe.converters.tensorflow.layers.reshape import (
    ReshapeLayerResolver,
    ReshapeLayerBuilder
)

from snpe.converters.tensorflow.layers.resize import (
    ResizeNearestNeighborLayerResolver,
    ResizeBilinearLayerResolver,
    ResizeLayerBuilder
)

from snpe.converters.tensorflow.layers.lstm import (
    LstmLayerResolver,
    LstmLayerBuilder
)
from snpe.converters.tensorflow.layers.ignored_patterns import (
    IgnoredLayersResolver,
    IgnoredLayersBuilder
)

from snpe.converters.tensorflow.layers.fill import (
    FillLayerResolver,
    FillLayerBuilder
)

from snpe.converters.tensorflow.layers.ssd import (
    SSDDecoderResolver,
    SSDDecoderLayersBuilder,
    SSDNmsResolver,
    SSDNmsLayersBuilder,
    SSDAnchorGeneratorResolver,
)

from snpe.converters.tensorflow.layers.crop import (
    CropLayerResolver,
    CropLayerBuilder
)

from snpe.converters.tensorflow.layers.constant import (
    ConstantLayerResolver,
    ConstantLayerBuilder
)

from snpe.converters.tensorflow.layers.pad import (
    PadLayerResolver,
    PadLayerBuilder
)

from snpe.converters.tensorflow.layers.strided_slice import (
    StridedSliceLayerResolver,
    StridedSliceLayerBuilder
)

from snpe.converters.tensorflow.layers.permute import (
    PermuteLayerResolver,
    PermuteLayerBuilder
)

from snpe.converters.tensorflow.layers.argmax import (
    ArgMaxLayerResolver,
    ArgMaxLayerBuilder
)

from snpe.converters.tensorflow.layers.channel_shuffle import (
    ChannelShuffleLayerResolver,
    ChannelShuffleLayerBuilder
)

from snpe.converters.tensorflow.layers.elu import (
    EluLayerResolver,
    EluLayerBuilder
)

from snpe.converters.tensorflow.layers.reduction import (
    ReductionMeanLayerResolver,
    ReductionMeanLayerBuilder,
    ReductionProdLayerResolver,
    ReductionProdLayerBuilder,
    ReductionSumLayerResolver,
    ReductionSumLayerBuilder,
    ReductionMinLayerResolver,
    ReductionMinLayerBuilder,
    ReductionMaxLayerResolver,
    ReductionMaxLayerBuilder
)

from snpe.converters.tensorflow.layers.eltwise_unary import (
    EltWiseUnarySqrtLayerResolver,
    EltWiseUnarySqrtLayerBuilder
)

from snpe.converters.tensorflow.layers.pow import (
    PowLayerResolver,
    PowLayerBuilder
)

from snpe.converters.tensorflow.layers.tile import (
    TileLayerResolver,
    TileLayerBuilder
)

from snpe.converters.tensorflow.layers.extract_glimpse import (
    ExtractGlimpseLayerResolver,
    ExtractGlimpseLayerBuilder
)

from snpe.converters.tensorflow.layers.image_projective_transform import (
    ImageProjectiveTransformLayerResolver,
    ImageProjectiveTransformLayerBuilder
)

from snpe.converters.tensorflow.layers.fake_quant import (
    FakeQuantLayerResolver,
    FakeQuantLayerBuilder
)

from snpe.converters.tensorflow.layers.pixel_shuffle import (
    PixelShuffleLayerResolver,
    PixelShuffleLayerBuilder
)


from snpe.converters.tensorflow.layers.crop_and_resize import (
    CropAndResizeLayerResolver,
    CropAndResizeLayerBuilder
)
from snpe.converters.tensorflow.layers.non_max_suppression import (
    NonMaxSuppressionLayerResolver,
    NonMaxSuppressionLayerBuilder
)

from snpe.converters.tensorflow.layers.moments import (
    MomentsLayerResolver,
    MomentsLayerBuilder
)

from snpe.converters.tensorflow.layers.space_to_depth import (
    SpaceToDepthLayerResolver,
    SpaceToDepthLayerBuilder
)

from snpe.converters.tensorflow.common import (
    LayerDescriptor,
    LayerResolver,
    LayerBuilder
)

layer_resolvers = [
    IgnoredLayersResolver,
    FakeQuantLayerResolver,
    SSDAnchorGeneratorResolver,
    SSDNmsResolver,
    ConvolutionLayerResolver,
    ConcatLayerResolver,
    FullyConnectedLayerResolver,
    ReluLayerResolver,
    Relu6LayerResolver,
    ReluMinMaxLayerResolver,
    SigmoidLayerResolver,
    TanhLayerResolver,
    AvgPoolingLayerResolver,
    MaxPoolingLayerResolver,
    NonMaxSuppressionLayerResolver,
    SoftmaxLayerResolver,
    LrnLayerResolver,
    DeConvolutionOptimizedLayerResolver,
    InstanceNormRMSLayerResolver,
    InstanceNormLayerResolver,
    EltWiseSumLayerResolver,
    EltWiseSubLayerResolver,
    EltWiseMulLayerResolver,
    EltWiseMaxLayerResolver,
    EltWiseDivLayerResolver,
    UnscaledBatchNormLayerResolver,
    ScaledBatchNormLayerResolver,
    BatchNormWithGlobalNormLayerResolver,
    GenericBatchNormLayerResolver,
    GroupedConvolutionLayerResolver,
    SliceLayerResolver,
    PReLuLayerResolver,
    DilatedConvolutionLayerResolver,
    ReshapeLayerResolver,
    ResizeBilinearLayerResolver,
    ResizeNearestNeighborLayerResolver,
    DepthwiseConvolutionLayerResolver,
    DilatedDepthwiseConvolutionLayerResolver,
    AddNLayerResolver,
    LstmLayerResolver,
    FillLayerResolver,
    SSDDecoderResolver,
    CropLayerResolver,
    FusedBatchNormNormLayerResolver,
    EmbeddingLayerResolver,
    PadLayerResolver,
    PowLayerResolver,
    PixelShuffleLayerResolver,
    StridedSliceLayerResolver,
    PermuteLayerResolver,
    ArgMaxLayerResolver,
    ChannelShuffleLayerResolver,
    EluLayerResolver,
    TileLayerResolver,
    ReductionMeanLayerResolver,
    ReductionProdLayerResolver,
    ReductionSumLayerResolver,
    ReductionMinLayerResolver,
    ReductionMaxLayerResolver,
    EltWiseUnarySqrtLayerResolver,
    ExtractGlimpseLayerResolver,
    ImageProjectiveTransformLayerResolver,
    CropAndResizeLayerResolver,
    MomentsLayerResolver,
    SpaceToDepthLayerResolver
]
"""
type: list[type(LayerResolver)]
"""

layer_builders = {
    BatchNormLayerResolver.Descriptor: BatchNormLayerBuilder,
    BatchNormWithGlobalNormLayerResolver.Descriptor: BatchNormLayerBuilder,
    GenericBatchNormLayerResolver.Descriptor: BatchNormLayerBuilder,
    ConcatLayerResolver.Descriptor: ConcatLayerBuilder,
    ConvolutionLayerResolver.Descriptor: ConvolutionLayerBuilder,
    DeConvolutionOptimizedLayerResolver.Descriptor: DeConvolutionLayerBuilder,
    EltWiseMaxLayerResolver.Descriptor: EltWiseMaxLayerBuilder,
    EltWiseMulLayerResolver.Descriptor: EltWiseMulLayerBuilder,
    EltWiseSumLayerResolver.Descriptor: EltWiseSumLayerBuilder,
    EltWiseSubLayerResolver.Descriptor: EltWiseSubLayerBuilder,
    EltWiseDivLayerResolver.Descriptor: EltWiseDivLayerBuilder,
    InstanceNormRMSLayerResolver.Descriptor: InstanceNormRMSLayerBuilder,
    InstanceNormLayerResolver.Descriptor: InstanceNormLayerBuilder,
    AddNLayerResolver.Descriptor: AddNLayerBuilder,
    TileLayerResolver.Descriptor: TileLayerBuilder,
    FullyConnectedLayerResolver.Descriptor: FullyConnectedLayerBuilder,
    FakeQuantLayerResolver.Descriptor: FakeQuantLayerBuilder,
    LrnLayerResolver.Descriptor: LrnLayerBuilder,
    ReluLayerResolver.Descriptor: ReluLayerBuilder,
    Relu6LayerResolver.Descriptor: ReluMinMaxLayerBuilder,
    ReluMinMaxLayerResolver.Descriptor: ReluMinMaxLayerBuilder,
    SigmoidLayerResolver.Descriptor: SigmoidLayerBuilder,
    SoftmaxLayerResolver.Descriptor: SoftmaxLayerBuilder,
    TanhLayerResolver.Descriptor: TanhLayerBuilder,
    AvgPoolingLayerResolver.Descriptor: PoolingLayerBuilder,
    MaxPoolingLayerResolver.Descriptor: PoolingLayerBuilder,
    NonMaxSuppressionLayerResolver.Descriptor: NonMaxSuppressionLayerBuilder,
    GroupedConvolutionLayerResolver.Descriptor: ConvolutionLayerBuilder,
    SliceLayerResolver.Descriptor: SliceLayerBuilder,
    PixelShuffleLayerResolver.Descriptor: PixelShuffleLayerBuilder,
    PReLuLayerResolver.Descriptor: PReLuLayerBuilder,
    DilatedConvolutionLayerResolver.Descriptor: ConvolutionLayerBuilder,
    ReshapeLayerResolver.Descriptor: ReshapeLayerBuilder,
    ResizeBilinearLayerResolver.Descriptor: ResizeLayerBuilder,
    ResizeNearestNeighborLayerResolver.Descriptor: ResizeLayerBuilder,
    LstmLayerResolver.UnrolledTimeStepDescriptor: LstmLayerBuilder,
    LstmLayerResolver.StateDescriptor: LstmLayerBuilder,
    IgnoredLayersResolver.Descriptor: IgnoredLayersBuilder,
    FillLayerResolver.Descriptor: FillLayerBuilder,
    SSDDecoderResolver.Descriptor: SSDDecoderLayersBuilder,
    CropLayerResolver.Descriptor: CropLayerBuilder,
    SSDNmsResolver.Descriptor: SSDNmsLayersBuilder,
    ConstantLayerResolver.Descriptor: ConstantLayerBuilder,
    FusedBatchNormNormLayerResolver.Descriptor: BatchNormLayerBuilder,
    EmbeddingLayerResolver.Descriptor: EmbeddingLayerBuilder,
    PadLayerResolver.Descriptor: PadLayerBuilder,
    StridedSliceLayerResolver.Descriptor: StridedSliceLayerBuilder,
    PermuteLayerResolver.Descriptor: PermuteLayerBuilder,
    ArgMaxLayerResolver.Descriptor: ArgMaxLayerBuilder,
    ChannelShuffleLayerResolver.Descriptor: ChannelShuffleLayerBuilder,
    EluLayerResolver.Descriptor: EluLayerBuilder,
    PowLayerResolver.Descriptor: PowLayerBuilder,
    ReductionMeanLayerResolver.Descriptor: ReductionMeanLayerBuilder,
    ReductionProdLayerResolver.Descriptor: ReductionProdLayerBuilder,
    ReductionSumLayerResolver.Descriptor: ReductionSumLayerBuilder,
    ReductionMinLayerResolver.Descriptor: ReductionMinLayerBuilder,
    ReductionMaxLayerResolver.Descriptor: ReductionMaxLayerBuilder,
    EltWiseUnarySqrtLayerResolver.Descriptor: EltWiseUnarySqrtLayerBuilder,
    ExtractGlimpseLayerResolver.Descriptor: ExtractGlimpseLayerBuilder,
    ImageProjectiveTransformLayerResolver.Descriptor: ImageProjectiveTransformLayerBuilder,
    CropAndResizeLayerResolver.Descriptor: CropAndResizeLayerBuilder,
    MomentsLayerResolver.Descriptor: MomentsLayerBuilder,
    SpaceToDepthLayerResolver.Descriptor: SpaceToDepthLayerBuilder
}
"""
type: dict[type(LayerDescriptor), type(LayerBuilder)]
"""
