# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from abc import ABCMeta, abstractmethod

import snpe.converters.common.converter_ir.op_graph as op_graph
from snpe.converters.common.converter_ir.op_policies import ConversionNamePolicy, ConversionShapeInferencePolicy
from snpe.converters.common.utils import snpe_validation_utils
from snpe.converters.common.utils.argparser_util import ArgParserWrapper
from snpe.converters.common.utils.snpe_converter_utils import *


class ConverterBase(object):
    __metaclass__ = ABCMeta

    class ArgParser(object):
        def __init__(self, src_framework):
            self.parser = ArgParserWrapper(description='Script to convert ' + src_framework + 'model into a DLC file.')
            # TODO: as deprecation step, setting required to False for now so that scripts don't break.
            #       please adjust for 1.31.0 release
            self.parser.add_required_argument("--input_network", "-i", type=str, required=False,
                                              help="Path to the source framework model.")

            self.parser.add_optional_argument('-o', '--output_path', type=str,
                                              help='Path where the converted Output model should be saved.If not '
                                                   'specified, the converter model will be written to a file with same '
                                                   'name as the input model')
            self.parser.add_optional_argument('--copyright_file', type=str,
                                              help='Path to copyright file. If provided, the content of the file will '
                                                   'be added to the output model.')
            self.parser.add_optional_argument('--model_version', type=str, default=None,
                                              help='User-defined ASCII string to identify the model, only first '
                                                   '64 bytes will be stored')
            self.parser.add_optional_argument("--disable_batchnorm_folding",
                                              help="If not specified, converter will try to fold batchnorm into "
                                                   "previous convolution layer", action="store_true")
            self.parser.add_optional_argument('--input_type', "-t", nargs=2, action='append',
                                              help='Type of data expected by each input op/layer. Type for each input '
                                                   'is |default| if not specified. For example: "data" image.Note that '
                                                   'the quotes should always be included in order to handle special '
                                                   'characters, spaces,etc. For multiple inputs specify multiple '
                                                   '--input_type on the command line. Eg: --input_type "data1" image '
                                                   '--input_type "data2" opaque '
                                                   'These options get used by DSP runtime and following descriptions '
                                                   'state how input will be handled for each option.'
                                                   ' Image: input is float between 0-255 and the input\'s mean is 0.0f '
                                                   'and the input\'s max is 255.0f. We will cast the float to uint8ts '
                                                   'and pass the uint8ts to the DSP. '
                                                   ' Default: pass the input as floats to the dsp directly and the DSP '
                                                   'will quantize it.'
                                                   ' Opaque: assumes input is float because the consumer layer(i.e next'
                                                   ' layer) requires it as float, therefore it won\'t be quantized.'
                                                   'Choices supported:' + str(op_graph.InputType.
                                                                              get_supported_types()),
                                              metavar=('INPUT_NAME', 'INPUT_TYPE'), default=[])
            self.parser.add_optional_argument('--input_encoding', "-e", nargs=2, action='append',
                                              help='Image encoding of the source images. Default is bgr. '
                                                   'Eg usage: "data" rgba Note the quotes should always be included '
                                                   'in order to handle special characters, spaces, etc. For multiple '
                                                   'inputs specify --input_encoding for each on the command line. Eg: '
                                                   '--input_encoding "data1" rgba --input_encoding "data2" other. '
                                                   'Use options: '
                                                   'color encodings(bgr,rgb, nv21...) if input is image; '
                                                   'time_series: for inputs of rnn models; '
                                                   'other: if input doesn\'t follow above categories or is unknown. '
                                                   'Choices supported:' + str(op_graph.InputEncodings.
                                                                              get_supported_encodings()),
                                              metavar=('INPUT_NAME', 'INPUT_ENCODING'), default=[])
            self.parser.add_optional_argument('--validation_target', nargs=2,
                                              action=snpe_validation_utils.ValidateTargetArgs,
                                              help="A combination of processor and runtime target against which model "
                                                   "will be validated. Choices for RUNTIME_TARGET: {cpu, gpu, dsp}. "
                                                   "Choices for PROCESSOR_TARGET: {snapdragon_801, snapdragon_820, "
                                                   "snapdragon_835}.If not specified, will validate model against "
                                                   "{snapdragon_820, snapdragon_835} across all runtime targets.",
                                              metavar=('RUNTIME_TARGET', 'PROCESSOR_TARGET'), default=[],)
            self.parser.add_optional_argument('--strict', dest="enable_strict_validation", action="store_true",
                                              default=False,
                                              help="If specified, will validate in strict mode whereby model will not "
                                                   "be produced if it violates constraints of the specified validation "
                                                   "target. If not specified, will validate model in permissive mode "
                                                   "against the specified validation target.")
            self.parser.add_optional_argument("--debug", type=int, nargs='?', const=0, default=-1,
                                              help="Run the converter in debug mode.")

        def parse_args(self, args=None, namespace=None):
            return self.parser.parse_args(args, namespace)

    def __init__(self, args,
                 naming_policy=ConversionNamePolicy(),
                 shape_inference_policy=ConversionShapeInferencePolicy()):

        setup_logging(args)
        self.graph = op_graph.IROpGraph(naming_policy, shape_inference_policy, args.input_type, args.input_encoding)
        self.input_model_path = args.input_network
        self.output_model_path = args.output_path
        self.copyright_str = get_string_from_txtfile(args.copyright_file)
        self.model_version = args.model_version
        self.disable_batchnorm_folding = args.disable_batchnorm_folding
        self.validation_target = args.validation_target
        self.enable_strict_validation = args.enable_strict_validation
        # TODO: remove caffe and onnx ignore arguments once deprecation is complete
        self.converter_command = sanitize_args(args, args_to_ignore=['input_network', 'i', 'output_path', 'o',
                                                                     'caffe_txt', 'c', 'model_path', 'm'])
        self.debug = args.debug

    @abstractmethod
    def convert(self):
        """
        Convert the input framework model to IROpGraph
        """
        pass
