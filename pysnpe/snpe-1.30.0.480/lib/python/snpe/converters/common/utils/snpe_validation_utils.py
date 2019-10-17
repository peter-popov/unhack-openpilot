#!/usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import argparse

valid_processor_choices = ('snapdragon_801', 'snapdragon_820', 'snapdragon_835')
valid_runtime_choices = ('cpu', 'gpu', 'dsp')


class ValidateTargetArgs(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        specified_runtime, specified_processor = values
        if specified_runtime not in valid_runtime_choices:
            raise ValueError('invalid runtime_target {s1!r}. Valid values are {s2}'.format(s1=specified_runtime,
                                                                                           s2=valid_runtime_choices)
                             )
        if specified_processor not in valid_processor_choices:
            raise ValueError('invalid processor_target {s1!r}. Valid values are {s2}'.format(s1=specified_processor,
                                                                                             s2=valid_processor_choices)
                             )
        setattr(args, self.dest, values)
