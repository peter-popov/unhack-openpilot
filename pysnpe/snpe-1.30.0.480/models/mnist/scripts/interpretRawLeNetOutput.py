#==============================================================================
#
#  Copyright (c) 2016 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

#!/usr/bin/python

# This script takes in a file of float values and print
# out index position and value pairs.
# Specifically, this script should take in output from
# the LeNet MNIST classification model.  The end of the
# scripts prints out a message indicating the index
# with the highest float value as the digit classified.

import numpy as np
import os
import os.path
import sys

def usage(msg='unknown error'):
  print('%s %s' % (sys.argv[0], 'raw_output_file'))
  print('error: %s' % msg)
  exit(1)

if len(sys.argv) != 2:
  usage('missing argument')

raw_output_file = sys.argv[1]

if not os.path.isfile(raw_output_file) or not os.access(raw_output_file, os.R_OK):
  usage('raw_output_file not accessible')

# load floats from file
float_array = np.fromfile(raw_output_file, dtype=np.float32)

if len(float_array) != 10:
  usage('cannot read 10 floats from raw_output_file')

max_prob = float_array[0]
max_idx = 0

# print out index and value pair, saving index with highest value
for i in range(len(float_array)):
  prob = float_array[i]
  if prob >= max_prob:
    max_prob = prob
    max_idx = i
  print(' %d : %f' % (i, float_array[i]))

print('LeNet MNIST classifies the digit as a %d' % max_idx)
