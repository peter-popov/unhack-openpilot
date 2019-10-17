#!/usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import pprint

pp = pprint.PrettyPrinter()

# Class (an another way of enum class)
# TBD: Once minimum python version is upgraded for converter from 2.7 to 3.0
#      replace with enum class
class AxisAnnotation(object):
   """
   This class contains axis annotation required for axis tracking.
   """
   HEIGHT = 0
   WIDTH = 1
   CHANNEL = 2
   BATCH = 3
   TIME = 4
   FEATURE = 5
   # ANY indicates any of axis annotation is acceptable
   # Layers such as permute or concat
   ANY = 6
   # NONTRIVIAL indicates none of axis annotation is valid and not trivial to be derived
   # Layers such as reshape/flatten specify this axis annotation.
   NONTRIVIAL = 7

class LayerOrderedAxes(object):
   """
   This class maintains all axes order for each layer.
   If no axes order is specified for a layer, default axes order is returned.
   """
   def __init__(self, name, default_axes_order):

      """ Default axes order is specified for dict
          So, when layer type is absent in dict, it returns the
          default axes order.
      """
      self._name = name
      self._default_axes_order = default_axes_order
      self._layer_input_axes_dict = {}
      self._layer_output_axes_dict = {}
      self.logger = logging.getLogger()

   def dump(self):
      self.logger.debug(self._name + ":: Dump of layer ordered axes for input:")
      self.logger.debug(pp.pprint(self._layer_input_axes_dict))

      self.logger.debug(self._name + ":: Dump of layer ordered axes for output:")
      self.logger.debug(pp.pprint(self._layer_output_axes_dict))

   def add_axis_order(self, layer_type, input_axis_order, output_axis_order = [] ):
      """ Adds both input and output axis order by layer type
          For multiple i/o axis order for the same layer type, it
          is invoked multiple times.
      """
      curr_ival = self._layer_input_axes_dict.setdefault(layer_type, [])
      curr_ival.append(input_axis_order)
      self._layer_input_axes_dict[layer_type] = curr_ival

      if len(output_axis_order) == 0:
         output_axis_order = input_axis_order

      curr_oval = self._layer_output_axes_dict.setdefault(layer_type, [])
      curr_oval.append(output_axis_order)
      self._layer_output_axes_dict[layer_type] = curr_oval

      #self.dump()

   def get_input_axis_order(self, layer_type, input_rank):
      """ Returns input axis order by layer type
          and input rank.
      """
      self.logger.debug(self._name + ":: get input axis order for layer " + layer_type + " with input rank " + str(input_rank))
      input_axes_order_list = self._layer_input_axes_dict.get(layer_type, [self._default_axes_order])
      input_axes_order = input_axes_order_list[0]
      for item in input_axes_order_list:
         if len(item) == input_rank:
            input_axes_order = item
            break
      return input_axes_order, self.is_axis_order_trackable(input_axes_order)

   def is_axis_order_trackable(self, axes):
      return False if AxisAnnotation.NONTRIVIAL in axes else True

   def get_output_axis_order(self, layer_type, input_rank):
      """ Returns output axis order by layer type
          and output rank.
      """
      self.logger.debug(self._name + ":: get output axis order for layer " + layer_type + " with input rank " + str(input_rank))

      output_axes_order_list = self._layer_output_axes_dict.get(layer_type, [self._default_axes_order])
      input_axes_order_list = self._layer_input_axes_dict.get(layer_type, [self._default_axes_order])
      assert(len(output_axes_order_list)  == len(input_axes_order_list))

      output_axes_order = output_axes_order_list[0]

      # For a given input rank, find the output axis order
      for input_order, output_order in zip(input_axes_order_list, output_axes_order_list):
         if len(input_order) == input_rank:
            output_axes_order = output_order
            break

      return output_axes_order, self.is_axis_order_trackable(output_axes_order)

class AxisTracker(object):
   """
   This class tracks the current axis order for each buffer.
   """
   def __init__(self, name):
      self._name = name
      self._axes_order = {}
      self.logger = logging.getLogger()

   def dump(self):
      self.logger.debug(self._name + ":: Dump of axis tracker below:")
      self.logger.debug(pp.pprint(self._axes_order))

   def update_axis_order(self, buffer_name, axis_order):

      # Axis of buffer shouldn't contain ANY
      for axis in axis_order:
         assert(axis != AxisAnnotation.ANY)
      self._axes_order[buffer_name] = axis_order
      #self.dump()

   def has_axis_order(self, buffer_name):
      return buffer_name in self._axes_order

   def get_axis_order(self, buffer_name):
      assert buffer_name in self._axes_order, "Instance name " + self._name + ": Buffer name " + buffer_name + " doesnt exist in axis order."
      axes = self._axes_order[buffer_name]
      self.logger.debug(self._name + ":: Buffer = " + buffer_name + " Axes = " + str(axes) + " Trackable=" + str(self.is_axis_order_trackable(axes)))
      return axes, self.is_axis_order_trackable(axes)

   def is_axis_order_trackable(self, axes):
      return False if AxisAnnotation.NONTRIVIAL in axes else True

   def get_axis_annotation(self, buffer_name, axis):
      axis_order, trackable = self.get_axis_order(buffer_name)
      return axis_order[axis]

   def get_axis(self, buffer_name, annotation):
      axis_order, trackable = self.get_axis_order(buffer_name)
      assert( len(axis_order) )
      for i in range(len(axis_order)):
         if axis_order[i] == annotation:
            return i
      return -1

class AxisTransformer(object):
   def __init__(self, src_layer_ordered_axes, src_axis_tracker, target_layer_ordered_axes, target_axis_tracker):
     """
     This class is reponsible for detecting conditions that require
     any force convergence/divergence. It uses source and target
     axis trackers, and predefined source and target axis order for each layer type (and its input dim variants)
     to track current axes in both source and target.
     """
     self._src_layer_ordered_axes = src_layer_ordered_axes
     self._target_layer_ordered_axes = target_layer_ordered_axes
     self._src_axis_tracker = src_axis_tracker
     self._target_axis_tracker = target_axis_tracker
     self.logger = logging.getLogger()

   def compute_permute_order(self, current_order, expected_order):

     self.logger.debug("Current Axes=" + str(current_order) + " Expected Axes=" + str(expected_order))
     assert( set(current_order) == set(expected_order) )
     permute_order = []
     for axis in expected_order:
        permute_order.append(current_order.index(axis))
     return permute_order

   def get_implicit_permute_order(self, layer_type, input_rank, src_buffer_name, target_buffer_name):

      # Get both current and next axes order for source and target buffers.
      curr_target_axes, curr_target_trackable = self._target_axis_tracker.get_axis_order(target_buffer_name)
      next_target_axes, next_target_trackable = self._target_layer_ordered_axes.get_input_axis_order(layer_type, input_rank)

      curr_src_axes, curr_src_trackable = self._src_axis_tracker.get_axis_order(src_buffer_name)
      next_src_axes, next_src_trackable = self._src_layer_ordered_axes.get_input_axis_order(layer_type, input_rank)

      permute_order = []  # By default no permute is required.
      if curr_target_trackable: # Current state
         if next_target_trackable: # Next state
            # Check if both curr source and curr target axis order is the same
            comp = [i for i, j in zip(curr_src_axes, curr_target_axes) if i == j]
            if len(comp) == len(curr_src_axes):
               if AxisAnnotation.ANY not in next_target_axes:
                  self.logger.debug("Check if permute required going from trackable to trackable state w/ force divergence.")
                  permute_order = self.compute_permute_order(next_src_axes, next_target_axes)
            elif AxisAnnotation.ANY not in next_target_axes:
               self.logger.debug("Check if permute required going from trackable to trackable state.")
               permute_order = self.compute_permute_order(curr_target_axes, next_target_axes)
         else: # Next state
            # Compute permute order for force convergence
            self.logger.debug("Check if permute required going from trackable to untrackable state.")
            permute_order = self.compute_permute_order(curr_target_axes, curr_src_axes)

      else: # Current state
         if next_target_trackable: # Next state
            if AxisAnnotation.ANY not in next_target_axes:
               # Compute permute order for force divergence
               self.logger.debug("Check if permute required going from untrackable to trackable state.")
               permute_order = self.compute_permute_order(next_src_axes, next_target_axes)
      return permute_order

   def get_explicit_permute_order(self, layer_type, input_rank, src_buffer_name, target_buffer_name, src_permute_order):

      permute_order = []
      curr_target_axes, curr_target_trackable = self._target_axis_tracker.get_axis_order(target_buffer_name)
      curr_src_axes, curr_src_trackable = self._src_axis_tracker.get_axis_order(src_buffer_name)

      assert(curr_target_trackable == curr_src_trackable)
      if curr_target_trackable:
         # Determine the axis order of src after permute_order
         next_src_axis_order = []
         for order in src_permute_order:
            next_src_axis_order.append(curr_src_axes[order])

         # Compute permute order of target buffer from the "next" axis order of src
         permute_order = self.compute_permute_order(curr_target_axes, next_src_axis_order)

      return permute_order

   def has_target_axis_order(self, target_buffer_name):
      return self._target_axis_tracker.has_axis_order(target_buffer_name)

   def get_permute_order(self, layer_type, input_rank, src_buffer_name, target_buffer_name, src_permute_order = [] ):

      # Since no explicit permute order is requested
      if len(src_permute_order) == 0:
         target_permute_order = self.get_implicit_permute_order(layer_type, input_rank, src_buffer_name, target_buffer_name)
      else:
         # Explicit permute order is requested, determine the correct target permute order
         target_permute_order = self.get_explicit_permute_order(layer_type, input_rank, src_buffer_name, target_buffer_name, src_permute_order)
      return target_permute_order

   def update_src_axis_order(self, layer_type, output_rank, output_buffer_name, input_rank, input_buffer_name = None, axis_order = [] ):
      self.logger.debug("update_src_axis_order: type " + layer_type + " outrank = " + str(output_rank) +  " output buffer name " + output_buffer_name)
      self.logger.debug("update_src_axis_order: type " + layer_type + " input_rank = " + str(input_rank) +  " input buffer name " + str(input_buffer_name))

      save_axis_order = axis_order
      if len(save_axis_order) == 0:
         layer_axis_order, istrackable = self._src_layer_ordered_axes.get_output_axis_order(layer_type, input_rank)

         if not istrackable:               # Reshape/flatten layer
            save_axis_order = [AxisAnnotation.NONTRIVIAL] * output_rank
         elif istrackable and AxisAnnotation.ANY in layer_axis_order: # concat/softmax layer
            # Since layer type is indicating that its output axis order is determined by input (via ANY)
            # let's look at the input's axis order and save that axis order.
            assert(input_buffer_name != None)
            input_axis_order, istrackable = self._src_axis_tracker.get_axis_order(input_buffer_name)
            save_axis_order = input_axis_order
         else:                            #
            save_axis_order = layer_axis_order

      self.logger.debug("Update source axis order for buffer name " + output_buffer_name + " with axis " + str(save_axis_order).strip('[]'))
      self._src_axis_tracker.update_axis_order(output_buffer_name, save_axis_order)

   def update_target_axis_order(self, layer_type, output_rank, output_buffer_name, input_rank, input_buffer_name = None, axis_order = [] ):

      self.logger.debug("update_target_axis_order: type " + layer_type + " outrank = " + str(output_rank) +  " output buffer name " + output_buffer_name)
      self.logger.debug("update_target_axis_order: type " + layer_type + " input_rank = " + str(input_rank) +  " input buffer name " + str(input_buffer_name))
      save_axis_order = axis_order
      if len(save_axis_order) == 0:
         layer_axis_order, istrackable = self._target_layer_ordered_axes.get_output_axis_order(layer_type, input_rank)

         if not istrackable:               # Reshape/flatten layer
            save_axis_order = [AxisAnnotation.NONTRIVIAL] * output_rank
         elif istrackable and AxisAnnotation.ANY in layer_axis_order: # concat/softmax layer
            # Since layer type is indicating that its output axis order is determined by input (via ANY)
            # let's look at the input's axis order and save that axis order.
            assert(input_buffer_name != None)
            input_axis_order, istrackable = self._target_axis_tracker.get_axis_order(input_buffer_name)
            save_axis_order = input_axis_order
         else:                            # Others...
            save_axis_order = layer_axis_order

      self.logger.debug("Update target axis order for buffer name " + output_buffer_name + " with axis order " + str(save_axis_order).strip('[]'))
      self._target_axis_tracker.update_axis_order(output_buffer_name, save_axis_order)

   def get_target_axis(self, src_buffer_name, src_axis, target_buffer_name ):

      # Get the annotation from src axis tracker
      anno = self._src_axis_tracker.get_axis_annotation(src_buffer_name, src_axis)

      self.logger.debug("Source buffer " + src_buffer_name + " axis " + str(src_axis) + " annotation " + str(anno))

      # Since current axis annotation is nontrivial, it means that
      # axes for both src and target are aligned. Thus, axis mapping is straightforward
      if anno == AxisAnnotation.NONTRIVIAL:

         # If the source axis order is 1 greater than target axis order
         # This is current case b/n Caffe and SNPE as SNPE doesn't support Batch dimension.
         src_axis_order, src_istrackable = self._src_axis_tracker.get_axis_order(src_buffer_name)
         target_axis_order, target_istrackable = self._target_axis_tracker.get_axis_order(target_buffer_name)
         srclen = len(src_axis_order)
         targetlen = len(target_axis_order)
         if srclen == targetlen:
            return src_axis

      # Get the axis from target axis tracker using the source annotation
      target_axis =  self._target_axis_tracker.get_axis(target_buffer_name, anno)
      self.logger.debug("Target buffer " + target_buffer_name + " axis " + str(target_axis) + " annotation " + str(anno))
      return target_axis

   def get_src_axis_order(self, buffer_name):
      axis_order, istrackable = self._src_axis_tracker.get_axis_order(buffer_name)
      return axis_order

   def get_target_axis_order(self, buffer_name):
      axis_order, istrackable = self._target_axis_tracker.get_axis_order(buffer_name)
      return axis_order
