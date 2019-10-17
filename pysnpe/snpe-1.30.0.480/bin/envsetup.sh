#!/bin/bash
#==============================================================================
#
#  Copyright (c) 2016-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

# This script sets up the various environment variables needed to run various sdk binaries and scripts
OPTIND=1

_usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-h] [-c CAFFE_DIRECTORY] [-f CAFFE2_DIRECTORY] [-t TENSORFLOW_DIRECTORY]

Script sets up environment variables needed for running sdk binaries and scripts, where only one of the
Caffe, Caffe2, or Tensorflow directories have to be specified.

optional arguments:
 -c CAFFE_DIRECTORY            Specifies Caffe directory
 -f CAFFE2_DIRECTORY           Specifies Caffe2 directory
 -o ONNX_DIRECTORY             Specifies ONNX directory
 -t TENSORFLOW_DIRECTORY       Specifies TensorFlow directory

EOF
}

function _setup_snpe()
{
  # get directory of the bash script
  local SOURCEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  export SNPE_ROOT=$(readlink -f $SOURCEDIR/..)
  export PATH=$SNPE_ROOT/bin/x86_64-linux-clang:$PATH

  # setup LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH

  # setup PYTHONPATH
  export PYTHONPATH=$SNPE_ROOT/lib/python:$PYTHONPATH
  export PYTHONPATH=$SNPE_ROOT/models/lenet/scripts:$PYTHONPATH
  export PYTHONPATH=$SNPE_ROOT/models/alexnet/scripts:$PYTHONPATH
}

function _setup_caffe()
{
  if ! _is_valid_directory $1; then
    return 1
  fi

  # common setup
  _setup_snpe

  local CAFFEDIR=$1

  # current tested SHA for caffe
  local VERIFY_CAFFE_SHA="18b09e807a6e146750d84e89a961ba8e678830b4"

  # setup an environment variable called $CAFFE_HOME
  export CAFFE_HOME=$CAFFEDIR
  echo "[INFO] Setting CAFFE_HOME="$CAFFEDIR

  # update PATH
  export PATH=$CAFFEDIR/build/install/bin:$PATH
  export PATH=$CAFFEDIR/distribute/bin:$PATH

  # update LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$CAFFEDIR/build/install/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CAFFEDIR/distribute/lib

  # update PYTHONPATH
  export PYTHONPATH=$CAFFEDIR/build/install/python:$PYTHONPATH
  export PYTHONPATH=$CAFFEDIR/distribute/python:$PYTHONPATH

  # check Caffe SHA
  pushd $CAFFEDIR > /dev/null
  local CURRENT_CAFFE_SHA=$(git rev-parse HEAD)
  if [ "$VERIFY_CAFFE_SHA" != "$CURRENT_CAFFE_SHA" ]; then
    echo "[WARNING] Expected CAFFE HEAD rev "$VERIFY_CAFFE_SHA" but found "$CURRENT_CAFFE_SHA" instead. This SHA is not tested."
  fi
  popd > /dev/null

  return 0
}

function _setup_caffe2()
{
  if ! _is_valid_directory $1; then
    return 1
  fi

  # common setup
  _setup_snpe

  local CAFFE2DIR=$1

  # setup an environment variable called $CAFFE2_HOME
  export CAFFE2_HOME=$CAFFE2DIR
  echo "[INFO] Setting CAFFE2_HOME="$CAFFE2DIR

  # setup LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

  # setup PYTHONPATH
  export PYTHONPATH=$CAFFE2DIR/build/:$PYTHONPATH
  export PYTHONPATH=/usr/local/:$PYTHONPATH

  return 0
}

function _setup_onnx()
{
  if ! _is_valid_directory $1; then
    return 1
  fi

  # common setup
  _setup_snpe

  local ONNXDIR=$1

  # setup an environment variable called $ONNX_HOME
  export ONNX_HOME=$ONNXDIR
  echo "[INFO] Setting ONNX_HOME="$ONNXDIR

  return 0
}

function _setup_tensorflow()
{
  if ! _is_valid_directory $1; then
    return 1
  fi

  # common setup
  _setup_snpe

  local TENSORFLOWDIR=$1

  # setup an environment variable called $TENSORFLOW_HOME
  export TENSORFLOW_HOME=$TENSORFLOWDIR
  echo "[INFO] Setting TENSORFLOW_HOME="$TENSORFLOWDIR

  return 0
}

function _check_ndk()
{
  # check NDK in path
  if [[ ! -d "$ANDROID_NDK_ROOT" ]]; then
    local ndkDir=$(which ndk-build)
    if [ ! -s "$ndkDir" ]; then
      echo "[WARNING] Can't find ANDROID_NDK_ROOT or ndk-build. SNPE needs android ndk to build the NativeCppExample"
    else
      ANDROID_NDK_ROOT=$(dirname $ndkDir)
      echo "[INFO] Found ndk-build at " $ndkDir
    fi
  else
    echo "[INFO] Found ANDROID_NDK_ROOT at "$ANDROID_NDK_ROOT
  fi
}

function _is_valid_directory()
{
  if [[ ! -z $1 ]]; then
    if [[ ! -d $1 ]]; then
      echo "[ERROR] Invalid directory "$1" specified. Please rerun the srcipt with a valid directory path."
      return 1
    else
      return 0
    fi
  else
    return 1
  fi
}

function _cleanup()
{
  unset -f _usage
  unset -f _setup_snpe
  unset -f _setup_caffe
  unset -f _setup_caffe2
  unset -f _setup_tensorflow
  unset -f _check_ndk
  unset -f _is_valid_directory
  unset -f _cleanup
}

# script can only handle one framework per execution
[[ ($# -le 2) && ($# -gt 0) ]] || { echo "[ERROR] Invalid number of arguments. See -h for help."; return 1; }
_setup_snpe
# parse arguments
while getopts "h?c:f:o:t:" opt; do
  case $opt in
    h  ) _usage; return 0 ;;
    c  ) _setup_caffe $OPTARG || return 1 ;;
    f  ) _setup_caffe2 $OPTARG || return 1 ;;
    o  ) _setup_onnx $OPTARG || return 1 ;;
    t  ) _setup_tensorflow $OPTARG || return 1 ;;
    \? ) return 1 ;;
  esac
done

# check for NDK
_check_ndk

# cleanup
_cleanup
