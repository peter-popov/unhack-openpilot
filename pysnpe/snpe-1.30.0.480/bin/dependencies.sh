#! /bin/bash
#==============================================================================
#
#  Copyright (c) 2016 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

#Dependencies that are needed for sdk running
needed_depends=()
needed_depends+=('libleveldb-dev')
needed_depends+=('libsnappy-dev')
needed_depends+=('libopencv-dev')
needed_depends+=('libhdf5-serial-dev')
needed_depends+=('libboost-all-dev')
needed_depends+=('libatlas-base-dev')
needed_depends+=('cmake')
needed_depends+=('libgflags-dev')
needed_depends+=('libgoogle-glog-dev')
needed_depends+=('liblmdb-dev')
needed_depends+=('libprotobuf-dev')
needed_depends+=('protobuf-compiler')
needed_depends+=('python-dev')
needed_depends+=('wget')
needed_depends+=('zip')

#number of version_depends must match number of needed_depends
version_depends=()
version_depends+=('Version: 1.15.0-2')
version_depends+=('Version: 1.1.0-1ubuntu1')
version_depends+=('Version: 2.4.8+dfsg1-2ubuntu1')
version_depends+=('Version: 1.8.11-5ubuntu7')
version_depends+=('Version: 1.54.0.1ubuntu1')
version_depends+=('Version: 3.10.1-4')
version_depends+=('Version: 2.8.12.2-0ubuntu3')
version_depends+=('Version: 2.0-1.1ubuntu1')
version_depends+=('Version: 0.3.3-1')
version_depends+=('Version: 0.9.10-1')
version_depends+=('Version: 2.5.0-9ubuntu1')
version_depends+=('Version: 2.5.0-9ubuntu1')
version_depends+=('Version: 2.7.5-5ubuntu3')
version_depends+=('Version: 1.15-1ubuntu1')
version_depends+=('Version: 3.0-8')

#Unmet dependencies
need_to_install=()

i=0
while [ $i -lt ${#needed_depends[*]} ]; do
  PKG_INSTALLED=$(dpkg-query -W --showformat='${Status}\n' ${needed_depends[$i]}|grep "install ok installed")
  echo "Checking for ${needed_depends[$i]}: $PKG_INSTALLED"
  if [ "$PKG_INSTALLED" == "" ]; then
      echo "${needed_depends[$i]} is not installed. Adding to list of packages to be installed"
      need_to_install+=(${needed_depends[$i]})
  else
      current_version=$(dpkg -s ${needed_depends[$i]} | grep Version)
      if [ "$current_version" == "${version_depends[$i]}" ]; then
          echo "Success: Version of ${needed_depends[$i]} matches tested version"
      else
          echo "WARNING: Version of ${needed_depends[$i]} on this system which is $current_version does not match tested version which is ${version_depends[$i]}"
      fi
  fi
  i=$(( $i +1));
done

for j in "${need_to_install[@]}"
do
    sudo apt-get install $j
done


