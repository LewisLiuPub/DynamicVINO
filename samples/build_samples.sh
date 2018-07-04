#!/bin/bash

# Copyright (c) 2018 Intel Corporation
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

error() {
    local code="${3:-1}"
    if [[ -n "$2" ]];then
        echo "Error on or near line $1: $2; exiting with status ${code}"
    else
        echo "Error on or near line $1; exiting with status ${code}"
    fi
    exit "${code}" 
}
trap 'error ${LINENO}' ERR

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$ROOT_DIR/../..

if [[ -z "${INTEL_CVSDK_DIR}" ]]; then
    printf "\n\nINTEL_CVSDK_DIR environment variable is not set. Trying to run ./setvars.sh to set it. \n"
    
    if [ -e "$ROOT_DIR/inference_engine/bin/setvars.sh" ]; then # for Intel Deep Learning Deployment Toolkit package
        setvars_path="$ROOT_DIR/inference_engine/bin/setvars.sh"
    elif [ -e "$ROOT_DIR/../bin/setupvars.sh" ]; then # for OpenVINO package
        setvars_path="$ROOT_DIR/../bin/setupvars.sh"
    elif [ -e "$ROOT_DIR/../setupvars.sh" ]; then # for GO SDK package
        setvars_path="$ROOT_DIR/../setupvars.sh"
    else
        printf "Error: setvars.sh is not found\n"
    fi 
    if ! source $setvars_path ; then
        printf "Unable to run ./setvars.sh. Please check its presence. ${run_again}"
        exit 1
    fi
fi

cvsdk_install_dir="${INTEL_CVSDK_DIR}"

samples_path="${cvsdk_install_dir}/deployment_tools/inference_engine/samples"
cd $samples_path

if ! command -v cmake &>/dev/null; then
    printf "\n\nCMAKE is not installed. It is required to build Inference Engine samples. Please install it. ${run_again}"
    exit 1
fi

build_dir=$HOME/inference_engine_samples_build
mkdir -p $build_dir
cd $build_dir
cmake -DCMAKE_BUILD_TYPE=Release $samples_path
make -j8

printf "\nBuild completed, you can find binaries for all samples in the $HOME/inference_engine_samples_build/intel64/Release subfolder.\n\n"