#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IEPlugin


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--input",
                        help="Path to video file or image. 'cam' for capturing video stream from camera", required=True,
                        type=str)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    return parser


def main():
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    # Plugin initialization for specified device and load extensions library if specified
    print("Initializing plugin for {} device...".format(args.device))
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)
    # Read IR
    print("Reading IR...")
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)
    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    print("Loading IR to the plugin...")
    exec_net = plugin.load(network=net)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob]
    img = cv2.imread(args.input)
    img = cv2.resize(img, (w, h))
    img = img.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    img = img.reshape((n, c, h, w))
    res = exec_net.infer(inputs={input_blob: img})[out_blob]
    # Post process output
    res = np.squeeze(res, axis=0)  # Remove batch dimension
    # Clip values to [0, 255] range
    res = np.swapaxes(res, 0, 2)
    res = np.swapaxes(res, 0, 1)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    res[res < 0] = 0
    res[res > 255] = 255
    cv2.imwrite("out.jpg", res)
    print("Output image was saved to {}".format(os.path.join(os.path.dirname(__file__), "out.jpg")))
    # Explicit object deleting required to guarantee that plugin will not be deleted before executable network
    del exec_net
    del plugin


if __name__ == '__main__':
    sys.exit(main() or 0)
