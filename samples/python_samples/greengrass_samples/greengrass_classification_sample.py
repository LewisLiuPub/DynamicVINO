"""
BSD 3-clause "New" or "Revised" license

Copyright (C) 2018 Intel Coporation.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import sys
import os
import cv2
import numpy as np
import greengrasssdk
import boto3
import timeit
import datetime
import json

from openvino.inference_engine import IENetwork, IEPlugin

# Specify the delta in seconds between each report
reporting_interval = 1.0

# Parameters for IoT Cloud
enable_iot_cloud_output = True
topic_name = "openvino/classification"

# Parameters for Kinesis
enable_kinesis_output = False
kinesis_stream_name = ""
kinesis_partition_key = ""
kinesis_region = ""

# Parameters for S3
enable_s3_jpeg_output = False
s3_bucket_name = ""

# Parameters for jpeg output on local disk
enable_local_jpeg_output = False

# Create a Greengrass Core SDK client for publishing messages to AWS Cloud
client = greengrasssdk.client("iot-data")

# Create an S3 client for uploading files to S3
s3_client = boto3.client("s3")

# Create a Kinesis client for putting records to streams
kinesis_client = boto3.client("kinesis", "us-west-2")

# Read environment variables set by Lambda function configuration
PARAM_MODEL_XML = os.environ.get("PARAM_MODEL_XML")
PARAM_INPUT_SOURCE = os.environ.get("PARAM_INPUT_SOURCE")
PARAM_DEVICE = os.environ.get("PARAM_DEVICE")
PARAM_OUTPUT_DIRECTORY = os.environ.get("PARAM_OUTPUT_DIRECTORY")
PARAM_CPU_EXTENSION_PATH = os.environ.get("PARAM_CPU_EXTENSION_PATH")
PARAM_NUM_TOP_RESULTS = int(os.environ.get("PARAM_NUM_TOP_RESULTS"))

def report(res_json, frame):
    now = datetime.datetime.now()
    date_prefix = str(now).replace(" ", "_")
    if enable_iot_cloud_output:
        data = json.dumps(res_json)
        client.publish(topic=topic_name, payload=data)
    if enable_kinesis_output:
        kinesis_client.put_record(StreamName=kinesis_stream_name, Data=json.dumps(res_json), PartitionKey=kinesis_partition_key)
    if enable_s3_jpeg_output:
        temp_image = os.path.join(PARAM_OUTPUT_DIRECTORY, "inference_result.jpeg")
        cv2.imwrite(temp_image, frame)
        with open(temp_image) as file:
            image_contents = file.read()
            s3_client.put_object(Body=image_contents, Bucket=s3_bucket_name, Key=date_prefix + ".jpeg")
    if enable_local_jpeg_output:
        cv2.imwrite(os.path.join(PARAM_OUTPUT_DIRECTORY, date_prefix + ".jpeg"), frame)
        
def greengrass_classification_sample_run():
    client.publish(topic=topic_name, payload="OpenVINO: Initializing...")
    model_bin = os.path.splitext(PARAM_MODEL_XML)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device=PARAM_DEVICE, plugin_dirs="")
    if "CPU" in PARAM_DEVICE:
        plugin.add_cpu_extension(PARAM_CPU_EXTENSION_PATH)
    # Read IR
    net = IENetwork.from_ir(model=PARAM_MODEL_XML, weights=model_bin)
    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob]
    cap = cv2.VideoCapture(PARAM_INPUT_SOURCE)
    exec_net = plugin.load(network=net)
    del net
    client.publish(topic=topic_name, payload="Starting inference on %s" % PARAM_INPUT_SOURCE)
    start_time = timeit.default_timer()
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (w, h))
        in_frame = frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))

        # Start sync inference
        res = exec_net.infer(inputs={input_blob: in_frame})
        top_ind = np.argsort(res[out_blob], axis=1)[0, -PARAM_NUM_TOP_RESULTS:][::-1]
        res_json = []
        for i in top_ind:
            res_json.append({"confidence": round(res[out_blob][0, i], 2), "label": i})
        
        # Measure elapsed seconds since the last report
        seconds_elapsed = timeit.default_timer() - start_time
        if seconds_elapsed >= reporting_interval:
            report(res_json, frame)
            start_time = timeit.default_timer()

    client.publish(topic=topic_name, payload="End of the input, exiting...")
    del exec_net
    del plugin

greengrass_classification_sample_run()

def function_handler(event, context):
    client.publish(topic=topic_name, payload='HANDLER_CALLED!')
    return
