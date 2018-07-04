# Crossroad Camera Sample {#CrossroadCameraSampleApplication}

This sample provides an inference pipeline for persons' detection, recognition and reidentification. The sample uses Person Detection network followed by the Person Attributes Recognition and Person Reidentification Retail networks applied on top of the detection results. The corresponding pre-trained models are delivered with the product:

* `person-vehicle-bike-detection-crossroad-0078`, which is a primary detection network for finding the persons (and other objects if needed)
* `person-attributes-recognition-crossroad-0031`, which is executed on top of the results from the first network and
reports person attributes like gender, has hat, has long-sleeved clothes
* `person-reidentification-retail-0079`, which is executed on top of the results from the first network and prints
a vector of features for each detected person. This vector is used to conclude if it is already detected person or not.

For details on the models, please refer to the descriptions in the `deployment_tools/intel_models` folder of the
 OpenVINO&trade; toolkit installation directory.

Other sample objectives are:
* Images/Video/Camera as inputs, via OpenCV*
* Example of simple networks pipelining: Person Attributes and Person Reidentification networks are executed on top of
the Person Detection results
* Visualization of Person Attributes and Person Reidentification (REID) information for each detected person


## How It Works

On the start-up, the application reads command line parameters and loads the specified networks. The Person Detection
network is required, the other two are optional.

Upon getting a frame from the OpenCV VideoCapture, the application performs inference of Person Detection network, then performs another
two inferences of Person Attributes Recognition and Person Reidentification Retail networks if they were specified in the
command line, and displays the results.
In case of the Person Reidentification Retail network, the resulting vector is generated for each detected person. This vector is
compared one-by-one with all previously detected persons vectors using cosine similarity algorithm. If comparison result
is greater than the specified (or default) threshold value, it is concluded that the person was already detected and a known
REID value is assigned. Otherwise, the vector is added to a global list, and a new REID value is assigned.

## Running

Running the application with the `-h` option yields the following usage message:
```sh
./crossroad_camera_sample -h
InferenceEngine:
	API version ............ 1.0

crossroad_camera_sample [OPTION]
Options:

    -h                           Print a usage message.
    -i "<path>"                  Required. Path to a video or image file. Default value is "cam" to work with camera.
    -m "<path>"                  Required. Path to the Person/Vehicle/Bike Detection Crossroad model (.xml) file.
    -m_pa "<path>"               Optional. Path to the Person Attributes Recognition Crossroad model (.xml) file.
    -m_reid "<path>"             Optional. Path to the Person Reidentification Retail model (.xml) file.
      -l "<absolute_path>"       For MKLDNN (CPU)-targeted custom layers, if any. Absolute path to a shared library with the kernels impl.
          Or
      -c "<absolute_path>"       For clDNN (GPU)-targeted custom kernels, if any. Absolute path to the xml file with the kernels desc.
    -d "<device>"                Specify the target device for Person/Vehicle/Bike Detection (CPU, GPU, FPGA, MYRIAD, or HETERO).
    -d_pa "<device>"             Specify the target device for Person Attributes Recognition (CPU, GPU, FPGA, MYRIAD, or HETERO).
    -d_reid "<device>"           Specify the target device for Person Reidentification Retail (CPU, GPU, FPGA, MYRIAD, or HETERO).
    -no_show                     No show processed video.
    -pc                          Enables per-layer performance statistics.
    -r                           Output Inference results as raw values.
    -t                           Probability threshold for person/vehicle/bike crossroad detections.
    -t_reid                      Cosine similarity threshold between two vectors for person reidentification.

```
## Sample Output

The sample uses OpenCV to display the resulting frame with detections rendered as bounding boxes and text.
In the default mode, the sample reports **Person Detection time** - inference time for the Person/Vehicle/Bike Detection network.

If Person Attributes Recognition or Person Reidentification Retail are enabled, the additional info below is reported also:
	* **Person Attributes Recognition time** - Inference time of Person Attributes Recognition averaged by the number of detected persons.
	* **Person Reidentification time** - Inference time of Person Reidentification averaged by the number of detected persons.


## See Also
* [Using Inference Engine Samples](@ref SamplesOverview)