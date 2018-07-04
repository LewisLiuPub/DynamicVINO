# Security Barrier Camera Sample {#SecurityBarrierCameraSampleApplication}

This sample showcases Vehicle Detection followed by the Vehicle Attributes and License-Plate Recognition applied on top
of the detection results. The corresponding topologies are shipped with the product ("intel_models" folder):
* _vehicle-license-plate-detection-barrier-0007_, which is primary detection network to find the vehicles and licence-plates
* _vehicle-attributes-recognition-barrier-0010_, this network is executed on top of the results from the first network and
reports the general vehicle attributes like type (car/van/bus/track) and color
* _license-plate-recognition-barrier-0001_, this network is executed on top of the results from the first network
and reports a string per recognized license-plate.
For more details on the topologies please refer to the descriptions in the `deployment_tools/intel_models` folder of the Intel OpenVINO&trade; toolkit installation.

Other demo objectives are:
* Images/Video/Camera as inputs, via OpenCV
* Example of simple networks pipelining: Attributes and LPR networks are executed on top of the Vehicle Detection results
* Visualization of Vehicle Attributes and Licence Plate information for each detected vehicle


### How it works

On the start-up the application reads command line parameters and loads the specified networks. The Vehicle/License-Plate
Detection network is required, and the other two are optional.

Upon getting a frame from the OpenCV's VideoCapture the app performs inference of Vehicles/License-Plates, then performs
another two inferences using  Vehicle Attributes and LPR detection networks (if those specified in command line) and displays the results.

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```sh
./security_barrier_camera_sample -h
    InferenceEngine:
        API version ............ 1.0
    [ INFO ] Parsing input parameters

    interactive_vehicle_detection [OPTION]
    Options:

        -h                         Print a usage message.
        -i "<path>"                Required. Path to a video or image file. Default value is "cam" to work with camera.
        -m "<path>"                Required. Path to the Vehicle/License-Plate Detection model (.xml) file.
        -m_va "<path>"             Optional. Path to the Vehicle Attributes model (.xml) file.
        -m_lpr "<path>"            Optional. Path to the License-Plate Recognition model (.xml) file.
          -l "<absolute_path>"     For MKLDNN (CPU)-targeted custom layers, if any. Absolute path to a shared library with the kernels impl.
              Or
          -c "<absolute_path>"     For clDNN (GPU)-targeted custom kernels, if any. Absolute path to the xml file with the kernels desc.
        -d "<device>"              Specify the target device for Vehicle Detection (CPU, GPU, FPGA, MYRIAD, or HETERO).
        -d_va "<device>"           Specify the target device for Vehicle Attributes (CPU, GPU, FPGA, MYRIAD, or HETERO).
        -d_lpr "<device>"          Specify the target device for License Plate Recognition (CPU, GPU, FPGA, MYRIAD, or HETERO).
        -pc                        Enables per-layer performance statistics.
        -r                         Output Inference results as raw values.
        -t                         Probability threshold for Vehicle/Licence-Plate detections.
```
### Demo Output

The demo uses OpenCV to display the resulting frame with detections rendered as bounding boxes and text:
![Security Camera Sample example output](example_sample_output.png)


## See Also
* [Using Inference Engine Samples](@ref SamplesOverview)
