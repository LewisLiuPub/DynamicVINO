# Interactive Face Detection Sample {#InferenceEngineInteractiveFaceDetectionSampleApplication}

This sample showcases Object Detection task applied for face recognition using sequence of neural networks.
Async API usage can improve overall frame-rate of the application, because rather than wait for inference to complete,
the app can continue doing things on the host, while accelerator is busy.
Specifically, this demo keeps three parallel infer requests for Age Gender Head Pose and Emotions detection that run simultaneously.

Other demo objectives are:
* Video as input support via OpenCV
* Visualization of the resulting face bounding boxes from Face Detection network
* Visualization of age gender, head pose and emotion recognition information for each detected face
* OpenCV is used to draw resulting bounding boxes, labels, etc, so you can copy paste this code without
need to pull Inference Engine samples helpers to your app

### How it works

On the start-up the application reads command line parameters and loads one, two, three or four networks depending on -d... options family to the Inference
Engine. Upon getting a frame from the OpenCV's VideoCapture it performs inference of frame detection network, then performs three simultaneous inferences
using Age Gender, Head Pose and Emotions detection networks (if those specified in command line) and displays the results.

New "Async API" operates with new notion of the "Infer Request" that encapsulates the inputs/outputs and separates *scheduling and waiting for result*,
next section. And here what makes the performance look different:
1. In the default ("Sync") mode the frame is captured and then immediately processed, below in pseudo-code:
```cpp
    while(true) {
        capture frame
        populate FaceDetection InferRequest
        wait for the FaceDetection InferRequest
        populate AgeGender InferRequest using dyn batch technique
        populate HeadPose InferRequest using dyn batch technique
        populate EmotionDetection InferRequest using dyn batch technique
        wait AgeGender
        wait HeadPose
        wait EmotionDetection
        display detection results
    }
```
    So, this is rather reference implementation, where the new Async API is used in the serialized/synch fashion.


## Running

Running the application with the <code>-h</code> option yields the following usage message:
```sh
./interactive_face_detection -h
InferenceEngine: 
    API version ............ <version>
    Build .................. <number>

interactive_face_detection [OPTION]
Options:

    -h                         Print a usage message.
    -i "<path>"                Optional. Path to an video file. Default value is "cam" to work with camera.
    -m "<path>"                Required. Path to an .xml file with a trained face detection model.
    -m_ag "<path>"             Optional. Path to an .xml file with a trained age gender model.
    -m_hp "<path>"             Optional. Path to an .xml file with a trained head pose model.
    -m_em "<path>"             Optional. Path to an .xml file with a trained emotions model.
      -l "<absolute_path>"     Required for MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels impl.
          Or
      -c "<absolute_path>"     Required for clDNN (GPU)-targeted custom kernels.Absolute path to the xml file with the kernels desc.
    -d "<device>"              Specify the target device for Face Detection (CPU, GPU, FPGA, or MYRIAD). Sample will look for a suitable plugin for device specified.
    -d_ag "<device>"           Specify the target device for Age Gender Detection (CPU, GPU, FPGA, or MYRIAD). Sample will look for a suitable plugin for device specified.
    -d_hp "<device>"           Specify the target device for Head Pose Detection (CPU, GPU, FPGA, or MYRIAD). Sample will look for a suitable plugin for device specified.
    -d_em "<device>"           Specify the target device for Emotions Detection (CPU, GPU, FPGA, or MYRIAD). Sample will look for a suitable plugin for device specified.
    -n_ag "<num>"              Specify number of maximum simultaneously processed faces for Age Gender Detection (default is 16).
    -n_hp "<num>"              Specify number of maximum simultaneously processed faces for Head Pose Detection (default is 16).
    -n_em "<num>"              Specify number of maximum simultaneously processed faces for Emotions Detection (default is 16).
    -no_wait                   No wait for key press in the end.
    -no_show                   No show processed video.
    -pc                        Enables per-layer performance report.
    -r                         Inference results as raw values.
    -t                         Probability threshold for detections.

```

Running the application with the empty list of options yields the usage message given above and an error message.
You can use the following command to do inference on a GPU with an example
pre-trained GoogleNet based SSD* available at https://software.intel.com/file/609199/download:
```sh
./interactive_face_detection -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/ssd.xml -d GPU
```
Notice that the network should be converted from the Caffe* (*.prototxt + *.model) to the Inference Engine format
(*.xml + *bin) first, by use of the ModelOptimizer tool
(https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer).

### Sample Output

The sample uses OpenCV to display the resulting frame with detections (rendered as bounding boxes and labels, if provided).
In the default mode the sample reports
* **OpenCV time**: frame decoding + time to render the bounding boxes, labels, and displaying the results.
* **Face Detection time**: inference time for the face Detection network. 

If Age Gender Head Pose or Emotion detections are enabled the additional info below is reported also:
* **Age Gender + Head Pose + Emotions Detection time**: combined inference time of simultaneously executed
age gender, head pose and emotion recognition networks.

## See Also
* [Using Inference Engine Samples](@ref SamplesOverview)
