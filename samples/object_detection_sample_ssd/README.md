# Object Detection Sample SSD {#InferenceEngineObjectDetectionSSDSampleApplication}

This topic demonstrates how to run the Object Detection sample application, which does inference using object detection 
networks like SSD-VGG on Intel® Processors and Intel® HD Graphics.

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```sh
./object_detection_sample_ssd -h
InferenceEngine: 
    API version ............ <version>
    Build .................. <number>

object_detection_sample_ssd [OPTION]
Options:

    -h                      
                            Print a usage message.
    -i "<path>"
                            Required. Path to an image file.
    -m "<path>"             
                            Required. Path to an .xml file with a trained model.
        -l "<absolute_path>"    
                            Optional. Absolute path to library with MKL-DNN (CPU) custom layers (*.so).
        Or
        -c "<absolute_path>"
                            Optional. Absolute path to clDNN (GPU) custom layers config (*.xml).
    -pp "<path>"            
                            Path to a plugin folder.
    -d "<device>"           
                            Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified

    -ni "<integer>"         
                            Number of iterations (default 1)
    -pc                     
                            Enables per-layer performance report

```

Running the application with the empty list of options yields the usage message given above and an error message.

You can use the following command to do inference on Intel&reg; Processors on an image using a trained SSD network:
```sh
./object_detection_sample_ssd -i <path_to_image>/inputImage.bmp -m <path_to_model>/VGG_ILSVRC2016_SSD.xml -d CPU
```

### Outputs

The application outputs an image (<code>out_0.bmp</code>) with detected objects enclosed in rectangles. It outputs the list of classes 
of the detected objects along with the respective confidence values and the coordinates of the 
rectangles to the standard output stream.

### How it works

Upon the start-up the sample application reads command line parameters and loads a network and an image to the Inference 
Engine plugin. When inference is done, the application creates an 
output image and outputs data to the standard output stream.

## See Also 
* [Using Inference Engine Samples](@ref SamplesOverview)
