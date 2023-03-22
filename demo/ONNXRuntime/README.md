## SSmokeNet-ONNXRuntime in Python

This doc introduces how to convert your pytorch model into onnx, and how to run an onnxruntime demo to verify your convertion.

### Download ONNX models.

| Model | Parameters | GFLOPs | Test Size | mAP | Weights |
|:------| :----: | :----: | :---: | :---: | :---: |
|  SSmokeNet-Nano |  0.91M  | 1.08 | 416x416 | 25.8 |[github](https://github.com/Megvii-BaseDetection/SSmokeNet/releases/download/0.1.1rc0/SSmokeNet_nano.onnx) |
|  SSmokeNet-Tiny | 5.06M     | 6.45 | 416x416 |32.8 | [github](https://github.com/Megvii-BaseDetection/SSmokeNet/releases/download/0.1.1rc0/SSmokeNet_tiny.onnx) |
|  SSmokeNet-S | 9.0M | 26.8 | 640x640 |40.5 | [github](https://github.com/Megvii-BaseDetection/SSmokeNet/releases/download/0.1.1rc0/SSmokeNet_s.onnx) |
|  SSmokeNet-M | 25.3M | 73.8 | 640x640 |47.2 | [github](https://github.com/Megvii-BaseDetection/SSmokeNet/releases/download/0.1.1rc0/SSmokeNet_m.onnx) |
|  SSmokeNet-L | 54.2M | 155.6 | 640x640 |50.1 | [github](https://github.com/Megvii-BaseDetection/SSmokeNet/releases/download/0.1.1rc0/SSmokeNet_l.onnx) |
|  SSmokeNet-Darknet53| 63.72M | 185.3 | 640x640 |48.0 | [github](https://github.com/Megvii-BaseDetection/SSmokeNet/releases/download/0.1.1rc0/SSmokeNet_darknet.onnx) |
|  SSmokeNet-X | 99.1M | 281.9 | 640x640 |51.5 | [github](https://github.com/Megvii-BaseDetection/SSmokeNet/releases/download/0.1.1rc0/SSmokeNet.onnx) |


### Convert Your Model to ONNX

First, you should move to <SSmokeNet_HOME> by:
```shell
cd <SSmokeNet_HOME>
```
Then, you can:

1. Convert a standard SSmokeNet model by -n:
```shell
python3 tools/export_onnx.py --output-name SSmokeNet_s.onnx -n SSmokeNet-s -c SSmokeNet_s.pth
```
Notes:
* -n: specify a model name. The model name must be one of the [SSmokeNet-s,m,l,x and SSmokeNet-nane, SSmokeNet-tiny, yolov3]
* -c: the model you have trained
* -o: opset version, default 11. **However, if you will further convert your onnx model to [OpenVINO](https://github.com/Megvii-BaseDetection/SSmokeNet/demo/OpenVINO/), please specify the opset version to 10.**
* --no-onnxsim: disable onnxsim
* To customize an input shape for onnx model,  modify the following code in tools/export.py:

    ```python
    dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])
    ```

2. Convert a standard SSmokeNet model by -f. When using -f, the above command is equivalent to:

```shell
python3 tools/export_onnx.py --output-name SSmokeNet_s.onnx -f exps/default/SSmokeNet_s.py -c SSmokeNet_s.pth
```

3. To convert your customized model, please use -f:

```shell
python3 tools/export_onnx.py --output-name your_SSmokeNet.onnx -f exps/your_dir/your_SSmokeNet.py -c your_SSmokeNet.pth
```

### ONNXRuntime Demo

Step1.
```shell
cd <SSmokeNet_HOME>/demo/ONNXRuntime
```

Step2. 
```shell
python3 onnx_inference.py -m <ONNX_MODEL_PATH> -i <IMAGE_PATH> -o <OUTPUT_DIR> -s 0.3 --input_shape 640,640
```
Notes:
* -m: your converted onnx model
* -i: input_image
* -s: score threshold for visualization.
* --input_shape: should be consistent with the shape you used for onnx convertion.
