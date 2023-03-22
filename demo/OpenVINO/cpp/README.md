# SSmokeNet-OpenVINO in C++

This tutorial includes a C++ demo for OpenVINO, as well as some converted models.

### Download OpenVINO models.

| Model | Parameters | GFLOPs | Test Size | mAP | Weights |
|:------| :----: | :----: | :---: | :---: | :---: |
|  [SSmokeNet-Nano](../../../exps/default/nano.py) |  0.91M  | 1.08 | 416x416 | 25.8 | [github](https://github.com/Megvii-BaseDetection/SSmokeNet/releases/download/0.1.1rc0/SSmokeNet_nano_openvino.tar.gz) |
|  [SSmokeNet-Tiny](../../../exps/default/SSmokeNet_tiny.py) | 5.06M     | 6.45 | 416x416 |32.8 | [github](https://github.com/Megvii-BaseDetection/SSmokeNet/releases/download/0.1.1rc0/SSmokeNet_tiny_openvino.tar.gz) |
|  [SSmokeNet-S](../../../exps/default/SSmokeNet_s.py) | 9.0M | 26.8 | 640x640 |40.5 | [github](https://github.com/Megvii-BaseDetection/SSmokeNet/releases/download/0.1.1rc0/SSmokeNet_s_openvino.tar.gz) |
|  [SSmokeNet-M](../../../exps/default/SSmokeNet_m.py) | 25.3M | 73.8 | 640x640 |47.2 | [github](https://github.com/Megvii-BaseDetection/SSmokeNet/releases/download/0.1.1rc0/SSmokeNet_m_openvino.tar.gz) |
|  [SSmokeNet-L](../../../exps/default/SSmokeNet_l.py) | 54.2M | 155.6 | 640x640 |50.1 | [github](https://github.com/Megvii-BaseDetection/SSmokeNet/releases/download/0.1.1rc0/SSmokeNet_l_openvino.tar.gz) |
|  [SSmokeNet-Darknet53](../../../exps/default/yolov3.py) | 63.72M | 185.3 | 640x640 |48.0 | [github](https://github.com/Megvii-BaseDetection/SSmokeNet/releases/download/0.1.1rc0/SSmokeNet_dark_openvino.tar.gz) | 
|  [SSmokeNet-X](../../../exps/default/SSmokeNet_x.py) | 99.1M | 281.9 | 640x640 |51.5 | [github](https://github.com/Megvii-BaseDetection/SSmokeNet/releases/download/0.1.1rc0/SSmokeNet_x_openvino.tar.gz) |

## Install OpenVINO Toolkit

Please visit [Openvino Homepage](https://docs.openvinotoolkit.org/latest/get_started_guides.html) for more details.

## Set up the Environment

### For Linux

**Option1. Set up the environment tempororally. You need to run this command everytime you start a new shell window.**

```shell
source /opt/intel/openvino_2021/bin/setupvars.sh
```

**Option2. Set up the environment permenantly.**

*Step1.* For Linux:
```shell
vim ~/.bashrc 
```

*Step2.* Add the following line into your file:

```shell
source /opt/intel/openvino_2021/bin/setupvars.sh
```

*Step3.* Save and exit the file, then run:

```shell
source ~/.bashrc
```


## Convert model

1. Export ONNX model
   
   Please refer to the [ONNX toturial](../../ONNXRuntime). **Note that you should set --opset to 10, otherwise your next step will fail.**

2. Convert ONNX to OpenVINO 

   ``` shell
   cd <INSTSLL_DIR>/openvino_2021/deployment_tools/model_optimizer
   ```

   Install requirements for convert tool

   ```shell
   sudo ./install_prerequisites/install_prerequisites_onnx.sh
   ```

   Then convert model.
   ```shell
   python3 mo.py --input_model <ONNX_MODEL> --input_shape <INPUT_SHAPE> [--data_type FP16]
   ```
   For example:
   ```shell
   python3 mo.py --input_model SSmokeNet_tiny.onnx --input_shape [1,3,416,416] --data_type FP16
   ```  

   Make sure the input shape is consistent with [those](SSmokeNet_openvino.cpp#L24-L25) in cpp file. 

## Build 

### Linux
```shell
source /opt/intel/openvino_2021/bin/setupvars.sh
mkdir build
cd build
cmake ..
make
```

## Demo

### c++

```shell
./SSmokeNet_openvino <XML_MODEL_PATH> <IMAGE_PATH> <DEVICE>
```
