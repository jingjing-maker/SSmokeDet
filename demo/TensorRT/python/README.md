# SSmokeNet-TensorRT in Python

This tutorial includes a Python demo for TensorRT.

## Install TensorRT Toolkit

Please follow the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) and [torch2trt gitrepo](https://github.com/NVIDIA-AI-IOT/torch2trt) to install TensorRT and torch2trt.

## Convert model

SSmokeNet models can be easily conveted to TensorRT models using torch2trt

   If you want to convert our model, use the flag -n to specify a model name:
   ```shell
   python tools/trt.py -n <SSmokeNet_MODEL_NAME> -c <SSmokeNet_CHECKPOINT>
   ```
   For example:
   ```shell
   python tools/trt.py -n SSmokeNet-s -c your_ckpt.pth
   ```
   <SSmokeNet_MODEL_NAME> can be: SSmokeNet-nano, SSmokeNet-tiny. SSmokeNet-s, SSmokeNet-m, SSmokeNet-l, SSmokeNet-x.

   If you want to convert your customized model, use the flag -f to specify you exp file:
   ```shell
   python tools/trt.py -f <SSmokeNet_EXP_FILE> -c <SSmokeNet_CHECKPOINT>
   ```
   For example:
   ```shell
   python tools/trt.py -f /path/to/your/SSmokeNet/exps/SSmokeNet_s.py -c your_ckpt.pth
   ```
   *SSmokeNet_s.py* can be any exp file modified by you.

The converted model and the serialized engine file (for C++ demo) will be saved on your experiment output dir.  

## Demo

The TensorRT python demo is merged on our pytorch demo file, so you can run the pytorch demo command with ```--trt```.

```shell
python tools/demo.py image -n SSmokeNet-s --trt --save_result
```
or
```shell
python tools/demo.py image -f exps/default/SSmokeNet_s.py --trt --save_result
```
