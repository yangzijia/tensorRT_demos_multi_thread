## 1、配置

    TensorRT 5.0.2.6
    python 3.5
    cuda 9.0
    cudnn 7.3.0

## 2、运行

### yolov3

#### 将darknet模型转化为onnx模型, 打开 yolov3/darknet_to_onnx.py, 修改main方法中的参数,然后云溪它
	python yolov3_to_onnx.py

#### 将onnx模型转化为trt模型, 打开 yolov3/onnx_to_tensorrt.py, 修改main方法中的参数,运行
	python onnx_to_tensorrt.py

#### 多线程执行 tensorrt 推理
	python multi_thread_detect.py

关键代码是 31行到43行的代码

	cuda_ctx = cuda.Device(0).make_context()    # important

    """  """
    if self.mode == "yolov3":
        model = Yolov3TRT()
    elif self.mode == "vgg16":
        model = VGG16(overall_model_path="vgg16.uff")

    matrix = self.queue.get(block=True)

    # print detection info
    model.predict(matrix)

    cuda_ctx.pop()   # important
    del cuda_ctx     # important

### vgg16



## 3、链接
1 安装cuda cudnn 的方法 https://blog.csdn.net/qq_20265187/article/details/89029011

2 安装TensorRT的方法,这里使用的是tar包安装 https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/#installing-tar

