
在网上关于TensorRT的资料很杂乱，这里是我自己在GTX1070和GTX1080ti上做测试的结果。包括tensorflow(keras)和tensorrt性能对比，TensorRT如何多线程运行。模型转化的方法。

## 1、配置

    ubuntu 16.04
    TensorRT 5.0.2.6
    python 3.5
    cuda 9.0
    cudnn 7.3.0

## 2、性能

使用keras版本和使用tensorrt加速后的FPS对比如下表。

GTX1070

|  模型类别  | keras(tensorflow)    | TensorRT |
| :-------: | :------------------- | --------- |
| vgg16 | 58 | 87 |
| yolov3 | 13 | 18 |

GTX1080ti

|  模型类别  | keras(tensorflow)    | TensorRT |
| :-------: | :------------------- | --------- |
| vgg16 | 95 | 136 |
| yolov3 | 17 | 23 |

性能上面提高的不是很多，我后来测试发现不管是使用tensorflow还是tensorrt，他们在推理上面消耗的时间基本是差不多的，唯一的区别就是在处理数据上面需要的时间不同，下面是我在GTX1070上面用vgg16消耗时间表，可以看出大部分消耗的时间是在数据处理上面，我们可以在数据处理上做进一步的优化，以提高速度。

|  类型  | 耗时（ms）    |
| :-------: | :-------------------: |
| 准备数据 | 6 | 
| 推理 | 18 |
|处理结果<br>（将矩阵转化为class name）| 26|
|画图|1|
|总时间|51| 

## 3、运行

### 3.1、yolov3

#### 3.1.1、将darknet模型转化为onnx模型, 打开 yolov3/darknet_to_onnx.py, 修改main方法中的参数,然后运行它
	python yolov3_to_onnx.py

#### 3.1.2、将onnx模型转化为trt模型, 打开 yolov3/onnx_to_tensorrt.py, 修改main方法中的参数,运行
	python onnx_to_tensorrt.py

#### 3.1.3、多线程执行 tensorrt 推理

	python multi_thread_detect.py --mode yolov3

关键代码是 31行到43行的代码

	cuda_ctx = cuda.Device(0).make_context()    # important
	
	""" 创建模型类的方法必须要在线程的run方法中 """
	if self.mode == "yolov3":
	    model = Yolov3TRT()
	elif self.mode == "vgg16":
	    model = VGG16(overall_model_path="vgg16.uff")
	
	matrix = self.queue.get(block=True)
	
	# print detection info
	model.predict(matrix)
	
	cuda_ctx.pop()   # important
	del cuda_ctx     # important

### 3.2、vgg16

#### 3.2.1、.h5文件转.pb文件的方法

使用的是vgg16/h5_2_pb.py文件，需要注意的是.h5文件最好是model文件，如果是weights，挺麻烦的
    
    python h5_2_pb.py

#### 3.2.2、将pb文件转为uff文件，需要你安装好tensorRT的基本包，在terminal下直接运行下面的命令即可

    convert-to-uff [your_pb_model]

转化之后会在终端打印出uff模型的input节点和output节点的名称，要运行除vgg16以外的模型的时候只需要在创建Classifier的时候输入所需的model_input_name、model_output_name、input_shape和labels即可。

#### 3.2.3、执行vgg16的tensorRT推理

    python multi_thread_detect.py --mode vgg16


## 4、链接
1、[安装cuda和cudnn的方法](https://blog.csdn.net/qq_20265187/article/details/89029011 "安装cuda和cudnn的方法")  

2、[安装TensorRT的方法](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/#installing-tar "安装TensorRT的方法"),这里使用的是tar包安装 

