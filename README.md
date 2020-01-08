## 配置

    TensorRT 5.0.2.6
    python 3.5
    cuda 9.0
    cudnn 7.3.0

## 1 将darknet模型转化为onnx模型, 打开 yolov3_to_onnx.py, 修改main方法中的参数,然后云溪它
	python yolov3_to_onnx.py

## 2 将onnx模型转化为trt模型, 打开onnx_to_tensorrt.py, 修改main方法中的参数,运行
	python onnx_to_tensorrt.py

## 3 多线程执行 tensorrt 推理
	python multi_thread_detect.py

关键代码是 29行到38行的代码

	cuda_ctx = cuda.Device(0).make_context()
    yolo = Yolov3TRT() # create class
    matrix = self.queue.get(block=True)

    # print detection info
    yolo.predict(matrix)

    cuda_ctx.pop()
    del cuda_ctx


