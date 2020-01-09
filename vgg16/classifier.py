#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class Classifier(object):

    def __init__(self, overall_model_path=None, 
                model_input_name="input_1", 
                model_output_name="predictions/Softmax",
                input_shape=(3, 224, 224),
                labels=["normal", "abnormal"]):
        self.model_path = overall_model_path
        self.input_shape = input_shape
        self.model_dtype = trt.float32
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.model_input_name = model_input_name
        self.model_output_name = model_output_name
        self.labels = labels
        self.engine = self.build_engine_uff()
        self.tmp_buffers = self.allocate_buffers()

    @staticmethod
    def GiB(val):
        return val * 1 << 30

    def allocate_buffers(self):
        h_input = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)),
                                        dtype=trt.nptype(self.model_dtype))
        h_output = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)),
                                         dtype=trt.nptype(self.model_dtype))
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        stream = cuda.Stream()
        return [h_input, d_input, h_output, d_output, stream]

    @staticmethod
    def do_inference(context, h_input, d_input, h_output, d_output, stream):
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # Run inference.
        context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()

    def build_engine_uff(self):
        with trt.Builder(self.trt_logger) as builder, builder.create_network() as network, trt.UffParser() as parser:
            builder.max_workspace_size = self.GiB(1)
            parser.register_input(self.model_input_name, self.input_shape)
            parser.register_output(self.model_output_name)
            parser.parse(self.model_path, network)
            return builder.build_cuda_engine(network)

    def process_image(self, matrix, pagelocked_buffer):
        def normalize_image(matrix):
            c, h, w = self.input_shape
            return np.asarray(cv2.resize(matrix, (w, h))).transpose([2, 0, 1]).astype(
                trt.nptype(self.model_dtype)).ravel()

        return np.copyto(pagelocked_buffer, normalize_image(matrix))

    def predict(self, matrix):
        t0 = time.time()
        h_input, d_input, h_output, d_output, stream = self.tmp_buffers
        with self.engine.create_execution_context() as context:
            self.process_image(matrix, h_input)
            self.do_inference(context, h_input, d_input, h_output, d_output, stream)
            index = np.argmax(h_output)
            pred = self.labels[index]

        print(pred)
        if pred == "abnormal":
            return True, h_output[index]
        else:
            return False, h_output[index]