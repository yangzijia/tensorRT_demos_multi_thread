#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import sys, os

TRT_LOGGER = trt.Logger()

def get_engine(onnx_file_path, engine_file_path=""):
    """如果已经有序列化engine，则直接用，否则构建新的tensorrt engine然后保存."""
    # 闭包
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder,\
              builder.create_network() as network, \
              trt.OnnxParser(network, TRT_LOGGER) as parser:

            builder.max_workspace_size = 1 << 30 # 1GB
            builder.max_batch_size = 1

            # 解析模型文件
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')

            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")

            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    return build_engine()


def main():
    onnx_file_path = 'classes3/classes3.onnx'
    engine_file_path = 'classes3/classes3.trt'
    get_engine(onnx_file_path, engine_file_path)
    

if __name__ == '__main__':
    main()