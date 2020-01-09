#!/usr/bin/env python
# -*- coding:utf-8 -*-

from yolov3.yolov3_trt import Yolov3TRT
from vgg16.vgg16 import VGG16
import pycuda.driver as cuda
import argparse

from threading import Thread, Lock
from queue import Queue
import cv2


class MyThread(Thread):

    def __init__(self, mode=None):
        super().__init__()
        self.mode = mode
        self.queue = Queue(200)
        self.alive = True

    def append_to_queue(self, matrix):
        self.queue.put(matrix, block=True, timeout=5)

    def set_daemon_start(self):
        self.setDaemon(True)
        self.start()

    def run(self):
        while self.alive:
            try:
                cuda_ctx = cuda.Device(0).make_context()
                if self.mode == "yolov3":
                    model = Yolov3TRT()
                elif self.mode == "vgg16":
                    model = VGG16(overall_model_path="vgg16.uff")

                matrix = self.queue.get(block=True)

                # print detection info
                model.predict(matrix)

                cuda_ctx.pop()
                del cuda_ctx
            except Exception as e:
                print("ERROR", e)

    def stop(self):
        self.alive = False
        return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        '--mode', type=str, default="yolov3",
        help='vgg16 or yolov3'
    )

    flags = parser.parse_args()

    # camera address
    cap = cv2.VideoCapture(0)

    t = MyThread(mode=flags.mode)
    t.set_daemon_start()
    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        index += 1
        if index in [6, 18]:
            t.append_to_queue(frame)
        if index == 25:
            index = 0

        """ display frame """

    cap.release()
    t.stop()