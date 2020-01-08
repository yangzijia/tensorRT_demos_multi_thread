#!/usr/bin/env python
# -*- coding:utf-8 -*-

from yolov3_trt import Yolov3TRT
import pycuda.driver as cuda

from threading import Thread, Lock
from queue import Queue
import cv2


class MyThread(Thread):

    def __init__(self):
        super().__init__()
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
                yolo = Yolov3TRT()

                matrix = self.queue.get(block=True)

                # print detection info
                yolo.predict(matrix)

                cuda_ctx.pop()
                del cuda_ctx
            except Exception as e:
                print("ERROR", e)

    def stop(self):
        self.alive = False
        return


if __name__ == '__main__':

    # camera address
    cap = cv2.VideoCapture(0)

    t = MyThread()
    t.set_daemon_start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t.append_to_queue(frame)

        """ display frame """

    cap.release()
    t.stop()