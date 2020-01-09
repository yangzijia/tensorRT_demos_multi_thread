#!/usr/bin/env python
# -*- coding:utf-8 -*-

from yolov3.yolov3_trt import Yolov3TRT
from vgg16.classifier import Classifier
from framereader import FrameReader
import pycuda.driver as cuda
import argparse

from threading import Thread, Lock
from queue import Queue
import cv2
import os
import time

yolov3_settings = {
        "model_path": os.path.join('classes3', 'classes3.trt'),
        "anchors_path": os.path.join('classes3', 'train_anchors.txt'),
        "classes_path": os.path.join('classes3', 'voc_labels.txt'),
        "model_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],
        "score": 0.3,         # 对象覆盖的阈值，[0,1]之间
        "nms_threshold": 0.5,       # nms的阈值，[0,1]之间
        "model_image_size": (416, 416),
        "output_shapes": [(1, 24, 13, 13), (1, 24, 26, 26), (1, 24, 52, 52)]
    }

class MyThread(Thread):

    def __init__(self, mode=None):
        super().__init__()
        self.mode = mode
        self.queue = Queue(200)
        self.alive = True
        self.use_time = 0
        self.num = 0

    def append_to_queue(self, matrix):
        self.queue.put(matrix, block=True, timeout=5)

    def set_daemon_start(self):
        self.setDaemon(True)
        self.start()

    def run(self):
        try:
            cuda_ctx = cuda.Device(0).make_context()
            if self.mode == "yolov3":
                model = Yolov3TRT(**yolov3_settings)
            elif self.mode == "vgg16":
                model = Classifier(overall_model_path="vgg16.uff")

            while self.alive:
                matrix = self.queue.get(block=True)
                # print detection info
                t0 = time.time()
                model.predict(matrix)
                self.use_time += (time.time() - t0)
                self.num += 1

            cuda_ctx.pop()
            del cuda_ctx

        except Exception as e:
            print("ERROR", e)

    def stop(self):
        print("FPS: {:.3f}, use time {:.3f}".format(self.num / self.use_time, self.use_time/ self.num))
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
    address = "rtsp://admin:engyne123@192.168.1.64:554//Streaming/Channels/101"
    # cap = cv2.VideoCapture(address)
    frame_reader = FrameReader(uri=address)
    area_list = [851, 245, 1427, 807] 
    x1, y1, x2, y2 = area_list

    t = MyThread(mode=flags.mode)
    t.set_daemon_start()
    frame_reader.setDaemon(True)
    frame_reader.start()
    index = 0
    while True:
        # ret, frame = cap.read()
        frame  = frame_reader.get_matrix()
        # if not ret:
        #     break
        if frame is None:
            continue
        matrix = frame[y1:y2, x1:x2]
        
        index += 1
        if index in [6, 18]:
            t.append_to_queue(matrix)
        if index == 25:
            index = 0


        """ display frame """
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", matrix)

        if cv2.waitKey(1) & 0xff == 27:
            break

    # cap.release()
    frame_reader.stop()
    t.stop()