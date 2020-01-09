#!/usr/bin/env python
# -*- coding:utf-8 -*-

from threading import Thread, Lock
import cv2
import time


class FrameReader(Thread):
    def __init__(self, uri="0", rec_try=50, time_per_try=30, with_gs=False, latency=200, width=1920, height=1080, fps=25):
        Thread.__init__(self)
        self.uri = uri
        self.cap = None
        self.frame_lock = Lock()
        self.rec_try = rec_try
        self.time_per_try = time_per_try
        self.with_gs = with_gs
        self.latency = latency
        self.width = width
        self.height = height
        self.alive = True
        self.fps = fps
        self.matrix = None

    def get_cap(self):
        if self.with_gs:
            gs_str = (
                "rtspsrc location={} latency={} ! rtph264depay ! h264parse ! omxh264dec ! "
                "nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
                "videoconvert ! appsink").format(self.uri, self.latency, self.width, self.height)
            return cv2.VideoCapture(gs_str, cv2.CAP_GSTREAMER)
        else:
            return cv2.VideoCapture(self.uri)

    def get_matrix(self):
        return self.matrix

    def set_daemon_start(self):
        self.setDaemon(True)
        self.start()

    def set_area_list(self, area_list):
        self.area_list = area_list

    def run(self):
        try:
            self.cap = self.get_cap()
            self.width = self.cap.get(3)
            self.height = self.cap.get(4)

            while self.alive:
                ret, matrix = self.cap.read()
                if not ret:
                    print("ret is False.")
                    if self.reconnect() is False:
                        raise Exception("Lost frames and fail to reconnect.")
                    else:
                        # 又重新连接成功
                        print("uri: {} reconnect success, write success to file".format(self.uri))
                        # 返回上一次的状态
                        continue
                if matrix is not None:
                    self.matrix = matrix
                    
            # end while
        except Exception as e:
            # 发生异常时，通知调用者
            print(e)
            # self.on_exception()
            # con.log.exception(e)
        finally:
            if self.cap is not None:
                self.cap.release()
            print("FrameReader stopped.")

    def reconnect(self):
        try_count = 0
        recover = False
        print("uri {} connect fail.".format(self.uri))

        while not recover:
            self.cap = self.get_cap()
            st = time.time()
            while True:
                ret, frame = self.cap.read()
                # 重连成功，退出
                if ret:
                    recover = True
                    break
                time.sleep(10)
                use_time = time.time() - st + 10
                # 本次重连超时，退出
                if use_time >= self.time_per_try:
                    break
            if recover:
                break
            # end while
            try_count += 1
            print("{}th Reconnect fail ! uri is {}".format(try_count, self.uri))

            # 重连尝试次数已达到设定值，退出
            if try_count >= self.rec_try:
                break
        # end while
        return recover

    def stop(self):
        self.alive = False
        return
