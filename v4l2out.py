"""Вывод кадров в v4l2loopback без сторонних модулей: формат задаётся ioctl, кадры пишутся write().

Нужен в пакете для Arch: колесо pyvirtualcam собрано под одну версию Python и ломается при её обновлении.
"""
import fcntl
import os
import struct

import cv2

VIDIOC_S_FMT = 0xC0D05605           # _IOWR('V', 5, struct v4l2_format), sizeof == 208 на 64 битах
V4L2_BUF_TYPE_VIDEO_OUTPUT = 2
V4L2_FIELD_NONE = 1
V4L2_COLORSPACE_SRGB = 8
FOURCC_I420 = int.from_bytes(b"YU12", "little")  # такой же формат отдаёт pyvirtualcam — Discord и Chromium его берут


class Camera:
    def __init__(self, width, height, device):
        if not device:
            raise RuntimeError("нет устройства v4l2loopback")
        self.width, self.height = width - width % 2, height - height % 2
        self.device = device
        self.fd = os.open(device, os.O_RDWR)
        try:
            size = self.width * self.height * 3 // 2
            pix = struct.pack("12I", self.width, self.height, FOURCC_I420, V4L2_FIELD_NONE, self.width, size,
                              V4L2_COLORSPACE_SRGB, 0, 0, 0, 0, 0)
            fmt = struct.pack("I4x", V4L2_BUF_TYPE_VIDEO_OUTPUT) + pix.ljust(200, b"\0")
            fcntl.ioctl(self.fd, VIDIOC_S_FMT, bytearray(fmt))
        except OSError:
            os.close(self.fd)
            raise

    def send(self, bgr):
        if bgr.shape[1] != self.width or bgr.shape[0] != self.height:
            bgr = bgr[:self.height, :self.width]
        os.write(self.fd, cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).tobytes())

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
