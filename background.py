"""Замена фона: человек вырезается сегментацией MediaPipe Selfie (landscape, 256×144, Apache-2.0) через OpenCV DNN.

Маска сглаживается по времени (без мерцания краёв) и смягчается по краю; фон — размытие, картинка/GIF/видео
или зелёный для хромакея в OBS.
"""
import cv2
import numpy as np

SEG_SIZE = (256, 144)
GREEN = (64, 177, 0)  # BGR: классический хромакей


class Segmenter:
    def __init__(self, model):
        engine = getattr(cv2.dnn, "ENGINE_CLASSIC", None)
        self.net = (cv2.dnn.readNetFromTFLite(str(model), engine=engine) if engine is not None
                    else cv2.dnn.readNetFromTFLite(str(model)))
        self.outs = self.net.getUnconnectedOutLayersNames()
        self.prev = None

    def mask(self, img):
        """Вероятность «человек» в низком разрешении (144×256), сглаженная по кадрам."""
        self.net.setInput(cv2.dnn.blobFromImage(img, 1 / 255.0, SEG_SIZE, swapRB=True))
        m = np.squeeze(self.net.forward(self.outs)[0]).astype(np.float32)
        if self.prev is not None and self.prev.shape == m.shape:
            # там, где маска резко меняется (движение), верим новому кадру; на месте — сглаживаем дрожь краёв
            k = np.clip(np.abs(m - self.prev) * 2.5, 0.45, 1.0)
            m = self.prev + (m - self.prev) * k
        self.prev = m
        return m


def cover(img, w, h):
    """Картинка под размер кадра с обрезкой лишнего, без искажения пропорций."""
    ih, iw = img.shape[:2]
    k = max(w / iw, h / ih)
    nw, nh = max(w, round(iw * k)), max(h, round(ih * k))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    x, y = (nw - w) // 2, (nh - h) // 2
    return r[y:y + h, x:x + w]


def blurred(frame, strength):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)  # фон всё равно мыльный
    small = cv2.GaussianBlur(small, (0, 0), 0.8 + strength * 0.06)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def composite(frame, mask, bg):
    """Человек из frame, остальное — из bg (той же формы). Работает прямо в frame."""
    h, w = frame.shape[:2]
    # край считаем в половинном разрешении — в 3–4 раза быстрее, а на глаз разницы нет
    m = cv2.resize(mask, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_LINEAR)
    m = np.clip((m - 0.3) * 2.5, 0.0, 1.0)   # край сегментации размытый — делаем его чётче
    m = m * m * (3 - 2 * m)                  # плавная ступенька вместо резкой
    m = cv2.GaussianBlur(m, (0, 0), max(0.8, w / 1800))
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    frame[:] = cv2.blendLinear(frame, bg, m, 1.0 - m)
