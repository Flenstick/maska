#!/usr/bin/env python3
"""Маска — камера, которая прячет лицо: размытие, пикселизация, чёрная заливка или своя маска
из картинки, GIF или видео.

Лица ищет YuNet (OpenCV). Обработанный кадр можно отдать в виртуальную камеру
(v4l2loopback на Linux, виртуальная камера OBS на Windows) — тогда его видят Discord, OBS, браузер и т. д.
"""
import glob
import json
import math
import os
import shutil
import tempfile
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

# потоки OpenCV и OpenBLAS по умолчанию крутятся вхолостую в ожидании работы и отнимают ядра у игры, Discord
# и соседних потоков конвейера (замер: −20 % процессора при тех же fps). Задаётся до импорта cv2 и numpy
for _k, _v in (("OPENCV_THREAD_POOL_ACTIVE_WAIT_WORKER", "0"), ("OPENCV_THREAD_POOL_ACTIVE_WAIT_MAIN", "0"),
               ("OPENBLAS_NUM_THREADS", "1")):
    os.environ.setdefault(_k, _v)

import cv2
import numpy as np

import background
import effects
import face
import updater
from editor import EditDialog
from PyQt6.QtCore import (QEasingCurve, QPointF, QPropertyAnimation, QRectF, QSettings, QSize, QStandardPaths, Qt,
                          QThread, QTimer, pyqtProperty, pyqtSignal)
from PyQt6.QtGui import (QColor, QFont, QFontDatabase, QFontMetricsF, QIcon, QImage, QImageReader, QPainter, QPainterPath,
                         QPen)
from PyQt6.QtWidgets import (QAbstractButton, QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel,
                             QMainWindow, QMenu, QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget)

VERSION = "1.4.1"
IS_WIN = sys.platform == "win32"
APP_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))  # в сборке PyInstaller — папка распаковки
MODEL = APP_DIR / "models/face_detection_yunet_2023mar.onnx"
MESH_MODEL = APP_DIR / "models/face_landmark.tflite"
SEG_MODEL = APP_DIR / "models/selfie_segmentation_landscape.tflite"
if IS_WIN:
    DATA_DIR = Path(os.environ.get("APPDATA", Path.home())) / "Maska"
else:
    DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "maska"
MASKS_DIR = DATA_DIR / "masks"

BG, PANEL, PANEL_2, HOVER = "#0a0a0c", "#131316", "#1b1b1f", "#25252a"
LINE, LINE_2 = "#1f1f24", "#2e2e34"
INK, MUTED, DIM = "#f4f4f5", "#a1a1aa", "#6b6b74"
ACCENT, ACCENT_2, ACCENT_INK, RED = "#8fe03b", "#a6ef55", "#0c1600", "#f0445a"

COLORS = {"BG": BG, "PANEL": PANEL, "PANEL_2": PANEL_2, "HOVER": HOVER, "LINE_2": LINE_2, "INK": INK,
          "MUTED": MUTED, "ACCENT": ACCENT}

MODES = [("blur", "Размытие"), ("pixel", "Пиксели"), ("black", "Чёрный"), ("media", "Своя"), ("face", "Лицо"),
         ("off", "Выкл")]

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".heic", ".tif", ".tiff", ".jxl", ".svg"}
ANIM_EXT = {".gif", ".apng"}  # webp тоже бывает анимированным — его узнаёт QImageReader
VIDEO_EXT = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg"}
MEDIA_EXT = IMAGE_EXT | ANIM_EXT | VIDEO_EXT
MEDIA_FILTER = "Картинки, GIF и видео ({})".format(" ".join("*" + e for e in sorted(MEDIA_EXT)))


def shots_dir():
    pics = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation)
    return Path(pics or Path.home() / "Pictures") / "Маска"


# ───────────────────────────── обработка кадра ─────────────────────────────

class Tracker:
    """Ведёт лица между кадрами, чтобы маска не слетала.

    Детектор промахнулся (поворот головы, рука, смаз, темнота) — коробку двигает оптический поток
    по точкам внутри неё. Поток тоже потерялся — маска стоит на месте и расширяется.
    Всё в координатах уменьшенного кадра, где работает детектор.
    """
    FLOW_HOLD = 5.0   # сколько вести лицо без детектора, пока его держит поток
    BLIND_HOLD = 1.5  # без детектора и без потока
    STRONG = 0.88     # с такой уверенностью детектора лицо засчитывается сразу
    MAYBE_HOLD = 0.4  # непроверенная находка без детектора живёт недолго и потоком не ведётся
    REJECTS = 2       # столько раз сетка лица сказала «не лицо» — находку выбрасываем
    BAN = 3.0         # столько секунд неуверенные находки на месте отвергнутой не проверяются заново

    def __init__(self):
        # {"box": [x, y, w, h, наклон], "seen": t детектора, "flow": t потока,
        #  "ok": подтверждено, "best": лучшая уверенность детектора, "fake": сколько раз сетка отвергла}
        self.tracks = []
        self.banned = []  # [(box, до какого времени)] — места, где сетка сказала «не лицо»
        self.prev = None

    @staticmethod
    def iou(a, b):
        ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
        iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
        ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
        inter = iw * ih
        union = a[2] * a[3] + b[2] * b[3] - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def near(a, b):
        """Совпадение по центру: при резком движении IoU бывает нулевым, а лицо то же."""
        ca, cb = (a[0] + a[2] / 2, a[1] + a[3] / 2), (b[0] + b[2] / 2, b[1] + b[3] / 2)
        return abs(ca[0] - cb[0]) < max(a[2], b[2]) and abs(ca[1] - cb[1]) < max(a[3], b[3])

    def flow(self, gray, box):
        """Сдвиг коробки по оптическому потоку от прошлого кадра; None — точки потеряны."""
        H, W = gray.shape
        x, y, w, h = (int(v) for v in box[:4])  # пятое число — наклон головы
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        pts = cv2.goodFeaturesToTrack(self.prev[y0:y1, x0:x1], 60, 0.01, 3)
        if pts is None or len(pts) < 6:
            return None
        pts = pts + np.float32([x0, y0])
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, pts, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < 6:
            return None
        d = np.median((nxt - pts)[ok].reshape(-1, 2), axis=0)
        return float(d[0]), float(d[1])

    @staticmethod
    def smooth(track, box, now):
        """Рамка через One Euro (центр, размер, наклон в долях размера лица) — без дрожи на месте."""
        x, y, w, h, a = box
        size = track.setdefault("ref", max(w, h, 1.0))  # постоянный масштаб, иначе нормировка сама дрожит
        filt = track.setdefault("filter", face.OneEuro(min_cutoff=0.4, beta=8.0))
        cx, cy, fw, fh, fa = filt([(x + w / 2) / size, (y + h / 2) / size, w / size, h / size, a / 90.0], now) * \
            [size, size, size, size, 90.0]
        return [cx - fw / 2, cy - fh / 2, fw, fh, fa]

    def update(self, gray, boxes, now, scores=None):
        """Возвращает все дорожки; маскировать нужно только подтверждённые (t["ok"]), остальные — проверить сеткой."""
        scores = scores or [1.0] * len(boxes)
        self.banned = [b for b in self.banned if b[1] > now]
        free = list(self.tracks)
        for box, sc in zip(boxes, scores):
            if sc < self.STRONG and any(self.iou(b, box) > 0.3 for b, _ in self.banned):
                continue  # та же дверь или рука, что уже отвергли, — не тратим время на проверку
            best = max(free, key=lambda t: self.iou(t["box"], box), default=None)
            if best is None or (self.iou(best["box"], box) < 0.1 and not self.near(best["box"], box)):
                self.tracks.append({"box": list(box), "seen": now, "flow": now, "best": sc, "fake": 0,
                                    "ok": sc >= self.STRONG})
                continue
            free.remove(best)
            best["box"] = self.smooth(best, box, now)
            best["seen"] = best["flow"] = now
            best["best"] = max(best["best"], sc)
            best["ok"] = best["ok"] or sc >= self.STRONG
        if self.prev is not None and self.prev.shape == gray.shape:
            for t in free:
                if not t["ok"]:
                    continue
                d = self.flow(gray, t["box"])
                if d is not None:
                    t["box"][0] += d[0]
                    t["box"][1] += d[1]
                    t["flow"] = now
        self.prev = gray
        self.banned += [(t["box"], now + self.BAN) for t in self.tracks if t["fake"] >= self.REJECTS]
        self.tracks = [t for t in self.tracks if t["fake"] < self.REJECTS and (
            now - t["seen"] <= self.FLOW_HOLD and now - t["flow"] <= self.BLIND_HOLD if t["ok"]
            else now - t["seen"] <= self.MAYBE_HOLD)]
        # две дорожки на одном лице — оставляем подтверждённую, из них свежую
        self.tracks.sort(key=lambda t: (not t["ok"], -t["seen"]))
        kept = []
        for t in self.tracks:
            if not any(self.iou(t["box"], k["box"]) > 0.3 for k in kept):
                kept.append(t)
        self.tracks = kept
        return kept


def apply_mask(frame, boxes, mode, strength, pad, oval):
    h, w = frame.shape[:2]
    for x, y, bw, bh in boxes:
        px, py = bw * pad, bh * pad
        x0, y0 = int(max(0, x - px)), int(max(0, y - py * 1.3))  # сверху больше — волосы и лоб
        x1, y1 = int(min(w, x + bw + px)), int(min(h, y + bh + py))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        roi = frame[y0:y1, x0:x1]
        rh, rw = roi.shape[:2]
        if mode == "black":
            eff = np.zeros_like(roi)
        elif mode == "pixel":
            cells = max(3, int(24 - strength * 0.21))  # сила 100 → 3 клетки, 0 → 24
            small = cv2.resize(roi, (cells, max(1, round(cells * rh / rw))), interpolation=cv2.INTER_AREA)
            eff = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        else:
            f = 2 + strength * 0.3
            small = cv2.resize(roi, (max(1, int(rw / f)), max(1, int(rh / f))), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), 1.5 + strength * 0.03)
            eff = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR)
        if not oval:
            roi[:] = eff
            continue
        # мягкий овал считаем в маленьком виде и растягиваем — край размыт, разницы не видно, а быстрее в разы
        k = min(1.0, 96 / max(rw, rh))
        sw, sh = max(4, round(rw * k)), max(4, round(rh * k))
        m = np.zeros((sh, sw), np.float32)
        cv2.ellipse(m, (sw // 2, sh // 2), (int(sw * 0.46), int(sh * 0.46)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), max(0.5, min(sw, sh) * 0.04))
        m = cv2.resize(m, (rw, rh), interpolation=cv2.INTER_LINEAR)
        roi[:] = cv2.blendLinear(eff, roi, m, 1.0 - m)


# ───────────────────────────── свои маски ─────────────────────────────

MEDIA_SIDE = 512     # больше лицу в кадре не нужно, а память и время на масштабирование экономятся
MEDIA_FRAMES = 600   # предел кадров анимации в памяти


def qimage_to_bgra(img):
    img = img.convertToFormat(QImage.Format.Format_ARGB32)  # в памяти little-endian это B, G, R, A
    w, h = img.width(), img.height()
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    return np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine() // 4, 4)[:, :w].copy()


def fit_side(img, side=MEDIA_SIDE):
    h, w = img.shape[:2]
    k = side / max(h, w)
    if k >= 1:
        return img
    return cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)


def crop_frame(img, crop):
    """Кадрирование по доле кадра: crop = [x, y, w, h] от 0 до 1."""
    if not crop:
        return img
    h, w = img.shape[:2]
    x0, y0 = int(round(crop[0] * w)), int(round(crop[1] * h))
    x1, y1 = int(round((crop[0] + crop[2]) * w)), int(round((crop[1] + crop[3]) * h))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, max(x1, x0 + 2)), min(h, max(y1, y0 + 2))
    return img[y0:y1, x0:x1]


class MediaMask:
    """Кадр маски на момент t: картинка (в т. ч. с прозрачностью), GIF/анимированный WebP или видео по кругу.

    edit — правки из окна обрезки: {"crop": [x, y, w, h] в долях кадра, "start": с, "end": с}.
    """

    def __init__(self, path, edit=None, side=MEDIA_SIDE):
        self.path = Path(path)
        self.side = side
        edit = edit or {}
        self.crop = edit.get("crop")
        self.frames, self.ends, self.cap = [], None, None
        if self.path.suffix.lower() in VIDEO_EXT:
            self.cap = cv2.VideoCapture(str(self.path))
            ok, f = self.cap.read() if self.cap.isOpened() else (False, None)
            if not ok:
                raise ValueError("не удалось прочитать видео")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.first = max(0, int(round(float(edit.get("start") or 0) * self.fps)))
            end = edit.get("end")
            self.stop = int(round(float(end) * self.fps)) if end else (self.count if self.count > 0 else 0)
            if self.first:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.first)
                ok, f2 = self.cap.read()
                f = f2 if ok else f
            self.pos, self.last = self.first + 1, self._prep(f)
            self.alpha = False
            return
        reader = QImageReader(str(self.path))
        reader.setAutoTransform(True)  # поворот по EXIF у фото с телефона
        delays = []
        while len(self.frames) < MEDIA_FRAMES:
            img = reader.read()
            if img.isNull():
                break
            self.frames.append(fit_side(crop_frame(qimage_to_bgra(img), self.crop), side))
            delays.append(max(20, reader.nextImageDelay() or 100) / 1000)
            if not reader.supportsAnimation():
                break
        if not self.frames:
            raise ValueError(reader.errorString() or "не удалось прочитать файл")
        if len(self.frames) > 1 and (edit.get("start") or edit.get("end")):
            starts = np.concatenate([[0], np.cumsum(delays)[:-1]])
            t0, t1 = float(edit.get("start") or 0), float(edit.get("end") or 1e9)
            keep = [i for i, st in enumerate(starts) if t0 - 1e-3 <= st < t1] or [int(np.argmin(abs(starts - t0)))]
            self.frames = [self.frames[i] for i in keep]
            delays = [delays[i] for i in keep]
        self.ends = np.cumsum(delays)
        self.alpha = bool(self.frames[0][..., 3].min() < 250)

    def _prep(self, bgr):
        return cv2.cvtColor(fit_side(crop_frame(bgr, self.crop), self.side), cv2.COLOR_BGR2BGRA)

    @property
    def kind(self):
        return "video" if self.cap is not None else "anim" if len(self.frames) > 1 else "image"

    def frame(self, t):
        if self.cap is None:
            if len(self.frames) == 1:
                return self.frames[0]
            i = int(np.searchsorted(self.ends, t % self.ends[-1], side="right"))
            return self.frames[min(i, len(self.frames) - 1)]
        target = self.first + int(t * self.fps)
        if self.stop > self.first:
            target = self.first + (target - self.first) % (self.stop - self.first)
        if target < self.pos - 1 or target - self.pos > 90:  # назад или далеко вперёд — перемотка, а не чтение подряд
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.pos = target
        for _ in range(8):  # отстали — догоняем, но не больше нескольких кадров за раз
            if self.pos > target:
                break
            ok, f = self.cap.read()
            if not ok:  # конец ролика (или неверное число кадров) — с начала отрезка
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.first)
                if self.stop <= self.first:
                    self.stop = self.pos  # длительность не была известна — теперь известна
                self.pos = self.first
                break
            self.pos += 1
            self.last = self._prep(f)
        return self.last

    def close(self):
        if self.cap is not None:
            self.cap.release()


def thumbnail(path, size=128, edit=None):
    """Первый кадр маски как QImage для галереи."""
    try:
        m = MediaMask(path, edit)
        f = fit_side(m.frame(0), size)
        m.close()
    except Exception:  # noqa: BLE001 — битый файл покажем пустой плиткой
        return None
    h, w = f.shape[:2]
    f = np.ascontiguousarray(f)
    return QImage(f.data, w, h, w * 4, QImage.Format.Format_ARGB32).copy()


def apply_media(frame, box, media, alpha, pad, oval, angle=0.0, cover=True):
    """Накладывает кадр маски (BGRA) на область лица; angle — наклон головы в градусах."""
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    px, py = bw * pad, bh * pad  # сверху запас больше — волосы и лоб, как у apply_mask
    # холст — полная область маски; у края кадра он просто обрезается, а не сплющивается
    fx0, fy0 = int(round(x - px)), int(round(y - py * 1.3))
    cw, ch = int(round(bw + 2 * px)), int(round(bh + py * 2.3))
    x0, y0, x1, y1 = max(0, fx0), max(0, fy0), min(w, fx0 + cw), min(h, fy0 + ch)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    mh, mw = media.shape[:2]
    k = max(cw / mw, ch / mh) if cover else min(cw / mw, ch / mh)
    nw, nh = max(1, round(mw * k)), max(1, round(mh * k))
    scaled = cv2.resize(media, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((ch, cw, 4), np.uint8)
    ox, oy = (cw - nw) // 2, (ch - nh) // 2
    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    cpw, cph = min(nw - sx0, cw - dx0), min(nh - sy0, ch - dy0)
    canvas[dy0:dy0 + cph, dx0:dx0 + cpw] = scaled[sy0:sy0 + cph, sx0:sx0 + cpw]
    a = canvas[..., 3].astype(np.float32) / 255.0
    if oval and not alpha:
        m = np.zeros((ch, cw), np.float32)
        cv2.ellipse(m, (cw // 2, ch // 2), (int(cw * 0.46), int(ch * 0.46)), 0, 0, 360, 1.0, -1)
        a *= cv2.GaussianBlur(m, (0, 0), max(1.0, min(cw, ch) * 0.04))
    if abs(angle) > 1.0:
        rot = cv2.getRotationMatrix2D((cw / 2, ch / 2), -angle, 1.0)
        canvas = cv2.warpAffine(canvas, rot, (cw, ch), flags=cv2.INTER_LINEAR)
        a = cv2.warpAffine(a, rot, (cw, ch), flags=cv2.INTER_LINEAR)
    # из холста берём ту часть, что попала в кадр
    cx0, cy0 = x0 - fx0, y0 - fy0
    roi = frame[y0:y1, x0:x1]
    a = a[cy0:cy0 + y1 - y0, cx0:cx0 + x1 - x0, None]
    roi[:] = (canvas[cy0:cy0 + y1 - y0, cx0:cx0 + x1 - x0, :3] * a + roi * (1.0 - a)).astype(np.uint8)


class Library:
    """Свои маски: файлы копируются в папку программы, список с именами — masks.json."""

    def __init__(self):
        self.index = MASKS_DIR / "masks.json"
        try:
            self.items = json.loads(self.index.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.items = []
        self.items = [m for m in self.items if (MASKS_DIR / m["file"]).exists()]

    def save(self):
        MASKS_DIR.mkdir(parents=True, exist_ok=True)
        self.index.write_text(json.dumps(self.items, ensure_ascii=False, indent=1), encoding="utf-8")

    def get(self, mid):
        return next((m for m in self.items if m["id"] == mid), None)

    def path(self, mid):
        m = self.get(mid)
        return MASKS_DIR / m["file"] if m else None

    def spec(self, mid):
        """(путь, правки в JSON) — по нему поток камеры понимает, что маску надо перечитать."""
        m = self.get(mid)
        return (str(MASKS_DIR / m["file"]), json.dumps(m.get("edit") or {}, sort_keys=True)) if m else None

    def set_edit(self, mid, edit):
        m = self.get(mid)
        if m is not None:
            if edit:
                m["edit"] = edit
            else:
                m.pop("edit", None)
            self.save()

    def add(self, src):
        src = Path(src)
        mid = uuid.uuid4().hex[:12]
        MASKS_DIR.mkdir(parents=True, exist_ok=True)
        # имя латиницей: OpenCV на Windows не всегда открывает видео по пути с кириллицей
        shutil.copy2(src, MASKS_DIR / (mid + src.suffix.lower()))
        self.items.append({"id": mid, "name": src.stem, "file": mid + src.suffix.lower()})
        self.save()
        return mid

    def remove(self, mid):
        p = self.path(mid)
        self.items = [m for m in self.items if m["id"] != mid]
        self.save()
        if p:
            p.unlink(missing_ok=True)


# ───────────────────────────── камеры ─────────────────────────────

def loopback_device():
    """Выход v4l2loopback: сначала наш «Маска», иначе любой."""
    if IS_WIN:
        return None
    found = None
    for dev in glob.glob("/sys/class/video4linux/video*"):
        try:
            name = Path(dev, "name").read_text().strip()
        except OSError:
            continue
        if name == "Маска":
            return "/dev/" + Path(dev).name
        if "loopback" in Path(dev, "device").resolve().as_posix() or name.startswith("Dummy"):
            found = found or "/dev/" + Path(dev).name
    return found


def list_cameras():
    """Узлы захвата видео: /sys/.../index == 0 (у вебок второй узел — метаданные), без виртуальных выходов."""
    if IS_WIN:
        return list_cameras_win()
    cams = []
    for dev in sorted(glob.glob("/sys/class/video4linux/video*"), key=lambda p: int(p.rsplit("video", 1)[1])):
        try:
            if Path(dev, "index").read_text().strip() != "0":
                continue
            name = Path(dev, "name").read_text().strip()
        except OSError:
            continue
        if "loopback" in Path(dev, "device").resolve().as_posix() or name.startswith(("Маска", "Dummy")):
            continue
        cams.append((int(dev.rsplit("video", 1)[1]), name))
    return cams


def list_cameras_win():
    """Камеры DirectShow: их порядок совпадает с индексами cv2.CAP_DSHOW. Свои виртуальные выходы прячем."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
    except Exception:  # noqa: BLE001 — нет pygrabber или COM — просто пронумеруем
        names = [f"Камера {i + 1}" for i in range(4)]
    return [(i, n) for i, n in enumerate(names)
            if not any(v in n for v in ("OBS Virtual", "Unity Video Capture", "Маска"))]


def open_vcam(w, h):
    """Виртуальная камера: Linux — v4l2loopback напрямую, Windows — камера OBS через pyvirtualcam."""
    if not IS_WIN:
        import v4l2out
        try:
            return v4l2out.Camera(w, h, loopback_device())
        except OSError:
            try:  # запасной путь, если pyvirtualcam установлен (запуск из исходников)
                import pyvirtualcam
            except ImportError:
                raise
            return pyvirtualcam.Camera(w, h, 60, fmt=pyvirtualcam.PixelFormat.BGR, backend="v4l2loopback",
                                       device=loopback_device())
    import pyvirtualcam
    return pyvirtualcam.Camera(w, h, 60, fmt=pyvirtualcam.PixelFormat.BGR)


# как часто обновлять вырезание человека для замены фона, с
SEG_EVERY = 1 / 32

# поворот головы (face.turn), начиная с которого под чужим лицом размывается вся голова
TURN_HIDE = 0.3


class Grabber(threading.Thread):
    """Читает камеру в своём потоке: распаковка MJPG не тормозит обработку, а она всегда берёт свежий кадр.

    Тёмный кадр (камера держит частоту, и выдержки не хватает) осветляется гаммой. Из видеофайла
    (MASKA_SOURCE) кадры не пропускаются — поток ждёт, пока обработка заберёт прошлый, и крутит файл по кругу.
    """

    def __init__(self, cap, from_file):
        super().__init__(daemon=True)
        self.cap, self.from_file = cap, from_file
        self.cond = threading.Condition()
        self.img, self.alive, self.running = None, True, True
        self.gamma, self.lut, self.n = 1.0, None, 0

    def lift(self, img):
        self.n += 1
        if self.n % 6 == 1:  # яркость меняется медленно — мерить каждый кадр незачем
            mean = float(cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA).mean())
            want = 1.0 if mean >= 80 else max(0.45, math.log(95 / 255) / math.log(max(mean, 6) / 255))
            gamma = self.gamma + (want - self.gamma) * 0.3
            if abs(gamma - self.gamma) > 0.01 or (self.lut is None and gamma < 0.97):
                self.gamma = gamma
                self.lut = (None if gamma > 0.97 else
                            np.clip((np.arange(256) / 255.0) ** gamma * 255, 0, 255).astype(np.uint8))
        return img if self.lut is None else cv2.LUT(img, self.lut)

    def run(self):
        while self.running:
            ok, img = self.cap.read() if self.cap.isOpened() else (False, None)
            if not ok and self.from_file and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # файл кончился — по кругу
                continue
            if ok and not self.from_file:
                img = self.lift(img)
            with self.cond:
                while self.from_file and self.running and self.img is not None:
                    self.cond.wait(0.1)
                self.img, self.alive = (img, True) if ok else (None, False)
                self.cond.notify_all()
            if not ok:
                return

    def get(self, timeout=1.0):
        """Свежий кадр; None — кадра пока нет или камера отвалилась (тогда alive == False)."""
        with self.cond:
            self.cond.wait_for(lambda: self.img is not None or not self.alive, timeout)
            img, self.img = self.img, None
            self.cond.notify_all()
            return img

    def stop(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()
        self.join(3)
        self.cap.release()


class Worker(QThread):
    frame = pyqtSignal(QImage)
    stats = pyqtSignal(float, int, bool)
    error = pyqtSignal(str)
    vcam_state = pyqtSignal(bool, str)
    face_state = pyqtSignal(str)  # "" — лицо с фото готово, иначе текст ошибки

    def __init__(self):
        super().__init__()
        self.device = 0
        self.mode = "blur"
        self.strength = 60
        self.pad = 0.25
        self.oval = True
        self.mirror = True
        self.failsafe = True
        self.want_vcam = False
        self.media_spec = None   # своя маска (путь, правки); меняется из окна, грузится в потоке камеры
        self.face_spec = None    # фото для замены лица (путь, правки)
        self.rotate = True       # своя маска наклоняется вместе с головой
        self.match_color = True  # замена лица: подгонять цвет кожи под кадр
        self.keep_mouth = True   # замена лица: рот изнутри — из кадра
        self.hide_head = False   # замена лица: при повороте размывать голову под лицом
        self.feather = 0.5       # замена лица: мягкость края 0..1
        self.effects = set()     # включённые эффекты (effects.EFFECTS)
        self.effect_opts = {"laser_color": "red", "laser_dir": "camera"}
        self.bg_mode = "off"     # фон: off, blur, media, green
        self.bg_spec = None      # картинка/GIF/видео для фона (путь, правки)
        self.bg_blur = 60
        self.low_light = False   # True — камере можно ронять кадры в темноте ради яркости
        self.snapshot_to = None
        self._restart = False
        self._running = True

    def switch_device(self, dev):
        self.device = dev
        self._restart = True

    def stop(self):
        self._running = False
        self.wait(2000)

    def open_cam(self):
        if os.environ.get("MASKA_SOURCE"):  # видеофайл вместо камеры — для проверки и скриншотов
            return cv2.VideoCapture(os.environ["MASKA_SOURCE"])
        if not IS_WIN:  # иначе в полутьме вебка сама опускает кадры до 5–15 в секунду ради выдержки
            import v4l2out
            v4l2out.keep_framerate(f"/dev/video{self.device}", not self.low_light)
        cap = cv2.VideoCapture(self.device, cv2.CAP_DSHOW if IS_WIN else cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)  # сколько камера умеет: не умеет 60 — драйвер даст ближайшее (обычно 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def set_low_light(self, on):
        """Настройка камеры меняется на ходу — переоткрывать её (и рвать картинку в Discord) не нужно."""
        self.low_light = on
        if not IS_WIN and not os.environ.get("MASKA_SOURCE"):
            import v4l2out
            v4l2out.keep_framerate(f"/dev/video{self.device}", not on)

    def start_grabber(self):
        g = Grabber(self.open_cam(), bool(os.environ.get("MASKA_SOURCE")))
        g.start()
        return g

    def run(self):
        """Конвейер из трёх потоков, чтобы держать частоту камеры (до 60 fps).

        Grabber читает и распаковывает кадры. Пока здесь рисуется кадр N (маски, лицо, фон), в пуле уже
        разбирается кадр N+1: детектор лиц и сегментация фона идут параллельно, потом сетка лица.
        Готовый кадр уходит в виртуальную камеру и превью тоже в своём потоке.
        """
        # три потока конвейера работают одновременно; если каждая операция OpenCV ещё и сама расползается
        # на все ядра, потоки толкаются и выходит медленнее — по 4 на операцию быстрее всего (замер: +30 %)
        cv2.setNumThreads(max(2, min(4, (os.cpu_count() or 4) // 2)))
        detector = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), 0.6, 0.3, 50)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        tracker = Tracker()
        grab = self.start_grabber()
        analyst, helper, sender = (ThreadPoolExecutor(1) for _ in range(3))
        job = sent = None  # разбор следующего кадра и вывод прошлого
        vcam = None
        media, media_spec, media_t0 = None, None, 0.0
        source, face_spec = None, None
        segmenter, bg_media, bg_spec, bg_t0 = None, None, None, 0.0
        bg_cache = (None, None, None)  # (кадр фона, размер, готовый фон) — картинка-фон не пересчитывается
        seg_job, seg_last, seg_t = None, None, 0.0  # вырезание человека идёт само по себе и не держит кадр
        try:
            mesh = face.FaceMesh(MESH_MODEL)
        except Exception as e:  # noqa: BLE001 — без сетки просто не будет проверки находок и замены лица
            mesh = None
            self.error.emit(f"Сетка лица не загрузилась: {e}")

        def analyze(img, want_pts):
            """Где лица, их точки и где человек — по чистому кадру, до любой обработки."""
            nonlocal tracker, segmenter, seg_job, seg_last, seg_t
            try:
                h, w = img.shape[:2]
                scale = min(1.0, 640 / w)
                small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img
                fresh = False
                if self.bg_mode == "off":
                    seg_job, seg_last = None, None
                else:
                    if segmenter is None:
                        try:
                            segmenter = background.Segmenter(SEG_MODEL)
                        except Exception as e:  # noqa: BLE001 — без модели фон просто не меняем
                            self.bg_mode = "off"
                            self.error.emit(f"Замена фона недоступна: {e}")
                    if seg_job is not None and seg_job.done():
                        seg_last, seg_job = seg_job.result(), None
                    # человек движется плавно, и маски 30 раз в секунду хватает; на 60 fps это вдвое меньше работы
                    if segmenter is not None and seg_job is None and time.monotonic() - seg_t >= SEG_EVERY:
                        seg_t = time.monotonic()
                        seg_job, fresh = helper.submit(segmenter.mask, small), True
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                detector.setInputSize((small.shape[1], small.shape[0]))
                _, faces = detector.detect(small)
                if faces is None:  # тёмный или пересвеченный кадр — второй проход с выравниванием яркости
                    eq = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
                    _, faces = detector.detect(eq)
                # пятое число — наклон головы по линии глаз (точки 4–7 у YuNet)
                boxes = [] if faces is None else [[float(v) for v in f[:4]] +
                                                   [math.degrees(math.atan2(f[7] - f[5], f[6] - f[4]))]
                                                   for f in faces]
                scores = [] if faces is None else [float(f[14]) for f in faces]
                now = time.monotonic()
                tracks = tracker.update(gray, boxes, now, scores)
                # находки, в которых детектор не уверен (дверь, рука, узор на обоях), проверяем сеткой лица:
                # у настоящего лица она уверена, у всего остального — нет
                for t in tracks:
                    if not t["ok"]:
                        if mesh is None or mesh.is_face(img, [v / scale for v in t["box"][:4]], t["box"][4]):
                            t["ok"] = True
                        else:
                            t["fake"] += 1
                live = []
                for t in tracks:
                    if not t["ok"]:
                        continue
                    pts = (face.track_points(mesh, img, [v / scale for v in t["box"][:4]], t["box"][4],
                                             t.setdefault("mesh", {}), now) if want_pts and mesh else None)
                    # снимок дорожки: пока этот кадр рисуется, трекер уже правит её для следующего
                    live.append({"rect": [v / scale for v in t["box"][:4]], "angle": t["box"][4],
                                 "lost": now - t["seen"], "pts": pts,
                                 "face": t.setdefault("face", {}), "fx": t.setdefault("fx", {})})
                if seg_job is not None and (fresh or seg_last is None):
                    # маску этого кадра ждём недолго: не успела — берём прошлую (сдвиг на кадр не заметен),
                    # а не роняем частоту кадров, когда процессор занят игрой или Discord
                    try:
                        seg_last, seg_job = seg_job.result(timeout=None if seg_last is None else 0.004), None
                    except FutureTimeout:
                        pass
                return {"now": now, "live": live, "seg": seg_last}
            except Exception as e:  # noqa: BLE001 — кадр всё равно закроется при отрисовке
                tracker = Tracker()  # сбрасываем дорожки, чтобы одна сломанная не держала кадр чёрным
                seg_job = None
                return e

        def render(img, info):
            """Рисует маски и лицо прямо в img. Возвращает (лиц, кадр закрыт целиком)."""
            h, w = img.shape[:2]
            now, live = info["now"], info["live"]
            covered = False
            mode = self.mode
            mframe = None
            if mode == "face" and source is None:
                mode = "blur"  # фото не выбрано или на нём нет лица — пока просто размываем
            if mode == "face":
                if self.failsafe and not live:
                    covered = True
                    apply_mask(img, [[0, 0, w, h]], "blur", 80, 0, False)
                for t in live:
                    pts = t["pts"]
                    turn = face.turn(pts) if pts is not None else 0.0
                    if self.hide_head and pts is not None and abs(turn) > TURN_HIDE:
                        # голова повёрнута: чужое лицо закрывает только видимую половину — остальную голову
                        # (щёку, ухо, волосы, профиль) размываем под ним, а само лицо остаётся
                        face.hide_head(img, pts, min(1.0, (abs(turn) - TURN_HIDE) / 0.25), turn)
                    if pts is None or not face.swap_face(img, pts, source, self.feather, self.match_color,
                                                         self.keep_mouth, t["face"]):
                        # сетка не легла (рука у лица, лицо мелкое) — прячем размытием, как обычно
                        apply_mask(img, [t["rect"]], "blur", 80, self.pad + min(0.6, t["lost"] * 0.4), True)
                mode = "off"
            if mode == "media":
                if media is None:
                    mode = "blur"  # маска не выбрана или не открылась — лицо всё равно прячем
                else:
                    mframe = media.frame(now - media_t0)
            if mode != "off":
                if self.failsafe and not live:
                    covered = True
                    if mframe is None:
                        apply_mask(img, [[0, 0, w, h]], mode, max(self.strength, 70), 0, False)
                    else:  # под маску с прозрачностью — чёрный фон, чтобы ничего не просвечивало
                        img[:] = 0
                        apply_media(img, [0, 0, w, h], mframe, media.alpha, 0, False, cover=not media.alpha)
                for t in live:
                    grow = min(0.6, t["lost"] * 0.4)  # чем дольше лицо не подтверждено, тем шире маска
                    if mframe is None:
                        apply_mask(img, [t["rect"]], mode, self.strength, self.pad + grow, self.oval)
                    else:
                        apply_media(img, t["rect"], mframe, media.alpha, self.pad + grow, self.oval,
                                    t["angle"] if self.rotate else 0.0, cover=not media.alpha)
            return len(live), covered

        def finish(img, info):
            """Фон и эффекты поверх готовых масок. Идёт в потоке вывода, пока рисуется следующий кадр."""
            nonlocal bg_cache
            h, w = img.shape[:2]
            now, live, seg_mask = info["now"], info["live"], info["seg"]
            if seg_mask is not None and self.bg_mode != "off":
                if self.bg_mode == "blur":
                    bg = background.blurred(img, self.bg_blur)
                elif self.bg_mode == "green":
                    if bg_cache[0] is not background.GREEN or bg_cache[1] != (w, h):
                        bg_cache = (background.GREEN, (w, h), np.empty_like(img))
                        bg_cache[2][:] = background.GREEN
                    bg = bg_cache[2]
                elif bg_media is not None:
                    bgf = bg_media.frame(now - bg_t0)
                    if bg_cache[0] is not bgf or bg_cache[1] != (w, h):  # картинка не меняется — считаем раз
                        a = bgf[..., 3:].astype(np.float32) / 255.0  # прозрачные места картинки-фона — чёрные
                        bg_cache = (bgf, (w, h), background.cover((bgf[..., :3] * a).astype(np.uint8), w, h))
                    bg = bg_cache[2]
                else:
                    bg = background.blurred(img, 60)  # фон не выбран — пока просто размываем
                background.composite(img, seg_mask, bg)
            if self.effects:
                for t in live:
                    if t["pts"] is not None:
                        effects.apply(img, t["pts"], t["fx"], now, self.effects, self.effect_opts)

        def output(img, info, vcam, mirror, snap):
            if info is not None:
                try:
                    finish(img, info)
                except Exception as e:  # noqa: BLE001 — лицо уже закрыто, но кадр без фона мог бы выдать комнату
                    img[:] = 0
                    self.error.emit(f"Ошибка обработки, кадр закрыт: {e}")
            if vcam is not None:
                vcam.send(img)
            if snap:
                snap.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(snap), img)
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(cv2.flip(img, 1) if mirror else img, cv2.COLOR_BGR2RGB)
            self.frame.emit(QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy())

        t_prev, fps = time.monotonic(), 0.0
        while self._running:
            ready = job.result() if job else None  # (кадр, разбор) — разбор шёл, пока рисовался прошлый
            job = None
            # всё, что трогает сетку лица, — только здесь, когда разбор кадра не идёт
            if self.media_spec != media_spec:
                media_spec = self.media_spec
                if media:
                    media.close()
                media = None
                if media_spec:
                    try:
                        media, media_t0 = MediaMask(media_spec[0], json.loads(media_spec[1])), time.monotonic()
                    except Exception as e:  # noqa: BLE001
                        self.error.emit(f"Маска не открылась: {e}")
            if self.bg_spec != bg_spec:
                if sent:
                    sent.result()  # прошлый кадр ещё дорисовывает фон из старого файла
                bg_spec = self.bg_spec
                if bg_media:
                    bg_media.close()
                bg_media = None
                if bg_spec:
                    try:
                        bg_media, bg_t0 = MediaMask(bg_spec[0], json.loads(bg_spec[1]), side=1280), time.monotonic()
                    except Exception as e:  # noqa: BLE001
                        self.error.emit(f"Фон не открылся: {e}")
            if self.face_spec != face_spec:
                face_spec, source = self.face_spec, None
                if face_spec:
                    try:
                        if mesh is None:
                            raise RuntimeError("сетка лица не загрузилась")
                        m = MediaMask(face_spec[0], json.loads(face_spec[1]), side=1024)
                        bgra = m.frame(0.0)
                        m.close()
                        a = bgra[..., 3:].astype(np.float32) / 255.0  # прозрачный фон — серым, детектору так проще
                        bgr = (bgra[..., :3] * a + 128 * (1 - a)).astype(np.uint8)
                        source = face.FaceSource(bgr, mesh, MODEL)
                        self.face_state.emit("")
                    except Exception as e:  # noqa: BLE001
                        self.face_state.emit(str(e))
            if self._restart:
                self._restart = False
                ready = None
                grab.stop()
                grab = self.start_grabber()
                tracker = Tracker()
                if sent:
                    sent.result()
                if vcam:
                    vcam.close()
                    vcam = None

            img = grab.get()
            if img is None and not grab.alive:
                self.error.emit("Камера не отвечает. Проверьте, не занята ли она другой программой.")
                grab.stop()
                self.msleep(500)
                grab = self.start_grabber()
                tracker = Tracker()
            elif img is not None:
                want_pts = mesh is not None and bool(self.effects or (self.mode == "face" and source is not None))
                job = analyst.submit(lambda im=img, wp=want_pts: (im, analyze(im, wp)))
            if ready is None:
                continue

            img, info = ready
            h, w = img.shape[:2]
            try:
                if isinstance(info, Exception):
                    raise info
                faces_n, covered = render(img, info)
                now = info["now"]
            except Exception as e:  # noqa: BLE001 — что бы ни сломалось, лицо не должно утечь
                img[:] = 0
                info = None
                covered, faces_n, now = True, 0, time.monotonic()
                self.error.emit(f"Ошибка обработки, кадр закрыт: {e}")

            if sent:
                sent.result()  # прошлый кадр ещё уходит — ждём, иначе очередь вывода будет расти
            if self.want_vcam and vcam is None:
                try:
                    vcam = open_vcam(w, h)
                    self.vcam_state.emit(True, vcam.device)
                except Exception as e:  # noqa: BLE001 — нет модуля ядра, занято и т. п.
                    self.want_vcam = False
                    self.vcam_state.emit(False, str(e))
            elif not self.want_vcam and vcam is not None:
                vcam.close()
                vcam = None
                self.vcam_state.emit(False, "")
            snap, self.snapshot_to = self.snapshot_to, None
            sent = sender.submit(output, img, info, vcam, self.mirror, snap)

            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = fps * 0.9 + (1 / dt) * 0.1 if fps else 1 / dt
            self.stats.emit(fps, faces_n, covered)
        if job:
            job.result()
        if sent:
            sent.result()
        for pool in (analyst, helper, sender):
            pool.shutdown()
        grab.stop()
        if vcam:
            vcam.close()
        if media:
            media.close()


# ───────────────────────────── виджеты ─────────────────────────────

class Preview(QWidget):
    def __init__(self):
        super().__init__()
        self.img = None
        self.message = "Подключаю камеру…"
        self.setMinimumSize(480, 300)

    def set_image(self, img):
        self.img, self.message = img, None
        self.update()

    def set_message(self, text):
        self.message = text
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        r = QRectF(self.rect())
        clip = QPainterPath()
        clip.addRoundedRect(r, 14, 14)
        p.fillPath(clip, QColor(PANEL))
        if self.img is not None:
            sz = self.img.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRectF((r.width() - sz.width()) / 2, (r.height() - sz.height()) / 2, sz.width(), sz.height())
            path = QPainterPath()
            path.addRoundedRect(target, 14, 14)
            p.setClipPath(path)
            p.drawImage(target, self.img)
            p.setClipping(False)
        if self.message:
            p.setPen(QColor(MUTED if self.img is None else INK))
            f = QFont("Maska Golos", 11)
            p.setFont(f)
            if self.img is not None:
                band = QRectF(16, r.height() - 52, r.width() - 32, 36)
                bp = QPainterPath()
                bp.addRoundedRect(band, 10, 10)
                p.fillPath(bp, QColor(12, 12, 14, 200))
                p.drawText(band, Qt.AlignmentFlag.AlignCenter, self.message)
            else:
                p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.message)


class Segmented(QWidget):
    changed = pyqtSignal(str)

    def __init__(self, items):
        super().__init__()
        self.items = items
        self.index = 0
        self.hover = -1
        self._x = 0.0
        self.anim = QPropertyAnimation(self, b"indicator", self)
        self.anim.setDuration(180)
        self.anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.setFixedHeight(34)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def get_ind(self):
        return self._x

    def set_ind(self, v):
        self._x = v
        self.update()

    indicator = pyqtProperty(float, get_ind, set_ind)

    @staticmethod
    def font():
        f = QFont("Maska Golos")
        f.setPixelSize(12)
        f.setWeight(QFont.Weight.DemiBold)
        return f

    def cells(self):
        """Ячейки по ширине подписей: «Размытие» шире «Выкл», и при равных ячейках подписи слипаются."""
        fm = QFontMetricsF(self.font())
        tw = [fm.horizontalAdvance(label) for _, label in self.items]
        extra = max(0.0, (self.width() - 6 - sum(tw)) / len(tw))
        out, x = [], 3.0
        for t in tw:
            out.append((x, t + extra))
            x += t + extra
        return out

    def cell_at(self, v):
        """Положение индикатора для дробного индекса — между соседними ячейками."""
        cs = self.cells()
        i = max(0, min(len(cs) - 1, int(v)))
        j = min(len(cs) - 1, i + 1)
        k = v - i
        return cs[i][0] + (cs[j][0] - cs[i][0]) * k, cs[i][1] + (cs[j][1] - cs[i][1]) * k

    def at(self, x):
        for i, (cx, cw) in enumerate(self.cells()):
            if x < cx + cw:
                return i
        return len(self.items) - 1

    def select(self, i, animate=True):
        self.index = i
        self.anim.stop()
        if animate:
            self.anim.setStartValue(self._x)
            self.anim.setEndValue(float(i))
            self.anim.start()
        else:
            self.set_ind(float(i))
        self.changed.emit(self.items[i][0])

    def mousePressEvent(self, e):
        i = self.at(e.position().x())
        if i != self.index:
            self.select(i)

    def mouseMoveEvent(self, e):
        i = self.at(e.position().x())
        if i != self.hover:
            self.hover = i
            self.update()

    def leaveEvent(self, _):
        self.hover = -1
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg = QPainterPath()
        bg.addRoundedRect(QRectF(self.rect()), 9, 9)
        p.fillPath(bg, QColor(PANEL_2))
        cs = self.cells()
        if 0 <= self.hover != self.index:
            hp = QPainterPath()
            hp.addRoundedRect(QRectF(cs[self.hover][0], 3, cs[self.hover][1], self.height() - 6), 7, 7)
            p.fillPath(hp, QColor(HOVER))
        ix, iw = self.cell_at(self._x)
        ind = QPainterPath()
        ind.addRoundedRect(QRectF(ix, 3, iw, self.height() - 6), 7, 7)
        p.fillPath(ind, QColor(ACCENT))
        p.setFont(self.font())
        for i, (_, label) in enumerate(self.items):
            closeness = max(0.0, 1 - abs(self._x - i))
            p.setPen(QColor(ACCENT_INK) if closeness > 0.5 else QColor(INK if i == self.hover else MUTED))
            p.drawText(QRectF(cs[i][0], 0, cs[i][1], self.height()), Qt.AlignmentFlag.AlignCenter, label)


class Toggle(QAbstractButton):
    def __init__(self, checked=False):
        super().__init__()
        self.setCheckable(True)
        self.setChecked(checked)
        self._pos = 1.0 if checked else 0.0
        self.anim = QPropertyAnimation(self, b"knob", self)
        self.anim.setDuration(160)
        self.anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)
        self.setFixedSize(38, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def get_knob(self):
        return self._pos

    def set_knob(self, v):
        self._pos = v
        self.update()

    knob = pyqtProperty(float, get_knob, set_knob)

    def _animate(self, on):
        self.anim.stop()
        self.anim.setStartValue(self._pos)
        self.anim.setEndValue(1.0 if on else 0.0)
        self.anim.start()

    def sizeHint(self):
        return QSize(38, 22)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        off, on = QColor(LINE_2), QColor(ACCENT)
        t = self._pos
        track = QColor(int(off.red() + (on.red() - off.red()) * t), int(off.green() + (on.green() - off.green()) * t),
                       int(off.blue() + (on.blue() - off.blue()) * t))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, 0, 38, 22), 11, 11)
        p.setBrush(QColor(ACCENT_INK) if t > 0.5 else QColor(INK))
        p.drawEllipse(QPointF(11 + t * 16, 11), 8, 8)


def caption(text):
    lab = QLabel(text)
    lab.setObjectName("caption")
    lab.setWordWrap(True)
    return lab


def section(text):
    lab = QLabel(text.upper())
    lab.setObjectName("section")
    return lab


class SliderRow(QWidget):
    def __init__(self, title, lo, hi, value, fmt):
        super().__init__()
        self.fmt = fmt
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(QLabel(title))
        top.addStretch()
        self.val = QLabel(fmt(value))
        self.val.setObjectName("mono")
        top.addWidget(self.val)
        v.addLayout(top)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(lambda x: self.val.setText(fmt(x)))
        v.addWidget(self.slider)


class ToggleRow(QWidget):
    def __init__(self, title, checked, hint=None):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        h = QHBoxLayout()
        h.addWidget(QLabel(title))
        h.addStretch()
        self.toggle = Toggle(checked)
        h.addWidget(self.toggle)
        v.addLayout(h)
        self.hint = caption(hint or "")
        self.hint.setVisible(bool(hint))
        v.addWidget(self.hint)


class MaskGallery(QWidget):
    """Плитки своих масок; последняя плитка «+» добавляет файл. ПКМ — обрезать или удалить, двойной щелчок — обрезать."""
    picked = pyqtSignal(str)
    add_clicked = pyqtSignal()
    remove_clicked = pyqtSignal(str)
    edit_clicked = pyqtSignal(str)

    COLS, GAP = 4, 8

    def __init__(self):
        super().__init__()
        self.items = []    # [(id, name, QImage | None, kind)]
        self.current = None
        self.hover = -1
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def tile(self):
        return (self.width() - self.GAP * (self.COLS - 1)) / self.COLS

    def set_items(self, items):
        self.items = items
        self.fit_height()
        self.update()

    def fit_height(self):
        rows = (len(self.items) + self.COLS) // self.COLS  # +1 плитка «+»
        h = int(rows * self.tile() + (rows - 1) * self.GAP)
        if h != self.height():
            self.setFixedHeight(h)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.fit_height()

    def rect_of(self, i):
        t = self.tile()
        r, c = divmod(i, self.COLS)
        return QRectF(c * (t + self.GAP), r * (t + self.GAP), t, t)

    def at(self, pos):
        for i in range(len(self.items) + 1):
            if self.rect_of(i).contains(pos):
                return i
        return -1

    def mouseMoveEvent(self, e):
        i = self.at(e.position())
        if i != self.hover:
            self.hover = i
            name = self.items[i][1] if 0 <= i < len(self.items) else "Добавить картинку, GIF или видео" if i >= 0 else ""
            self.setToolTip(name)
            self.update()

    def leaveEvent(self, _):
        self.hover = -1
        self.update()

    def mousePressEvent(self, e):
        i = self.at(e.position())
        if i < 0:
            return
        if i == len(self.items):
            if e.button() == Qt.MouseButton.LeftButton:
                self.add_clicked.emit()
            return
        mid = self.items[i][0]
        if e.button() == Qt.MouseButton.RightButton:
            menu = QMenu(self)
            menu.addAction("Обрезать…").triggered.connect(lambda: self.edit_clicked.emit(mid))
            menu.addSeparator()
            menu.addAction("Удалить маску").triggered.connect(lambda: self.remove_clicked.emit(mid))
            menu.exec(e.globalPosition().toPoint())
        else:
            self.picked.emit(mid)

    def mouseDoubleClickEvent(self, e):
        i = self.at(e.position())
        if 0 <= i < len(self.items) and e.button() == Qt.MouseButton.LeftButton:
            self.edit_clicked.emit(self.items[i][0])

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        badge = QFont("Maska Mono")
        badge.setPixelSize(9)
        for i, (mid, _, img, kind) in enumerate(self.items):
            r = self.rect_of(i)
            path = QPainterPath()
            path.addRoundedRect(r, 9, 9)
            p.fillPath(path, QColor(PANEL_2))
            if img is not None:
                p.save()
                p.setClipPath(path)
                sz = img.size().scaled(r.size().toSize(), Qt.AspectRatioMode.KeepAspectRatioByExpanding)
                p.drawImage(QRectF(r.center().x() - sz.width() / 2, r.center().y() - sz.height() / 2,
                                   sz.width(), sz.height()), img)
                p.restore()
            if kind in ("video", "anim"):
                label = "GIF" if kind == "anim" else "▶"
                br = QRectF(r.left() + 5, r.bottom() - 19, 26 if kind == "anim" else 18, 14)
                bp = QPainterPath()
                bp.addRoundedRect(br, 4, 4)
                p.fillPath(bp, QColor(10, 10, 12, 210))
                p.setPen(QColor(INK))
                p.setFont(badge)
                p.drawText(br, Qt.AlignmentFlag.AlignCenter, label)
            if mid == self.current:
                p.setPen(QPen(QColor(ACCENT), 2))
                p.drawRoundedRect(r.adjusted(1, 1, -1, -1), 8, 8)
            elif i == self.hover:
                p.setPen(QPen(QColor(MUTED), 1.2))
                p.drawRoundedRect(r.adjusted(0.6, 0.6, -0.6, -0.6), 9, 9)
        r = self.rect_of(len(self.items))
        hov = self.hover == len(self.items)
        path = QPainterPath()
        path.addRoundedRect(r, 9, 9)
        p.fillPath(path, QColor(HOVER if hov else PANEL_2))
        pen = QPen(QColor(MUTED if hov else LINE_2), 1.2, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawRoundedRect(r.adjusted(0.6, 0.6, -0.6, -0.6), 9, 9)
        p.setPen(QPen(QColor(INK if hov else MUTED), 2, cap=Qt.PenCapStyle.RoundCap))
        c, k = r.center(), r.width() * 0.14
        p.drawLine(QPointF(c.x() - k, c.y()), QPointF(c.x() + k, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - k), QPointF(c.x(), c.y() + k))


class ColorChips(QWidget):
    """Кружки цветов; выбранный — в кольце."""
    changed = pyqtSignal(str)

    def __init__(self, items):
        super().__init__()
        self.items = items  # [(ключ, "#rrggbb")]
        self.current = items[0][0]
        self.setFixedHeight(30)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def at(self, x):
        i = int(x // 34)
        return self.items[i][0] if 0 <= i < len(self.items) else None

    def mousePressEvent(self, e):
        k = self.at(e.position().x())
        if k and k != self.current:
            self.current = k
            self.update()
            self.changed.emit(k)

    def set_current(self, k):
        if any(k == key for key, _ in self.items):
            self.current = k
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for i, (key, hexc) in enumerate(self.items):
            c = QPointF(15 + i * 34, 15)
            if key == self.current:
                p.setPen(QPen(QColor(INK), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(c, 13, 13)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(hexc))
            p.drawEllipse(c, 9, 9)


QSS = f"""
* {{ font-family: "Maska Golos"; font-size: 13.5px; color: {INK}; }}
QMainWindow, #root {{ background: {BG}; }}
#side {{ background: {PANEL}; border-radius: 14px; }}
#title {{ font-size: 20px; font-weight: 700; }}
#caption {{ color: {DIM}; font-size: 12px; }}
#section {{ color: {DIM}; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}
#mono, #stats {{ font-family: "Maska Mono"; color: {MUTED}; font-size: 12.5px; }}
QSlider {{ height: 18px; }}
QSlider::groove:horizontal {{ height: 4px; background: {LINE_2}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {INK}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}
QSlider::sub-page:horizontal:disabled {{ background: {LINE_2}; }}
QSlider::handle:horizontal:disabled {{ background: {DIM}; }}
QLabel:disabled {{ color: {DIM}; }}
QComboBox {{ background: {PANEL_2}; border: 1px solid {LINE_2}; border-radius: 8px; padding: 0 10px; min-height: 30px; }}
QComboBox:hover {{ background: {HOVER}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url({APP_DIR.as_posix()}/chevron.svg); width: 10px; height: 6px; margin-right: 10px; }}
QComboBox QAbstractItemView {{ background: {PANEL_2}; border: 1px solid {LINE_2}; border-radius: 8px; padding: 4px;
  outline: none; selection-background-color: {HOVER}; }}
QPushButton {{ background: {PANEL_2}; border: 1px solid {LINE_2}; border-radius: 8px; min-height: 32px;
  padding: 0 14px; font-weight: 600; }}
QPushButton:hover {{ background: {HOVER}; }}
QPushButton#primary {{ background: {ACCENT}; color: {ACCENT_INK}; border: none; }}
QPushButton#primary:hover {{ background: {ACCENT_2}; }}
QPushButton#primary:pressed {{ background: #7bc92f; }}
QPushButton#update {{ background: transparent; color: {ACCENT}; border: 1px solid {ACCENT}; }}
QPushButton#update:hover {{ background: rgba(143, 224, 59, 0.12); }}
QPushButton#update:disabled {{ color: {MUTED}; border-color: {LINE_2}; }}
QMenu {{ background: {PANEL_2}; border: 1px solid {LINE_2}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 14px; border-radius: 6px; }}
QMenu::item:selected {{ background: {HOVER}; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 6px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {LINE_2}; border-radius: 3px; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QFileDialog, QFileDialog QWidget {{ background: {PANEL}; color: {INK}; }}
QLineEdit {{ background: {PANEL_2}; border: 1px solid {LINE_2}; border-radius: 8px; padding: 5px 8px;
  selection-background-color: {ACCENT}; selection-color: {ACCENT_INK}; }}
QTreeView, QListView {{ background: {PANEL_2}; alternate-background-color: {PANEL}; border: 1px solid {LINE_2};
  border-radius: 8px; selection-background-color: {HOVER}; selection-color: {INK}; outline: none; }}
QHeaderView::section {{ background: {PANEL}; color: {MUTED}; border: none; border-bottom: 1px solid {LINE_2};
  padding: 4px 8px; }}
QToolButton {{ background: transparent; border: none; border-radius: 6px; padding: 4px; }}
QToolButton:hover {{ background: {HOVER}; }}
QToolTip {{ background: {PANEL_2}; color: {INK}; border: 1px solid {LINE_2}; border-radius: 6px; padding: 4px 8px; }}
"""


class Fetcher(QThread):
    """Скачивает картинку, GIF или видео по ссылке из буфера обмена."""
    done = pyqtSignal(str, str)  # путь, ошибка

    def __init__(self, url, folder, stem):
        super().__init__()
        self.url, self.folder, self.stem = url, Path(folder), stem

    def run(self):
        import urllib.request
        try:
            req = urllib.request.Request(self.url, headers={"User-Agent": "Mozilla/5.0 Maska"})
            with urllib.request.urlopen(req, timeout=20) as r:
                ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip()
                ext = Path(self.url.split("?")[0]).suffix.lower()
                if ext not in MEDIA_EXT:
                    ext = {"image/gif": ".gif", "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp",
                           "video/mp4": ".mp4", "video/webm": ".webm"}.get(ctype, "")
                if not ext:
                    raise ValueError("по ссылке не картинка и не видео")
                data = r.read(200 * 1024 * 1024)
            path = self.folder / (self.stem + ext)
            path.write_bytes(data)
            self.done.emit(str(path), "")
        except Exception as e:  # noqa: BLE001
            self.done.emit("", str(e))


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Маска")
        self.resize(1200, 720)
        self.setAcceptDrops(True)
        self.settings = QSettings("Maska", "Maska")
        self.library = Library()
        self.thumbs = {}
        root = QWidget(objectName="root")
        self.setCentralWidget(root)
        lay = QHBoxLayout(root)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(12)

        self.preview = Preview()
        lay.addWidget(self.preview, 1)

        side = QWidget(objectName="side")
        side.setFixedWidth(340)
        lay.addWidget(side)
        outer = QVBoxLayout(side)
        outer.setContentsMargins(0, 0, 0, 18)
        outer.setSpacing(10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)
        s = QVBoxLayout(inner)
        s.setContentsMargins(18, 18, 18, 4)
        s.setSpacing(10)

        title = QLabel("Маска", objectName="title")
        title.setToolTip(f"Версия {VERSION}")
        s.addWidget(title)
        s.addWidget(caption("Лицо прячется прямо в кадре камеры."))
        s.addSpacing(8)

        s.addWidget(section("Режим"))
        self.mode = Segmented(MODES)
        s.addWidget(self.mode)

        self.media_box = QWidget()
        mb = QVBoxLayout(self.media_box)
        mb.setContentsMargins(0, 4, 0, 4)
        mb.setSpacing(8)
        head = QHBoxLayout()
        self.media_title = section("Мои маски")
        head.addWidget(self.media_title)
        head.addStretch()
        mb.addLayout(head)
        self.gallery = MaskGallery()
        mb.addWidget(self.gallery)
        self.media_hint = caption("")
        mb.addWidget(self.media_hint)
        self.rotate = ToggleRow("Наклонять за головой", True)
        mb.addWidget(self.rotate)
        self.match_color = ToggleRow("Подгонять цвет кожи", True, "Лицо с фото подстраивается под свет и тон кожи в кадре.")
        mb.addWidget(self.match_color)
        self.keep_mouth = ToggleRow("Свой рот", True, "Когда говорите, видны настоящие губы изнутри и зубы — выглядит живее.")
        mb.addWidget(self.keep_mouth)
        self.hide_head = ToggleRow("Прятать голову при повороте", False,
                                   "В профиль под чужим лицом размывается вся голова: волосы, ухо, затылок.")
        mb.addWidget(self.hide_head)
        self.feather = SliderRow("Мягкость края", 0, 100, 50, lambda v: f"{v}%")
        mb.addWidget(self.feather)
        s.addWidget(self.media_box)

        self.strength = SliderRow("Сила", 0, 100, 60, lambda v: f"{v}%")
        s.addWidget(self.strength)
        self.pad = SliderRow("Запас вокруг лица", 0, 80, 25, lambda v: f"+{v}%")
        s.addWidget(self.pad)
        self.oval = ToggleRow("Овальная маска", True)
        s.addWidget(self.oval)
        self.failsafe = ToggleRow("Страховка", True,
                                  "Лица не видно — закрыт весь кадр, пока оно снова не найдётся. "
                                  "Так лицо не мелькнёт, даже если детектор его упустил.")
        s.addWidget(self.failsafe)
        s.addSpacing(8)

        s.addWidget(section("Фон"))
        self.bg_mode = Segmented([("off", "Выкл"), ("blur", "Размытие"), ("media", "Своё"), ("green", "Зелёный")])
        s.addWidget(self.bg_mode)
        self.bg_blur = SliderRow("Сила размытия фона", 0, 100, 60, lambda v: f"{v}%")
        s.addWidget(self.bg_blur)
        self.bg_gallery = MaskGallery()
        s.addWidget(self.bg_gallery)
        self.bg_hint = caption("")
        s.addWidget(self.bg_hint)
        s.addSpacing(8)

        s.addWidget(section("Эффекты"))
        self.fx_rows = {}
        for key, label in effects.EFFECTS:
            row = ToggleRow(label, False, "Лучи светят прямо в камеру или туда, куда повёрнута голова. Работает с любым режимом."
                            if key == "lasers" else None)
            s.addWidget(row)
            self.fx_rows[key] = row
        self.laser_colors = ColorChips([(k, v[1]) for k, v in effects.LASER_COLORS.items()])
        self.laser_colors.setVisible(False)
        s.addWidget(self.laser_colors)
        self.laser_dir = Segmented([("camera", "В камеру"), ("head", "Куда смотрит голова")])
        self.laser_dir.setVisible(False)
        s.addWidget(self.laser_dir)
        s.addSpacing(8)

        s.addWidget(section("Камера"))
        self.cam = QComboBox()
        self.cam.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cams = list_cameras()
        for idx, name in self.cams:
            self.cam.addItem(name if IS_WIN else f"{name}  ·  video{idx}", idx)
        if not self.cams:
            self.cam.addItem("Камер не найдено", 0)
        s.addWidget(self.cam)
        self.low_light = ToggleRow("Ярче в темноте", False,
                                   "Камера сама удлиняет выдержку и в полутьме снижает частоту до 5–15 кадров. "
                                   "Выключено — всегда максимум кадров (до 60), тёмный кадр осветляется программно.")
        self.low_light.setVisible(not IS_WIN)  # через DirectShow эту настройку камеры не задать
        s.addWidget(self.low_light)
        self.mirror = ToggleRow("Зеркалить превью", True)
        s.addWidget(self.mirror)
        self.vcam = ToggleRow("Виртуальная камера", False, self.vcam_idle_text())
        s.addWidget(self.vcam)
        s.addStretch()

        bottom = QVBoxLayout()
        bottom.setContentsMargins(18, 0, 18, 0)
        bottom.setSpacing(10)
        self.update_btn = QPushButton(objectName="update")
        self.update_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update_btn.hide()
        self.update_btn.clicked.connect(self.start_update)
        bottom.addWidget(self.update_btn)
        self.stats = QLabel("—", objectName="stats")
        bottom.addWidget(self.stats)
        shot = QPushButton("Сделать снимок", objectName="primary")
        shot.setCursor(Qt.CursorShape.PointingHandCursor)
        shot.setToolTip(f"Сохраняется в {shots_dir()}")
        bottom.addWidget(shot)
        outer.addLayout(bottom)

        self.worker = Worker()
        self.worker.frame.connect(self.preview.set_image)
        self.worker.stats.connect(self.on_stats)
        self.worker.error.connect(self.preview.set_message)
        self.worker.vcam_state.connect(self.on_vcam)

        self.update_kind = updater.install_kind(APP_DIR)
        self.update_info = None
        self.checker = None
        if not os.environ.get("MASKA_SELFTEST") and not os.environ.get("MASKA_NO_UPDATE"):
            QTimer.singleShot(3000, self.check_update)
            self.update_timer = QTimer(self, interval=6 * 3600 * 1000)
            self.update_timer.timeout.connect(self.check_update)
            self.update_timer.start()

        self.toast_timer = QTimer(self, singleShot=True, interval=2500)
        self.toast_timer.timeout.connect(lambda: self.preview.set_message(None))
        self.cur_mask = self.cur_face = self.cur_bg = None
        self.restore()

        self.mode.changed.connect(self.on_mode)
        self.strength.slider.valueChanged.connect(lambda v: setattr(self.worker, "strength", v))
        self.pad.slider.valueChanged.connect(lambda v: setattr(self.worker, "pad", v / 100))
        self.oval.toggle.toggled.connect(lambda v: setattr(self.worker, "oval", v))
        self.failsafe.toggle.toggled.connect(lambda v: setattr(self.worker, "failsafe", v))
        self.mirror.toggle.toggled.connect(lambda v: setattr(self.worker, "mirror", v))
        self.low_light.toggle.toggled.connect(self.worker.set_low_light)
        self.rotate.toggle.toggled.connect(lambda v: setattr(self.worker, "rotate", v))
        self.match_color.toggle.toggled.connect(lambda v: setattr(self.worker, "match_color", v))
        self.keep_mouth.toggle.toggled.connect(lambda v: setattr(self.worker, "keep_mouth", v))
        self.hide_head.toggle.toggled.connect(lambda v: setattr(self.worker, "hide_head", v))
        self.feather.slider.valueChanged.connect(lambda v: setattr(self.worker, "feather", v / 100))
        self.gallery.edit_clicked.connect(self.edit_mask)
        for key, row in self.fx_rows.items():
            row.toggle.toggled.connect(lambda on, k=key: self.set_effect(k, on))
        self.laser_colors.changed.connect(lambda c: self.worker.effect_opts.__setitem__("laser_color", c))
        self.laser_dir.changed.connect(lambda d: self.worker.effect_opts.__setitem__("laser_dir", d))
        self.bg_mode.changed.connect(self.on_bg_mode)
        self.bg_blur.slider.valueChanged.connect(lambda v: setattr(self.worker, "bg_blur", v))
        self.bg_gallery.picked.connect(self.set_bg)
        self.bg_gallery.add_clicked.connect(lambda: self.add_dialog(target="bg"))
        self.bg_gallery.remove_clicked.connect(self.remove_mask)
        self.bg_gallery.edit_clicked.connect(self.edit_mask)
        from PyQt6.QtGui import QKeySequence, QShortcut
        QShortcut(QKeySequence.StandardKey.Paste, self, activated=self.paste)
        self.worker.face_state.connect(self.on_face_state)
        self.vcam.toggle.toggled.connect(lambda v: setattr(self.worker, "want_vcam", v))
        self.cam.currentIndexChanged.connect(lambda _: self.worker.switch_device(self.cam.currentData()))
        self.gallery.picked.connect(self.pick_mask)
        self.gallery.add_clicked.connect(self.add_dialog)
        self.gallery.remove_clicked.connect(self.remove_mask)
        shot.clicked.connect(self.snapshot)
        self.worker.start()

    # ── настройки ──

    def restore(self):
        st = self.settings
        self.strength.slider.setValue(int(st.value("strength", 60)))
        self.pad.slider.setValue(int(st.value("pad", 25)))
        self.feather.slider.setValue(int(st.value("feather", 50)))
        for row, key in ((self.oval, "oval"), (self.failsafe, "failsafe"), (self.mirror, "mirror"),
                         (self.rotate, "rotate"), (self.match_color, "match_color"), (self.keep_mouth, "keep_mouth")):
            row.toggle.setChecked(st.value(key, True, type=bool))
            row.toggle.set_knob(1.0 if row.toggle.isChecked() else 0.0)
        w = self.worker
        w.strength, w.pad = self.strength.slider.value(), self.pad.slider.value() / 100
        w.oval, w.failsafe = self.oval.toggle.isChecked(), self.failsafe.toggle.isChecked()
        w.mirror, w.rotate = self.mirror.toggle.isChecked(), self.rotate.toggle.isChecked()
        w.match_color, w.keep_mouth = self.match_color.toggle.isChecked(), self.keep_mouth.toggle.isChecked()
        w.feather = self.feather.slider.value() / 100
        self.low_light.toggle.setChecked(st.value("low_light", False, type=bool))
        self.low_light.toggle.set_knob(1.0 if self.low_light.toggle.isChecked() else 0.0)
        w.low_light = self.low_light.toggle.isChecked()
        self.hide_head.toggle.setChecked(st.value("hide_head", False, type=bool))
        self.hide_head.toggle.set_knob(1.0 if self.hide_head.toggle.isChecked() else 0.0)
        w.hide_head = self.hide_head.toggle.isChecked()

        cam = st.value("camera", "")
        i = next((n for n, (_, name) in enumerate(self.cams) if name == cam), 0)
        self.cam.setCurrentIndex(i)
        w.device = self.cam.currentData()

        for k in (st.value("effects", "") or "").split(","):
            if k in self.fx_rows:
                self.fx_rows[k].toggle.setChecked(True)
                self.fx_rows[k].toggle.set_knob(1.0)
                self.set_effect(k, True)
        self.laser_colors.set_current(st.value("laser_color", "red"))
        w.effect_opts["laser_color"] = self.laser_colors.current
        ldir = st.value("laser_dir", "camera")
        self.laser_dir.select(0 if ldir != "head" else 1, animate=False)
        w.effect_opts["laser_dir"] = "head" if ldir == "head" else "camera"
        self.bg_blur.slider.setValue(int(st.value("bg_blur", 60)))
        w.bg_blur = self.bg_blur.slider.value()
        self.cur_bg = None
        self.set_bg(st.value("bg", ""))
        bgm = st.value("bg_mode", "off")
        keys = [k for k, _ in self.bg_mode.items]
        self.bg_mode.select(keys.index(bgm) if bgm in keys else 0, animate=False)
        self.on_bg_mode(self.bg_mode.items[self.bg_mode.index][0])
        self.cur_mask = self.cur_face = None
        self.set_mask(st.value("mask", ""))
        self.set_face(st.value("face", ""))
        keys = [k for k, _ in MODES]
        mode = st.value("mode", "blur")
        self.mode.select(keys.index(mode) if mode in keys else 0, animate=False)
        self.on_mode(self.mode.items[self.mode.index][0])

        # Linux: модуль загружен — сразу отдаём картинку в Discord. Windows: как было в прошлый раз
        want = loopback_device() is not None if not IS_WIN else st.value("vcam", False, type=bool)
        if want and not os.environ.get("MASKA_SELFTEST"):
            self.vcam.toggle.setChecked(True)
            self.vcam.toggle.set_knob(1.0)
            w.want_vcam = True

    def save(self):
        st = self.settings
        st.setValue("mode", self.worker.mode)
        st.setValue("mask", self.cur_mask or "")
        st.setValue("face", self.cur_face or "")
        st.setValue("feather", self.feather.slider.value())
        st.setValue("effects", ",".join(sorted(self.worker.effects)))
        st.setValue("laser_color", self.laser_colors.current)
        st.setValue("laser_dir", self.worker.effect_opts.get("laser_dir", "camera"))
        st.setValue("bg_mode", self.worker.bg_mode)
        st.setValue("bg", self.cur_bg or "")
        st.setValue("bg_blur", self.bg_blur.slider.value())
        st.setValue("strength", self.strength.slider.value())
        st.setValue("pad", self.pad.slider.value())
        for row, key in ((self.oval, "oval"), (self.failsafe, "failsafe"), (self.mirror, "mirror"),
                         (self.rotate, "rotate"), (self.match_color, "match_color"), (self.keep_mouth, "keep_mouth")):
            st.setValue(key, row.toggle.isChecked())
        if self.cams:
            st.setValue("camera", self.cams[self.cam.currentIndex()][1])
        st.setValue("vcam", self.vcam.toggle.isChecked())
        st.setValue("hide_head", self.hide_head.toggle.isChecked())
        st.setValue("low_light", self.low_light.toggle.isChecked())

    # ── свои маски ──

    def face_mode(self):
        return self.worker.mode == "face"

    def refresh_gallery(self):
        items = []
        for m in self.library.items:
            mid = m["id"]
            key = (mid, json.dumps(m.get("edit") or {}, sort_keys=True))
            if self.thumbs.get(mid, (None,))[0] != key:
                path = MASKS_DIR / m["file"]
                ext = path.suffix.lower()
                kind = "video" if ext in VIDEO_EXT else "anim" if ext in ANIM_EXT else "image"
                if ext == ".webp":
                    r = QImageReader(str(path))
                    kind = "anim" if r.supportsAnimation() and r.imageCount() > 1 else "image"
                self.thumbs[mid] = (key, thumbnail(path, edit=m.get("edit")), kind)
            _, img, kind = self.thumbs[mid]
            if self.face_mode() and kind != "image":
                continue  # лицо берётся только с фото
            items.append((mid, m["name"], img, kind))
        self.gallery.current = self.cur_face if self.face_mode() else self.cur_mask
        self.gallery.set_items(items)

    def set_mask(self, mid):
        spec = self.library.spec(mid) if mid else None
        self.cur_mask = mid if spec else None
        self.worker.media_spec = spec
        if not self.face_mode():
            self.gallery.current = self.cur_mask
        self.gallery.update()

    def set_face(self, mid):
        spec = self.library.spec(mid) if mid else None
        self.cur_face = mid if spec else None
        self.worker.face_spec = spec
        if self.face_mode():
            self.gallery.current = self.cur_face
        self.gallery.update()

    def pick_mask(self, mid):
        if self.face_mode():
            self.set_face(mid)
            return
        self.set_mask(mid)
        if self.worker.mode != "media":
            self.mode.select([k for k, _ in MODES].index("media"))

    def add_files(self, paths, target=None):
        """target: None — в текущий режим (маска или лицо), "bg" — как фон."""
        added, bad = [], []
        for p in paths:
            ext = Path(p).suffix.lower()
            if ext not in MEDIA_EXT or (target is None and self.face_mode() and ext not in IMAGE_EXT):
                bad.append(Path(p).name)
                continue
            try:
                added.append(self.library.add(p))
            except OSError as e:
                bad.append(f"{Path(p).name} ({e.strerror})")
        self.refresh_bg_gallery()
        if added:
            if target == "bg":
                self.set_bg(added[-1])
            else:
                self.pick_mask(added[-1])
            if len(added) == 1:  # один файл — сразу предлагаем обрезать
                QTimer.singleShot(0, lambda: self.edit_mask(added[0]))
        if bad:
            self.toast(("Для лица нужно фото: " if target is None and self.face_mode() else "Не подходит: ")
                       + ", ".join(bad))

    def edit_mask(self, mid):
        m = self.library.get(mid)
        if not m:
            return
        try:
            dlg = EditDialog(self, str(MASKS_DIR / m["file"]), m["name"], m.get("edit"), MediaMask, Segmented, COLORS)
        except Exception as e:  # noqa: BLE001
            self.toast(f"Не открылось: {e}")
            return
        if dlg.exec() != EditDialog.DialogCode.Accepted:
            return
        self.library.set_edit(mid, dlg.result_edit())
        self.refresh_bg_gallery()
        if mid == self.cur_mask:
            self.set_mask(mid)
        if mid == self.cur_face:
            self.set_face(mid)
        if mid == self.cur_bg:
            self.set_bg(mid)

    def add_dialog(self, target=None):
        start = self.settings.value("last_dir", str(Path.home()))
        title = "Выберите фон: картинку, GIF или видео" if target == "bg" else "Выберите картинку, GIF или видео"
        files, _ = QFileDialog.getOpenFileNames(self, title, start, f"{MEDIA_FILTER};;Все файлы (*)")
        if files:
            self.settings.setValue("last_dir", str(Path(files[0]).parent))
            self.add_files(files, target)

    def remove_mask(self, mid):
        if self.cur_mask == mid:
            self.set_mask(None)
        if self.cur_face == mid:
            self.set_face(None)
        if self.cur_bg == mid:
            self.set_bg(None)
        self.library.remove(mid)
        self.thumbs.pop(mid, None)
        self.refresh_bg_gallery()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(u.isLocalFile() and Path(u.toLocalFile()).suffix.lower() in MEDIA_EXT
                                          for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_files([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])

    # ── фон ──

    def on_bg_mode(self, mode):
        self.worker.bg_mode = mode
        self.bg_blur.setVisible(mode == "blur")
        self.bg_gallery.setVisible(mode == "media")
        self.bg_hint.setVisible(mode in ("media", "green"))
        self.bg_hint.setText("Картинка, GIF или видео за вашей спиной. Двойной щелчок — обрезать." if mode == "media"
                             else "Ровный зелёный фон — в OBS его убирает фильтр «Хромакей».")
        self.refresh_bg_gallery()

    def refresh_bg_gallery(self):
        self.refresh_gallery()  # обновит миниатюры
        items = []
        for m in self.library.items:
            _, img, kind = self.thumbs[m["id"]]
            items.append((m["id"], m["name"], img, kind))
        self.bg_gallery.current = self.cur_bg
        self.bg_gallery.set_items(items)

    def set_bg(self, mid):
        spec = self.library.spec(mid) if mid else None
        self.cur_bg = mid if spec else None
        self.worker.bg_spec = spec
        self.bg_gallery.current = self.cur_bg
        self.bg_gallery.update()

    # ── эффекты и вставка ──

    def set_effect(self, key, on):
        fx = set(self.worker.effects)
        (fx.add if on else fx.discard)(key)
        self.worker.effects = fx  # новое множество целиком — поток камеры не увидит его наполовину изменённым
        if key == "lasers":
            self.laser_colors.setVisible(on)
            self.laser_dir.setVisible(on)

    def paste(self):
        """Ctrl+V: файл из проводника, картинка или GIF из браузера, ссылка на картинку/видео."""
        md = QApplication.clipboard().mimeData()
        files = [u.toLocalFile() for u in md.urls() if u.isLocalFile()] if md.hasUrls() else []
        if files:
            self.add_files(files)
            return
        stamp = time.strftime("Вставка %H-%M-%S")
        tmp = Path(tempfile.mkdtemp(prefix="maska-"))
        for fmt in ("image/gif", "image/webp"):  # анимацию берём байтами — через QImage она потеряется
            if fmt in md.formats():
                path = tmp / f"{stamp}.{fmt.split('/')[1]}"
                path.write_bytes(bytes(md.data(fmt)))
                self.add_files([str(path)])
                shutil.rmtree(tmp, ignore_errors=True)
                return
        if md.hasImage():
            img = QImage(md.imageData())
            if not img.isNull():
                path = tmp / f"{stamp}.png"
                img.save(str(path))
                self.add_files([str(path)])
                shutil.rmtree(tmp, ignore_errors=True)
                return
        text = md.text().strip() if md.hasText() else ""
        if text and Path(text).is_file():
            self.add_files([text])
        elif text.startswith(("http://", "https://")):
            self.toast("Скачиваю по ссылке…")
            self.fetcher = Fetcher(text, tmp, stamp)
            self.fetcher.done.connect(self.on_fetched)
            self.fetcher.start()
            return
        else:
            self.toast("В буфере обмена нет картинки, GIF или видео")
        shutil.rmtree(tmp, ignore_errors=True)

    def on_fetched(self, path, err):
        if err:
            self.toast(f"Не скачалось: {err}")
        else:
            self.add_files([path])
        shutil.rmtree(self.fetcher.folder, ignore_errors=True)

    # ── обновления ──

    def check_update(self):
        if self.checker and self.checker.isRunning() or self.update_info and self.update_info.get("busy"):
            return
        self.checker = updater.Checker(VERSION, self.update_kind)
        self.checker.found.connect(self.on_update_found)
        self.checker.start()

    def on_update_found(self, version, page, asset):
        self.update_info = {"version": version, "page": page, "asset": asset}
        auto = asset is not None
        self.update_btn.setText(f"Обновить до {version}" if auto else f"Вышла версия {version} — открыть")
        self.update_btn.setToolTip("Скачать и установить новую версию" if auto else page)
        self.update_btn.setEnabled(True)
        self.update_btn.show()

    def start_update(self):
        info = self.update_info
        if not info:
            return
        if info["asset"] is None:
            from PyQt6.QtCore import QUrl
            from PyQt6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl(info["page"]))
            return
        info["busy"] = True
        self.update_btn.setEnabled(False)
        self.update_btn.setText("Скачиваю обновление…")
        self.installer = updater.Installer(info["asset"], self.update_kind)
        self.installer.progress.connect(lambda v: self.update_btn.setText(f"Скачиваю обновление… {v}%"))
        self.installer.done.connect(self.on_update_done)
        self.installer.start()

    def on_update_done(self, err):
        info = self.update_info
        info["busy"] = False
        if err:
            self.update_btn.setEnabled(True)
            self.update_btn.setText(f"Обновить до {info['version']}")
            self.toast(f"Обновление не удалось: {err}")
            return
        if IS_WIN:
            self.close()  # установщик уже запущен: он поставит новую версию и откроет её
            QApplication.quit()
        else:
            self.update_btn.setText(f"Версия {info['version']} установлена — перезапустить")
            self.update_btn.setEnabled(True)
            self.update_btn.clicked.disconnect()
            self.update_btn.clicked.connect(self.restart)

    def restart(self):
        self.close()
        os.execv(sys.executable, [sys.executable] + sys.argv)

    # ── прочее ──

    def on_mode(self, mode):
        self.worker.mode = mode
        media, swap = mode == "media", mode == "face"
        self.media_box.setVisible(media or swap)
        self.media_title.setText("ЧУЖОЕ ЛИЦО" if swap else "МОИ МАСКИ")
        self.media_hint.setText(
            "Фото человека — его лицо натянется на ваше и будет повторять мимику. Лучше всего — фото анфас при "
            "ровном свете. Двойной щелчок по фото — обрезать, если на нём несколько лиц." if swap else
            "Картинка, GIF или видео встанет на место лица. Файл можно перетащить в окно или вставить Ctrl+V. "
            "Двойной щелчок по маске — обрезать. PNG с прозрачностью ложится без фона.")
        self.rotate.setVisible(media)
        for row in (self.match_color, self.keep_mouth, self.hide_head, self.feather):
            row.setVisible(swap)
        self.strength.setVisible(not media and not swap)
        self.strength.setEnabled(mode in ("blur", "pixel"))
        self.pad.setVisible(not swap)
        self.oval.setVisible(not swap)
        self.refresh_gallery()
        if media and not self.cur_mask:
            self.toast("Добавьте картинку, GIF или видео — пока лицо просто размыто")
        if swap and not self.cur_face:
            self.toast("Добавьте фото с лицом — пока лицо просто размыто")

    def on_face_state(self, err):
        if err:
            self.toast(f"Это фото не подходит: {err}. Обрежьте его вокруг лица или выберите другое")
            self.preview.set_message(f"Это фото не подходит: {err}")

    def on_stats(self, fps, faces, covered):
        if covered:
            self.stats.setText(f"{fps:.0f} fps  ·  лица не видно — кадр закрыт")
            self.stats.setStyleSheet(f"color: {RED};")
        else:
            self.stats.setText(f"{fps:.0f} fps  ·  лиц в кадре: {faces}")
            self.stats.setStyleSheet("")

    @staticmethod
    def vcam_idle_text():
        if IS_WIN:
            return "Отдаёт картинку с маской в Discord, OBS, браузер через виртуальную камеру OBS."
        return "Отдаёт картинку с маской в Discord, OBS, браузер."

    def on_vcam(self, on, info):
        if on:
            name = "OBS Virtual Camera" if IS_WIN else "Маска"
            self.vcam.hint.setText(f"Работает. В Discord: Настройки → Голос и видео → Камера → «{name}». "
                                   "Не закрывайте это окно.")
            self.vcam.hint.setStyleSheet(f"color: {ACCENT};")
        else:
            self.vcam.toggle.blockSignals(True)
            self.vcam.toggle.setChecked(False)
            self.vcam.toggle._animate(False)
            self.vcam.toggle.blockSignals(False)
            if info:
                if IS_WIN:
                    text = "Не нашлась виртуальная камера OBS. Установите OBS Studio (obsproject.com) и включите снова."
                elif APP_DIR.as_posix().startswith("/usr/"):
                    text = "Не загружен модуль камеры — выполните в терминале maska-setup-vcam"
                else:
                    text = "Не загружен модуль камеры — запустите setup-vcam.sh"
                self.vcam.hint.setText(text)
                self.vcam.hint.setStyleSheet(f"color: {RED};")
                self.vcam.hint.setToolTip(info)
            else:
                self.vcam.hint.setText(self.vcam_idle_text())
                self.vcam.hint.setStyleSheet("")

    def toast(self, text):
        self.preview.set_message(text)
        self.toast_timer.start()

    def snapshot(self):
        path = shots_dir() / time.strftime("maska-%Y%m%d-%H%M%S.png")
        self.worker.snapshot_to = path
        self.toast(f"Снимок сохранён: {path.name}")

    def closeEvent(self, e):
        if not os.environ.get("MASKA_SELFTEST"):
            self.save()
        self.worker.stop()
        super().closeEvent(e)


def main():
    selftest = "--selftest" in sys.argv  # проверка сборки в CI: модель, шрифты и окно поднимаются без ошибок
    if selftest:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ["MASKA_SELFTEST"] = "1"
        cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), 0.5, 0.3, 50)
        face.FaceMesh(MESH_MODEL).landmarks(np.full((240, 320, 3), 128, np.uint8), (100, 60, 120, 120))
        background.Segmenter(SEG_MODEL).mask(np.full((360, 640, 3), 128, np.uint8))
    elif not IS_WIN:
        os.environ.setdefault("QT_QPA_PLATFORM", "wayland;xcb")
        # окна «Открыть файл» — системные (GNOME, KDE) через xdg-desktop-portal, а не встроенное окно Qt
        os.environ.setdefault("QT_QPA_PLATFORMTHEME", "xdgdesktopportal")
    app = QApplication(sys.argv)
    app.setApplicationName("Маска")
    app.setDesktopFileName("maska")
    app.setWindowIcon(QIcon(str(APP_DIR / "icon.svg")))
    for f in (APP_DIR / "fonts").glob("*.ttf"):
        QFontDatabase.addApplicationFont(str(f))
    app.setStyleSheet(QSS)
    w = Main()
    w.show()
    if selftest:
        QTimer.singleShot(1500, w.close)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
