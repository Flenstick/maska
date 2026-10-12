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
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import (QEasingCurve, QPointF, QPropertyAnimation, QRectF, QSettings, QSize, QStandardPaths, Qt,
                          QThread, QTimer, pyqtProperty, pyqtSignal)
from PyQt6.QtGui import (QColor, QFont, QFontDatabase, QFontMetricsF, QIcon, QImage, QImageReader, QPainter, QPainterPath,
                         QPen)
from PyQt6.QtWidgets import (QAbstractButton, QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel,
                             QMainWindow, QMenu, QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget)

VERSION = "1.0.0"
IS_WIN = sys.platform == "win32"
APP_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))  # в сборке PyInstaller — папка распаковки
MODEL = APP_DIR / "models/face_detection_yunet_2023mar.onnx"
if IS_WIN:
    DATA_DIR = Path(os.environ.get("APPDATA", Path.home())) / "Maska"
else:
    DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "maska"
MASKS_DIR = DATA_DIR / "masks"

BG, PANEL, PANEL_2, HOVER = "#0a0a0c", "#131316", "#1b1b1f", "#25252a"
LINE, LINE_2 = "#1f1f24", "#2e2e34"
INK, MUTED, DIM = "#f4f4f5", "#a1a1aa", "#6b6b74"
ACCENT, ACCENT_2, ACCENT_INK, RED = "#8fe03b", "#a6ef55", "#0c1600", "#f0445a"

MODES = [("blur", "Размытие"), ("pixel", "Пиксели"), ("black", "Чёрный"), ("media", "Своя"), ("off", "Выкл")]

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

    def __init__(self):
        self.tracks = []  # {"box": [x, y, w, h], "seen": t детектора, "flow": t потока}
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
        x, y, w, h = (int(v) for v in box)
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

    def update(self, gray, boxes, now):
        free = list(self.tracks)
        for box in boxes:
            best = max(free, key=lambda t: self.iou(t["box"], box), default=None)
            if best is None or (self.iou(best["box"], box) < 0.1 and not self.near(best["box"], box)):
                self.tracks.append({"box": list(box), "seen": now, "flow": now})
                continue
            free.remove(best)
            best["box"] = [o * 0.3 + n * 0.7 for o, n in zip(best["box"], box)]
            best["seen"] = best["flow"] = now
        if self.prev is not None and self.prev.shape == gray.shape:
            for t in free:
                d = self.flow(gray, t["box"])
                if d is not None:
                    t["box"][0] += d[0]
                    t["box"][1] += d[1]
                    t["flow"] = now
        self.prev = gray
        self.tracks = [t for t in self.tracks
                       if now - t["seen"] <= self.FLOW_HOLD and now - t["flow"] <= self.BLIND_HOLD]
        # две дорожки на одном лице — оставляем свежую
        self.tracks.sort(key=lambda t: -t["seen"])
        kept = []
        for t in self.tracks:
            if not any(self.iou(t["box"], k["box"]) > 0.3 for k in kept):
                kept.append(t)
        self.tracks = kept
        return [(t["box"], now - t["seen"]) for t in kept]


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
        m = np.zeros((rh, rw), np.float32)
        cv2.ellipse(m, (rw // 2, rh // 2), (int(rw * 0.46), int(rh * 0.46)), 0, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), max(1.0, min(rw, rh) * 0.04))[..., None]
        roi[:] = (eff * m + roi * (1.0 - m)).astype(np.uint8)


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


class MediaMask:
    """Кадр маски на момент t: картинка (в т. ч. с прозрачностью), GIF/анимированный WebP или видео по кругу."""

    def __init__(self, path):
        self.path = Path(path)
        self.frames, self.ends, self.cap = [], None, None
        if self.path.suffix.lower() in VIDEO_EXT:
            self.cap = cv2.VideoCapture(str(self.path))
            ok, f = self.cap.read() if self.cap.isOpened() else (False, None)
            if not ok:
                raise ValueError("не удалось прочитать видео")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.pos, self.last = 0, cv2.cvtColor(fit_side(f), cv2.COLOR_BGR2BGRA)
            self.alpha = False
            return
        reader = QImageReader(str(self.path))
        reader.setAutoTransform(True)  # поворот по EXIF у фото с телефона
        delays = []
        while len(self.frames) < MEDIA_FRAMES:
            img = reader.read()
            if img.isNull():
                break
            self.frames.append(fit_side(qimage_to_bgra(img)))
            delays.append(max(20, reader.nextImageDelay() or 100) / 1000)
            if not reader.supportsAnimation():
                break
        if not self.frames:
            raise ValueError(reader.errorString() or "не удалось прочитать файл")
        self.ends = np.cumsum(delays)
        self.alpha = bool(self.frames[0][..., 3].min() < 250)

    @property
    def kind(self):
        return "video" if self.cap is not None else "anim" if len(self.frames) > 1 else "image"

    def frame(self, t):
        if self.cap is None:
            if len(self.frames) == 1:
                return self.frames[0]
            i = int(np.searchsorted(self.ends, t % self.ends[-1], side="right"))
            return self.frames[min(i, len(self.frames) - 1)]
        target = int(t * self.fps)
        if self.count > 0:
            target %= self.count
        if target < self.pos or target - self.pos > 90:  # назад или далеко вперёд — перемотка вместо чтения подряд
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.pos = target
        for _ in range(8):  # отстали — догоняем, но не больше нескольких кадров за раз
            if self.pos > target:
                break
            ok, f = self.cap.read()
            if not ok:  # конец ролика (или неверное число кадров) — с начала
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.pos = 0
                if self.count <= 0:
                    self.count = target  # длительность не была известна — теперь известна
                break
            self.pos += 1
            self.last = cv2.cvtColor(fit_side(f), cv2.COLOR_BGR2BGRA)
        return self.last

    def close(self):
        if self.cap is not None:
            self.cap.release()


def thumbnail(path, size=128):
    """Первый кадр маски как QImage для галереи."""
    try:
        m = MediaMask(path)
        f = fit_side(m.frame(0), size)
        m.close()
    except Exception:  # noqa: BLE001 — битый файл покажем пустой плиткой
        return None
    h, w = f.shape[:2]
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

    def path(self, mid):
        for m in self.items:
            if m["id"] == mid:
                return MASKS_DIR / m["file"]
        return None

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
            return pyvirtualcam.Camera(w, h, 30, fmt=pyvirtualcam.PixelFormat.BGR, backend="v4l2loopback",
                                       device=loopback_device())
    import pyvirtualcam
    return pyvirtualcam.Camera(w, h, 30, fmt=pyvirtualcam.PixelFormat.BGR)


class Worker(QThread):
    frame = pyqtSignal(QImage)
    stats = pyqtSignal(float, int, bool)
    error = pyqtSignal(str)
    vcam_state = pyqtSignal(bool, str)

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
        self.media_path = None   # своя маска; меняется из окна, грузится в потоке камеры
        self.rotate = True       # своя маска наклоняется вместе с головой
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
        cap = cv2.VideoCapture(self.device, cv2.CAP_DSHOW if IS_WIN else cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    def run(self):
        detector = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), 0.5, 0.3, 50)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        tracker = Tracker()
        cap = self.open_cam()
        vcam = None
        media, media_path, media_t0 = None, None, 0.0
        t_prev, fps = time.monotonic(), 0.0
        while self._running:
            if self.media_path != media_path:
                media_path = self.media_path
                if media:
                    media.close()
                media = None
                if media_path:
                    try:
                        media, media_t0 = MediaMask(media_path), time.monotonic()
                    except Exception as e:  # noqa: BLE001
                        self.error.emit(f"Маска не открылась: {e}")
            if self._restart:
                self._restart = False
                cap.release()
                cap = self.open_cam()
                tracker = Tracker()
                if vcam:
                    vcam.close()
                    vcam = None
            ok, img = cap.read() if cap.isOpened() else (False, None)
            if not ok and os.environ.get("MASKA_SOURCE") and cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # файл кончился — по кругу
                continue
            if not ok:
                self.error.emit("Камера не отвечает. Проверьте, не занята ли она другой программой.")
                self.msleep(500)
                cap.release()
                cap = self.open_cam()
                continue

            h, w = img.shape[:2]
            covered = False
            try:
                scale = min(1.0, 640 / w)
                small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img
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
                now = time.monotonic()
                live = tracker.update(gray, boxes, now)
                faces_n = len(live)

                mode = self.mode
                mframe = None
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
                    for box, lost in live:
                        grow = min(0.6, lost * 0.4)  # чем дольше лицо не подтверждено, тем шире маска
                        rect = [v / scale for v in box[:4]]
                        if mframe is None:
                            apply_mask(img, [rect], mode, self.strength, self.pad + grow, self.oval)
                        else:
                            apply_media(img, rect, mframe, media.alpha, self.pad + grow, self.oval,
                                        box[4] if self.rotate else 0.0, cover=not media.alpha)
            except Exception as e:  # noqa: BLE001 — что бы ни сломалось, лицо не должно утечь
                img[:] = 0
                covered, faces_n, now = True, 0, time.monotonic()
                self.error.emit(f"Ошибка обработки, кадр закрыт: {e}")

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
            if vcam is not None:
                vcam.send(img)

            if self.snapshot_to:
                path, self.snapshot_to = self.snapshot_to, None
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), img)

            shown = cv2.flip(img, 1) if self.mirror else img
            rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
            self.frame.emit(QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy())

            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = fps * 0.9 + (1 / dt) * 0.1 if fps else 1 / dt
            self.stats.emit(fps, faces_n, covered)
        cap.release()
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
    """Плитки своих масок; последняя плитка «+» добавляет файл. ПКМ по плитке — удалить."""
    picked = pyqtSignal(str)
    add_clicked = pyqtSignal()
    remove_clicked = pyqtSignal(str)

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
            menu.addAction("Удалить маску").triggered.connect(lambda: self.remove_clicked.emit(mid))
            menu.exec(e.globalPosition().toPoint())
        else:
            self.picked.emit(mid)

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
QMenu {{ background: {PANEL_2}; border: 1px solid {LINE_2}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 14px; border-radius: 6px; }}
QMenu::item:selected {{ background: {HOVER}; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 6px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {LINE_2}; border-radius: 3px; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QToolTip {{ background: {PANEL_2}; color: {INK}; border: 1px solid {LINE_2}; border-radius: 6px; padding: 4px 8px; }}
"""


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
        head.addWidget(section("Мои маски"))
        head.addStretch()
        mb.addLayout(head)
        self.gallery = MaskGallery()
        mb.addWidget(self.gallery)
        self.media_hint = caption("Картинка, GIF или видео встанет на место лица. "
                                  "Файл можно просто перетащить в окно. PNG с прозрачностью ложится без фона.")
        mb.addWidget(self.media_hint)
        self.rotate = ToggleRow("Наклонять за головой", True)
        mb.addWidget(self.rotate)
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

        s.addWidget(section("Камера"))
        self.cam = QComboBox()
        self.cam.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cams = list_cameras()
        for idx, name in self.cams:
            self.cam.addItem(name if IS_WIN else f"{name}  ·  video{idx}", idx)
        if not self.cams:
            self.cam.addItem("Камер не найдено", 0)
        s.addWidget(self.cam)
        self.mirror = ToggleRow("Зеркалить превью", True)
        s.addWidget(self.mirror)
        self.vcam = ToggleRow("Виртуальная камера", False, self.vcam_idle_text())
        s.addWidget(self.vcam)
        s.addStretch()

        bottom = QVBoxLayout()
        bottom.setContentsMargins(18, 0, 18, 0)
        bottom.setSpacing(10)
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

        self.toast_timer = QTimer(self, singleShot=True, interval=2500)
        self.toast_timer.timeout.connect(lambda: self.preview.set_message(None))
        self.restore()

        self.mode.changed.connect(self.on_mode)
        self.strength.slider.valueChanged.connect(lambda v: setattr(self.worker, "strength", v))
        self.pad.slider.valueChanged.connect(lambda v: setattr(self.worker, "pad", v / 100))
        self.oval.toggle.toggled.connect(lambda v: setattr(self.worker, "oval", v))
        self.failsafe.toggle.toggled.connect(lambda v: setattr(self.worker, "failsafe", v))
        self.mirror.toggle.toggled.connect(lambda v: setattr(self.worker, "mirror", v))
        self.rotate.toggle.toggled.connect(lambda v: setattr(self.worker, "rotate", v))
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
        for row, key in ((self.oval, "oval"), (self.failsafe, "failsafe"), (self.mirror, "mirror"),
                         (self.rotate, "rotate")):
            row.toggle.setChecked(st.value(key, True, type=bool))
            row.toggle.set_knob(1.0 if row.toggle.isChecked() else 0.0)
        w = self.worker
        w.strength, w.pad = self.strength.slider.value(), self.pad.slider.value() / 100
        w.oval, w.failsafe = self.oval.toggle.isChecked(), self.failsafe.toggle.isChecked()
        w.mirror, w.rotate = self.mirror.toggle.isChecked(), self.rotate.toggle.isChecked()

        cam = st.value("camera", "")
        i = next((n for n, (_, name) in enumerate(self.cams) if name == cam), 0)
        self.cam.setCurrentIndex(i)
        w.device = self.cam.currentData()

        self.refresh_gallery()
        self.set_mask(st.value("mask", ""))
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
        st.setValue("mask", self.gallery.current or "")
        st.setValue("strength", self.strength.slider.value())
        st.setValue("pad", self.pad.slider.value())
        for row, key in ((self.oval, "oval"), (self.failsafe, "failsafe"), (self.mirror, "mirror"),
                         (self.rotate, "rotate")):
            st.setValue(key, row.toggle.isChecked())
        if self.cams:
            st.setValue("camera", self.cams[self.cam.currentIndex()][1])
        st.setValue("vcam", self.vcam.toggle.isChecked())

    # ── свои маски ──

    def refresh_gallery(self):
        items = []
        for m in self.library.items:
            mid = m["id"]
            if mid not in self.thumbs:
                path = MASKS_DIR / m["file"]
                ext = path.suffix.lower()
                kind = "video" if ext in VIDEO_EXT else "anim" if ext in ANIM_EXT else "image"
                if ext == ".webp":
                    r = QImageReader(str(path))
                    kind = "anim" if r.supportsAnimation() and r.imageCount() > 1 else "image"
                self.thumbs[mid] = (thumbnail(path), kind)
            img, kind = self.thumbs[mid]
            items.append((mid, m["name"], img, kind))
        self.gallery.set_items(items)

    def set_mask(self, mid):
        path = self.library.path(mid) if mid else None
        self.gallery.current = mid if path else None
        self.worker.media_path = str(path) if path else None
        self.gallery.update()

    def pick_mask(self, mid):
        self.set_mask(mid)
        if self.worker.mode != "media":
            self.mode.select([k for k, _ in MODES].index("media"))

    def add_files(self, paths):
        mid = None
        bad = []
        for p in paths:
            if Path(p).suffix.lower() not in MEDIA_EXT:
                bad.append(Path(p).name)
                continue
            try:
                mid = self.library.add(p)
            except OSError as e:
                bad.append(f"{Path(p).name} ({e.strerror})")
        self.refresh_gallery()
        if mid:
            self.pick_mask(mid)
        if bad:
            self.toast("Не подходит: " + ", ".join(bad))

    def add_dialog(self):
        start = self.settings.value("last_dir", str(Path.home()))
        files, _ = QFileDialog.getOpenFileNames(self, "Выберите картинку, GIF или видео", start,
                                                f"{MEDIA_FILTER};;Все файлы (*)")
        if files:
            self.settings.setValue("last_dir", str(Path(files[0]).parent))
            self.add_files(files)

    def remove_mask(self, mid):
        if self.gallery.current == mid:
            self.set_mask(None)
        self.library.remove(mid)
        self.thumbs.pop(mid, None)
        self.refresh_gallery()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(u.isLocalFile() and Path(u.toLocalFile()).suffix.lower() in MEDIA_EXT
                                          for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_files([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])

    # ── прочее ──

    def on_mode(self, mode):
        self.worker.mode = mode
        media = mode == "media"
        self.media_box.setVisible(media)
        self.strength.setVisible(not media)
        self.strength.setEnabled(mode in ("blur", "pixel"))
        if media and not self.gallery.current:
            self.toast("Добавьте картинку, GIF или видео — пока лицо просто размыто")

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
    elif not IS_WIN:
        os.environ.setdefault("QT_QPA_PLATFORM", "wayland;xcb")
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
