"""Эффекты поверх кадра по точкам лица. Первый — лазеры из глаз.

Эффект получает кадр, 468 точек лица (сетка MediaPipe), память дорожки между кадрами и время.
Свечение рисуется в слое в 1/4 разрешения (размытие там дешёвое), растягивается и прибавляется к кадру.
"""
import math

import cv2
import numpy as np

from face import EYES, OneEuro

# название, цвет BGR (для заголовка в окне — RGB хекс)
LASER_COLORS = {
    "red": ((40, 40, 255), "#ff2828"),
    "green": ((60, 255, 90), "#5aff3c"),
    "blue": ((255, 140, 40), "#288cff"),
    "purple": ((255, 60, 200), "#c83cff"),
}

EFFECTS = [("lasers", "Лазеры из глаз")]


def eye_centers(pts):
    out = []
    for up, low, _ in EYES:
        out.append(np.vstack([pts[up], pts[low]]).mean(0))
    return out


def head_direction(pts, state, now):
    """Куда смотрит голова на экране: сдвиг кончика носа от центра лица, сглаженный."""
    oval = pts[[10, 152, 234, 454]]
    cx, cy = (oval[2, 0] + oval[3, 0]) / 2, (oval[0, 1] + oval[1, 1]) / 2
    fw = max(1.0, np.linalg.norm(oval[3] - oval[2]))
    fh = max(1.0, np.linalg.norm(oval[1] - oval[0]))
    nose = pts[1]
    d = np.array([(nose[0] - cx) / fw, (nose[1] - cy) / fh - 0.08])  # анфас нос чуть ниже центра — поправка
    filt = state.setdefault("dir_filter", OneEuro(0.8, 10.0))
    return filt(d, now)


def _edge_point(p, d, w, h):
    """Точка, где луч из p в направлении d выходит за край кадра (с запасом)."""
    d = d / (np.linalg.norm(d) + 1e-9)
    ts = []
    for k, lim in ((0, w), (1, h)):
        if d[k] > 1e-6:
            ts.append((lim + 50 - p[k]) / d[k])
        elif d[k] < -1e-6:
            ts.append((-50 - p[k]) / d[k])
    t = min(ts) if ts else max(w, h)
    return p + d * t


def lasers(frame, pts, state, now, color="red"):
    bgr = np.array(LASER_COLORS.get(color, LASER_COLORS["red"])[0], np.float32)
    h, w = frame.shape[:2]
    q, c = 4, 2  # свечение — в 1/4 разрешения, сердцевина — в 1/2: на 720p эффект укладывается в пару мс
    lw, lh = (w + q - 1) // q, (h + q - 1) // q
    layer = np.zeros((lh, lw), np.float32)
    core = np.zeros(((h + c - 1) // c, (w + c - 1) // c), np.uint8)

    eyes = eye_centers(pts)
    eye_w = np.linalg.norm(pts[33] - pts[133]) + 1.0
    face_c = pts[[10, 152, 234, 454]].mean(0)
    look = head_direction(pts, state, now)
    flick = 0.85 + 0.15 * math.sin(now * 37.0) * math.sin(now * 13.0 + 1.3)  # лёгкое мерцание

    for eye in eyes:
        out = np.array([np.sign(eye[0] - face_c[0]) * 0.35, 0.22])  # в анфас лучи расходятся в стороны и вниз
        end = _edge_point(eye, look * 3.0 + out, w, h)
        a, b = tuple(np.round(eye / q).astype(int)), tuple(np.round(end / q).astype(int))
        cv2.line(layer, a, b, 1.0, max(1, int(eye_w * 0.55 * flick / q)), cv2.LINE_AA)
        cv2.circle(layer, a, max(2, int(eye_w * 0.9 * flick / q)), 1.4, -1, cv2.LINE_AA)
        a, b = tuple(np.round(eye / c).astype(int)), tuple(np.round(end / c).astype(int))
        cv2.line(core, a, b, 255, max(1, int(eye_w * 0.12 * flick / c)), cv2.LINE_AA)
        cv2.circle(core, a, max(1, int(eye_w * 0.28 * flick / c)), 255, -1, cv2.LINE_AA)

    glow = cv2.GaussianBlur(layer, (0, 0), max(1.0, eye_w * 0.35 / q))
    glow += cv2.GaussianBlur(layer, (0, 0), max(2.0, eye_w * 1.2 / q)) * 0.6
    colored = np.clip(glow[..., None] * bgr[None, None], 0, 255).astype(np.uint8)  # цвет — в свечении
    core = cv2.GaussianBlur(core, (0, 0), max(0.8, eye_w * 0.04 / c))
    light = cv2.add(cv2.resize(colored, (w, h), interpolation=cv2.INTER_LINEAR),
                    cv2.cvtColor(cv2.resize(core, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR))
    cv2.add(frame, light, dst=frame)  # белая сердцевина и цветное свечение, как у настоящего лазера


def apply(frame, pts, state, now, enabled, opts):
    if "lasers" in enabled:
        lasers(frame, pts, state, now, opts.get("laser_color", "red"))
