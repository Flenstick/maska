"""Замена лица: чужое лицо с фото натягивается на лицо в кадре по 468 точкам.

Точки ищет сетка лица MediaPipe (face_landmark.tflite, Apache-2.0) через OpenCV DNN — без mediapipe и onnxruntime.
Деформация — кусочно-аффинная по треугольникам Делоне, построенным на фото-источнике: вместо сотни мелких
warpAffine рисуется карта «номер треугольника» и делается один cv2.remap. Цвет кожи подгоняется в LAB,
внутренность рта остаётся из кадра — так речь выглядит живой.
"""
from pathlib import Path

import cv2
import numpy as np

MESH_SIZE = 192
# внутренний контур губ в сетке MediaPipe — эту область не заменяем, чтобы были видны настоящие зубы и язык
INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
# контур лица — по нему маска (выпуклая оболочка всех точек захватывает лишнее у висков)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


class FaceMesh:
    def __init__(self, model):
        engine = getattr(cv2.dnn, "ENGINE_CLASSIC", None)
        self.net = (cv2.dnn.readNetFromTFLite(str(model), engine=engine) if engine is not None
                    else cv2.dnn.readNetFromTFLite(str(model)))
        self.outs = self.net.getUnconnectedOutLayersNames()

    def landmarks(self, img, box, angle=0.0, scale=1.6):
        """468 точек (x, y) в координатах img для лица в box=(x, y, w, h) с наклоном angle; None — лица нет."""
        x, y, w, h = box[:4]
        cx, cy = x + w / 2, y + h / 2
        side = max(w, h) * scale
        if side < 16:
            return None
        m = cv2.getRotationMatrix2D((cx, cy), angle, MESH_SIZE / side)
        m[:, 2] += (MESH_SIZE / 2 - cx, MESH_SIZE / 2 - cy)
        crop = cv2.warpAffine(img, m, (MESH_SIZE, MESH_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        self.net.setInput(cv2.dnn.blobFromImage(crop, 1 / 255.0, (MESH_SIZE, MESH_SIZE), swapRB=True))
        res = self.net.forward(self.outs)
        lm = next(r for r in res if r.size == 1404)
        score = next((float(r.ravel()[0]) for r in res if r.size == 1), 10.0)
        if score < 0:  # логит уверенности: < 0 — в кропе не лицо
            return None
        pts = lm.reshape(-1, 3)[:, :2].astype(np.float64)
        inv = cv2.invertAffineTransform(m)
        return (pts @ inv[:, :2].T + inv[:, 2]).astype(np.float32)


def detect_main_face(img, yunet_model):
    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN.create(str(yunet_model), "", (w, h), 0.6, 0.3, 20)
    _, faces = det.detect(img)
    if faces is None:
        return None
    f = max(faces, key=lambda r: r[2] * r[3])
    return [float(v) for v in f[:4]], float(np.degrees(np.arctan2(f[7] - f[5], f[6] - f[4])))


class FaceSource:
    """Лицо с фото: картинка, точки и треугольники."""

    def __init__(self, img, mesh, yunet_model):
        h, w = img.shape[:2]
        k = 1024 / max(h, w)
        if k < 1:
            img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        found = detect_main_face(img, yunet_model)
        if found is None:
            raise ValueError("на фото не нашлось лицо")
        box, angle = found
        pts = mesh.landmarks(img, box, angle)
        if pts is None:
            raise ValueError("лицо на фото слишком мелкое или повёрнуто")
        # второй проход по уточнённой рамке из самих точек — точнее, как в трекинге MediaPipe
        refined = self._refine(img, mesh, pts)
        if refined is not None:
            pts = refined
        self.img = img
        self.pts = pts
        self.tris = self._delaunay(pts, img.shape)

    @staticmethod
    def _refine(img, mesh, pts):
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        eye_r, eye_l = pts[33], pts[263]
        angle = float(np.degrees(np.arctan2(eye_l[1] - eye_r[1], eye_l[0] - eye_r[0])))
        return mesh.landmarks(img, (x0, y0, x1 - x0, y1 - y0), angle, scale=1.35)

    @staticmethod
    def _delaunay(pts, shape):
        h, w = shape[:2]
        sub = cv2.Subdiv2D((0, 0, w, h))
        index = {}
        for i, (x, y) in enumerate(pts):
            x, y = float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))
            index.setdefault((round(x, 2), round(y, 2)), i)
            sub.insert((x, y))
        tris = []
        for t in sub.getTriangleList():
            ids = [index.get((round(float(t[j]), 2), round(float(t[j + 1]), 2))) for j in (0, 2, 4)]
            if None not in ids:
                tris.append(ids)
        return np.array(tris, np.int32)


def _affines(src, dst):
    """Аффинные матрицы dst→src для всех треугольников сразу: (T, 2, 3)."""
    ones = np.ones((*dst.shape[:2], 1))
    a = np.concatenate([dst, ones], 2)             # (T, 3, 3): строки — вершины [x y 1]
    b = src                                         # (T, 3, 2)
    a = a + np.eye(3)[None] * 1e-9                  # вырожденные треугольники не роняют solve
    m = np.linalg.solve(a, b)                       # (T, 3, 2): [x y 1] @ m = src
    return np.transpose(m, (0, 2, 1))               # (T, 2, 3)


def color_transfer(src, dst, mask):
    """Среднее и разброс цвета src (LAB) приводятся к dst внутри маски — таблицей LUT, это быстро."""
    m8 = (mask > 0.5).astype(np.uint8)
    if cv2.countNonZero(m8) < 50:
        return src
    s = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    d = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)
    sm, ss = (v.ravel() for v in cv2.meanStdDev(s, mask=m8))
    dm, ds = (v.ravel() for v in cv2.meanStdDev(d, mask=m8))
    ratio = np.clip((ds + 1e-3) / (ss + 1e-3), 0.6, 1.6)
    x = np.arange(256, dtype=np.float32)[:, None]
    lut = np.clip((x - sm) * ratio + dm, 0, 255).astype(np.uint8).reshape(256, 1, 3)
    return cv2.cvtColor(cv2.LUT(s, lut), cv2.COLOR_LAB2BGR)


def swap_face(frame, dst_pts, source, feather=0.5, match_color=True, keep_mouth=True):
    """Рисует лицо source поверх лица с точками dst_pts прямо в frame. Возвращает False, если не вышло."""
    h, w = frame.shape[:2]
    x0, y0 = np.floor(dst_pts.min(0)).astype(int) - 4
    x1, y1 = np.ceil(dst_pts.max(0)).astype(int) + 4
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    rw, rh = x1 - x0, y1 - y0
    if rw < 12 or rh < 12:
        return False
    local = dst_pts - (x0, y0)
    tris = source.tris
    # карта «какой треугольник в каждом пикселе» (0 — вне лица); у крупного лица считаем её в половинном
    # разрешении и растягиваем — внутри треугольника карта линейная, разницы не видно, а быстрее в 3–4 раза
    q = 2 if rw * rh > 30000 else 1
    qw, qh = (rw + q - 1) // q, (rh + q - 1) // q
    idx = np.zeros((qh, qw), np.int32)
    corners = np.round(local / q).astype(np.int32)[tris]
    for n, c in enumerate(corners, 1):
        cv2.fillConvexPoly(idx, c, n)
    aff = _affines(source.pts[tris].astype(np.float64), local[tris].astype(np.float64))
    aff = np.concatenate([np.zeros((1, 2, 3)), aff]).astype(np.float32).reshape(-1, 6)
    ys, xs = np.mgrid[0:qh, 0:qw].astype(np.float32) * q
    a = aff[idx]                                     # (qh, qw, 6)
    mx = a[..., 0] * xs + a[..., 1] * ys + a[..., 2]
    my = a[..., 3] * xs + a[..., 4] * ys + a[..., 5]
    if q > 1:
        mx = cv2.resize(mx, (rw, rh), interpolation=cv2.INTER_LINEAR)
        my = cv2.resize(my, (rw, rh), interpolation=cv2.INTER_LINEAR)
        idx = cv2.resize(idx.astype(np.float32), (rw, rh), interpolation=cv2.INTER_NEAREST)
    warped = cv2.remap(source.img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    roi = frame[y0:y1, x0:x1]
    outside = idx == 0
    warped[outside] = roi[outside]  # вне сетки — кадр, иначе размытый край маски затянет туда угол фото

    mask = np.zeros((rh, rw), np.float32)
    cv2.fillPoly(mask, [np.round(local[FACE_OVAL]).astype(np.int32)], 1.0)
    mask *= (idx > 0)
    if keep_mouth:
        cv2.fillPoly(mask, [np.round(local[INNER_LIPS]).astype(np.int32)], 0.0)
    size = max(rw, rh)
    core = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=max(1, int(size * 0.02)))
    if match_color:
        warped = color_transfer(warped, roi, core)
    # мягкий край: сначала сжимаем, потом размываем — шов уходит внутрь лица, а не на фон
    k = 0.01 + feather * 0.05
    soft = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=max(1, int(size * k * 0.6)))
    soft = cv2.GaussianBlur(soft, (0, 0), max(1.0, size * k))
    roi[:] = cv2.blendLinear(warped, roi, soft, 1.0 - soft)
    return True


def load_image(path):
    """Чтение картинки, которое не спотыкается о кириллицу в пути на Windows."""
    data = np.fromfile(str(Path(path)), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("не удалось открыть картинку")
    return img
