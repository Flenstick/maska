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


# веки: пары «точка верхнего века — точка нижнего» от внешнего уголка к внутреннему (правый глаз, левый глаз)
EYES = [
    ([246, 161, 160, 159, 158, 157, 173], [7, 163, 144, 145, 153, 154, 155], (33, 133)),
    ([466, 388, 387, 386, 385, 384, 398], [249, 390, 373, 374, 380, 381, 382], (263, 362)),
]


def eye_openness(pts, eye):
    up, low, (c0, c1) = eye
    width = np.linalg.norm(pts[c0] - pts[c1]) + 1e-6
    return float(np.mean(np.linalg.norm(pts[up[2:5]] - pts[low[2:5]], axis=1)) / width)


def blink(pts, state):
    """Опускает веки на точках кадра, когда глаз закрывается.

    Сетка ловит закрытие глаза не до конца, поэтому степень закрытия считаем от обычного открытого глаза
    этого человека (state["eye_ref"] — медленно забываемый максимум) и дотягиваем веко сами: треугольники
    глаза схлопываются, а кожа верхнего века с фото растягивается вниз — получается моргание.
    """
    out = pts.copy()
    refs = state.setdefault("eye_ref", [0.0, 0.0])
    closes = state.setdefault("eye_close", [0.0, 0.0])
    for i, eye in enumerate(EYES):
        r = eye_openness(pts, eye)
        refs[i] = max(r, refs[i] * 0.997)                       # подстраивается под разрез глаз за пару секунд
        rel = r / refs[i] if refs[i] > 1e-4 else 1.0
        target = float(np.clip((0.72 - rel) / 0.32, 0.0, 1.0))  # 72 % от открытого — начинает, 40 % — закрыт
        # закрывается быстро, открывается чуть плавнее — как настоящее веко
        closes[i] += (target - closes[i]) * (0.85 if target > closes[i] else 0.55)
        b = closes[i]
        if b < 0.02:
            continue
        up, low, _ = eye
        line = pts[low] + (pts[up] - pts[low]) * 0.18           # линия сомкнутых век — чуть выше нижнего
        out[up] = pts[up] + (line - pts[up]) * b
        out[low] = pts[low] + (line - pts[low]) * b
    return out


def turn(pts):
    """Поворот головы вбок по точкам: 0 — анфас, ~1 — пол-оборота, больше 1 — профиль; знак — куда смотрит нос
    по x кадра. Меряем, где кончик носа лежит между краями щёк: у анфаса посередине, в профиль — за краем."""
    a, b, nose = pts[234].astype(np.float64), pts[454].astype(np.float64), pts[1].astype(np.float64)
    axis = b - a
    t = float(np.dot(nose - a, axis) / (np.dot(axis, axis) + 1e-6)) - 0.5
    return 2 * t * (1.0 if axis[0] >= 0 else -1.0)


def hide_head(frame, pts, amount, side):
    """Сильно размывает всю голову — волосы, уши, щёку и затылок за краем сетки.

    При повороте сетка покрывает только видимую половину лица, а остальное узнаваемо — его и прячем
    под заменённым лицом. amount 0..1 — насколько, side — куда смотрит нос (затылок с другой стороны).
    """
    h, w = frame.shape[:2]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    fh = float(y1 - y0)
    side = float(np.clip(side, -1.5, 1.5))
    # эллипс с запасом во все стороны: сетка в профиль сжимается, а голова — нет (макушка, нос, затылок)
    cx = (x0 + x1) / 2 - side * fh * 0.12
    cy = (y0 + y1) / 2 - fh * 0.18
    ax, ay = fh * (0.75 + abs(side) * 0.18), fh * 0.98
    bx0, by0 = int(max(0, cx - ax)), int(max(0, cy - ay))
    bx1, by1 = int(min(w, cx + ax)), int(min(h, cy + ay))
    if bx1 - bx0 < 8 or by1 - by0 < 8:
        return
    roi = frame[by0:by1, bx0:bx1]
    rh, rw = roi.shape[:2]
    sw, sh = max(2, rw // 24), max(2, rh // 24)  # размытие сильное — черты (очки, нос, брови) не угадываются
    blur = cv2.GaussianBlur(cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA), (0, 0), 1.5)
    blur = cv2.resize(blur, (rw, rh), interpolation=cv2.INTER_LINEAR)
    m = np.zeros((sh * 4, sw * 4), np.float32)  # край эллипса — в уменьшенном виде, он всё равно мягкий
    kx, ky = sw * 4 / rw, sh * 4 / rh
    cv2.ellipse(m, (round((cx - bx0) * kx), round((cy - by0) * ky)), (round(ax * kx * 0.92), round(ay * ky * 0.92)),
                0, 0, 360, float(amount), -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), max(1.0, sw * 4 * 0.06))
    m = cv2.resize(m, (rw, rh), interpolation=cv2.INTER_LINEAR)
    roi[:] = cv2.blendLinear(blur, roi, m, 1.0 - m)


class OneEuro:
    """Фильтр One Euro: на месте сглаживает сильно (дрожь уходит), при движении — слабо (маска не отстаёт).

    Значения подаются в долях размера лица, поэтому настройки одинаково работают на близком и далёком лице.
    """

    def __init__(self, min_cutoff=0.5, beta=50.0, d_cutoff=1.0):  # подобрано замером: дрожь ×5 меньше, отставание ~1 px
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        r = 2 * np.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, x, t):
        x = np.asarray(x, np.float64)
        if self.x is None or self.x.shape != x.shape or t - self.t > 0.5:  # первый кадр или долгий перерыв
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(1e-3, t - self.t)
        self.t = t
        dx = (x - self.x) / dt
        self.dx += self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        self.x += self._alpha(cutoff, dt) * (x - self.x)
        return self.x.copy()


def track_points(mesh, img, box, angle, state, now):
    """Точки лица для дорожки с памятью между кадрами (state).

    Область для сетки берётся по точкам прошлого кадра (как в MediaPipe) — рамка детектора прыгает на пару
    пикселей каждый кадр, а с ней и точки. Результат сглаживается One Euro. None — сетка лица не легла.
    """
    prev = state.get("pts")
    if prev is not None and now - state.get("t", -1) < 0.25:
        x0, y0 = prev.min(0)
        x1, y1 = prev.max(0)
        er, el = prev[33], prev[263]
        score, pts = mesh.run(img, (x0, y0, x1 - x0, y1 - y0),
                              float(np.degrees(np.arctan2(el[1] - er[1], el[0] - er[0]))), scale=1.45)
        if score < 0:  # потеряли — пробуем по рамке детектора
            score, pts = mesh.run(img, box, angle)
    else:
        score, pts = mesh.run(img, box, angle)
    if score < 0:
        state.pop("pts", None)
        return None
    state["pts"], state["t"] = pts, now
    size = state.setdefault("ref", float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1.0)))
    filt = state.setdefault("filter", OneEuro())
    return (filt(pts / size, now) * size).astype(np.float32)


class FaceMesh:
    def __init__(self, model):
        engine = getattr(cv2.dnn, "ENGINE_CLASSIC", None)
        self.net = (cv2.dnn.readNetFromTFLite(str(model), engine=engine) if engine is not None
                    else cv2.dnn.readNetFromTFLite(str(model)))
        self.outs = self.net.getUnconnectedOutLayersNames()

    def run(self, img, box, angle=0.0, scale=1.6):
        """(уверенность, 468 точек в координатах img). Уверенность — логит: у лиц +15…+25, у рук и фона < 0."""
        x, y, w, h = box[:4]
        cx, cy = x + w / 2, y + h / 2
        side = max(w, h) * scale
        if side < 16:
            return -99.0, None
        m = cv2.getRotationMatrix2D((cx, cy), angle, MESH_SIZE / side)
        m[:, 2] += (MESH_SIZE / 2 - cx, MESH_SIZE / 2 - cy)
        crop = cv2.warpAffine(img, m, (MESH_SIZE, MESH_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        self.net.setInput(cv2.dnn.blobFromImage(crop, 1 / 255.0, (MESH_SIZE, MESH_SIZE), swapRB=True))
        res = self.net.forward(self.outs)
        lm = next(r for r in res if r.size == 1404)
        score = next((float(r.ravel()[0]) for r in res if r.size == 1), 10.0)
        pts = lm.reshape(-1, 3)[:, :2].astype(np.float64)
        inv = cv2.invertAffineTransform(m)
        return score, (pts @ inv[:, :2].T + inv[:, 2]).astype(np.float32)

    def landmarks(self, img, box, angle=0.0, scale=1.6):
        """468 точек (x, y) в координатах img для лица в box=(x, y, w, h) с наклоном angle; None — лица нет."""
        score, pts = self.run(img, box, angle, scale)
        return pts if score >= 0 else None

    def is_face(self, img, box, angle=0.0):
        return self.run(img, box, angle)[0] >= FACE_SCORE


FACE_SCORE = 2.0  # порог логита сетки: лица на проверке давали +18…+25, фон, руки и одежда — от −3.5 и ниже


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
        self.tris = self._delaunay(pts, img.shape)
        # от фото нужна только область лица с запасом — её дешевле перекрашивать под свет в кадре
        h, w = img.shape[:2]
        (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
        mx, my = (x1 - x0) * 0.1 + 4, (y1 - y0) * 0.1 + 4
        x0, y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
        x1, y1 = int(min(w, x1 + mx)), int(min(h, y1 + my))
        self.img = np.ascontiguousarray(img[y0:y1, x0:x1])
        self.pts = (pts - (x0, y0)).astype(np.float32)

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


def color_stats(src, dst, mask):
    """Как привести цвет src к dst внутри маски (LAB): (сдвиг, множитель) по каналам; None — маска пустая."""
    m8 = (mask > 0.5).astype(np.uint8)
    if cv2.countNonZero(m8) < 50:
        return None
    sm, ss = (v.ravel() for v in cv2.meanStdDev(cv2.cvtColor(src, cv2.COLOR_BGR2LAB), mask=m8))
    dm, ds = (v.ravel() for v in cv2.meanStdDev(cv2.cvtColor(dst, cv2.COLOR_BGR2LAB), mask=m8))
    ratio = np.clip((ds + 1e-3) / (ss + 1e-3), 0.6, 1.6)
    return np.concatenate([dm - sm * ratio, ratio])


def recolor(img, params):
    """Перекрашивает картинку в LAB таблицей LUT по параметрам color_stats."""
    x = np.arange(256, dtype=np.float64)[:, None]
    lut = np.clip(x * params[3:] + params[:3], 0, 255).astype(np.uint8).reshape(256, 1, 3)
    return cv2.cvtColor(cv2.LUT(cv2.cvtColor(img, cv2.COLOR_BGR2LAB), lut), cv2.COLOR_LAB2BGR)


_GRIDS = {}


def _grid(qh, qw, q):
    """Координаты пикселей карты: одна большая сетка на шаг q, из неё берётся угол нужного размера."""
    g = _GRIDS.get(q)
    if g is None or g[0].shape[0] < qh or g[0].shape[1] < qw:
        gh, gw = max(qh, 64) * 3 // 2, max(qw, 64) * 3 // 2
        if g is not None:
            gh, gw = max(gh, g[0].shape[0]), max(gw, g[0].shape[1])
        ys, xs = np.mgrid[0:gh, 0:gw].astype(np.float32) * q
        g = _GRIDS[q] = ys, xs
    return g[0][:qh, :qw], g[1][:qh, :qw]


def _shrink(mask, r):
    """Эрозия квадратом радиуса r — то же, что r проходов ядром 3×3, но одним проходом."""
    return cv2.erode(mask, np.ones((2 * r + 1, 2 * r + 1), np.uint8)) if r > 0 else mask


def swap_face(frame, dst_pts, source, feather=0.5, match_color=True, keep_mouth=True, state=None):
    """Рисует лицо source поверх лица с точками dst_pts прямо в frame. Возвращает False, если не вышло.

    state — словарь, живущий вместе с лицом между кадрами (для моргания); None — без моргания.
    """
    if state is not None:
        dst_pts = blink(dst_pts, state)
    h, w = frame.shape[:2]
    x0, y0 = np.floor(dst_pts.min(0)).astype(int) - 4
    x1, y1 = np.ceil(dst_pts.max(0)).astype(int) + 4
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    rw, rh = x1 - x0, y1 - y0
    if rw < 12 or rh < 12:
        return False
    local = dst_pts - (x0, y0)
    tris = source.tris
    # карта «какой треугольник в каждом пикселе» (0 — вне лица); у крупного лица считаем её в уменьшенном
    # разрешении и растягиваем — внутри треугольника карта линейная, разницы не видно, а быстрее в разы
    q = max(1, int(max(rw, rh) / 180))
    qw, qh = (rw + q - 1) // q, (rh + q - 1) // q
    idx = np.zeros((qh, qw), np.int32)
    corners = np.round(local / q).astype(np.int32)[tris]
    # схлопнутые треугольники (сомкнутые веки, профиль) не рисуем: их матрица вырождена и тащит случайные пиксели
    d = local[tris]
    area = np.abs((d[:, 1, 0] - d[:, 0, 0]) * (d[:, 2, 1] - d[:, 0, 1])
                  - (d[:, 2, 0] - d[:, 0, 0]) * (d[:, 1, 1] - d[:, 0, 1])) / 2
    for n in np.flatnonzero(area > 1.5):
        cv2.fillConvexPoly(idx, corners[n], int(n) + 1)
    aff = _affines(source.pts[tris].astype(np.float64), local[tris].astype(np.float64))
    aff = np.concatenate([np.zeros((1, 2, 3)), aff]).astype(np.float32).reshape(-1, 6)
    ys, xs = _grid(qh, qw, q)
    a = aff[idx]                                     # (qh, qw, 6)
    mx = a[..., 0] * xs + a[..., 1] * ys + a[..., 2]
    my = a[..., 3] * xs + a[..., 4] * ys + a[..., 5]
    inside_q = (idx > 0).astype(np.uint8)
    if q > 1:
        mx = cv2.resize(mx, (rw, rh), interpolation=cv2.INTER_LINEAR)
        my = cv2.resize(my, (rw, rh), interpolation=cv2.INTER_LINEAR)
        inside = cv2.resize(inside_q, (rw, rh), interpolation=cv2.INTER_NEAREST)
    else:
        inside = inside_q
    warped = cv2.remap(source.img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    roi = frame[y0:y1, x0:x1]
    outside = (inside == 0).view(np.uint8)
    cv2.copyTo(roi, outside, warped)  # вне сетки — кадр, иначе размытый край маски затянет туда угол фото

    # маска, её мягкий край и подгонка цвета — в уменьшенном виде: край всё равно размыт, а это в разы быстрее
    k = min(1.0, 160 / max(rw, rh))
    sw, sh = max(1, round(rw * k)), max(1, round(rh * k))
    small = local * (sw / rw, sh / rh)
    mask = np.zeros((sh, sw), np.float32)
    cv2.fillPoly(mask, [np.round(small[FACE_OVAL]).astype(np.int32)], 1.0)
    mask *= cv2.resize(inside_q, (sw, sh), interpolation=cv2.INTER_NEAREST)
    if keep_mouth:
        cv2.fillPoly(mask, [np.round(small[INNER_LIPS]).astype(np.int32)], 0.0)
    size = max(sw, sh)
    if match_color:
        # цвет сравниваем по маленьким копиям; перекрашивается само фото и только когда свет заметно сменился,
        # а между кадрами поправка сглаживается — так дешевле и лицо не мерцает от шума камеры
        st = state if state is not None else {}
        st["n"] = tick = st.get("n", -1) + 1
        params = None
        if tick % 3 == 0 or "color" not in st:  # свет меняется медленно — сверяем цвет раз в три кадра
            core = _shrink(mask, max(1, int(size * 0.02)))
            params = color_stats(cv2.resize(warped, (sw, sh), interpolation=cv2.INTER_LINEAR),
                                 cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR), core)
        elif st["color"][0] is source:
            params = st["color"][1]
        if params is not None:
            prev = st.get("color")
            if prev is not None and prev[0] is source:
                params = prev[1] + (params - prev[1]) * 0.25
            st["color"] = (source, params)
            done = st.get("recolored")
            if (done is None or done[0] is not source
                    or np.abs(done[1][:3] - params[:3]).max() > 1.5 or np.abs(done[1][3:] - params[3:]).max() > 0.02):
                done = st["recolored"] = (source, params, recolor(source.img, params))
            warped = cv2.remap(done[2], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            cv2.copyTo(roi, outside, warped)
    # мягкий край: сначала сжимаем, потом размываем — шов уходит внутрь лица, а не на фон
    f = 0.01 + feather * 0.05
    soft = _shrink(mask, max(1, int(size * f * 0.6)))
    soft = cv2.GaussianBlur(soft, (0, 0), max(0.6, size * f))
    soft = cv2.resize(soft, (rw, rh), interpolation=cv2.INTER_LINEAR)
    roi[:] = cv2.blendLinear(warped, roi, soft, 1.0 - soft)
    return True


def load_image(path):
    """Чтение картинки, которое не спотыкается о кириллицу в пути на Windows."""
    data = np.fromfile(str(Path(path)), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("не удалось открыть картинку")
    return img
