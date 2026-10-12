"""Окно «Обрезать маску»: рамка кадрирования по картинке и, для видео и GIF, отрезок по времени.

Правки не трогают файл — сохраняются в masks.json и применяются при чтении маски.
Зависимости из maska.py (класс маски, переключатель, палитра) передаются в конструктор: maska.py часто
запускается как __main__, и `import maska` отсюда загрузил бы его второй раз.
"""
import time

from PyQt6.QtCore import QMargins, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QImage, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

HANDLE = 9  # полуразмер зоны захвата ручки, px


def to_qimage(bgra):
    h, w = bgra.shape[:2]
    return QImage(bgra.tobytes(), w, h, w * 4, QImage.Format.Format_ARGB32).copy()


def fmt_time(t):
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m)}:{s:04.1f}"


class CropView(QWidget):
    changed = pyqtSignal()

    def __init__(self, colors):
        super().__init__()
        self.c = colors
        self.img = None
        self.crop = [0.0, 0.0, 1.0, 1.0]
        self.aspect = None  # отношение сторон рамки в пикселях кадра или None
        self.drag = None
        self.setMouseTracking(True)
        self.setMinimumSize(560, 360)

    def set_image(self, img):
        self.img = img
        self.update()

    def frame_rect(self):
        if self.img is None:
            return QRectF(self.rect())
        sz = self.img.size().scaled(self.size().shrunkBy(QMargins(14, 14, 14, 14)), Qt.AspectRatioMode.KeepAspectRatio)
        return QRectF((self.width() - sz.width()) / 2, (self.height() - sz.height()) / 2, sz.width(), sz.height())

    def crop_rect(self):
        f = self.frame_rect()
        x, y, w, h = self.crop
        return QRectF(f.x() + x * f.width(), f.y() + y * f.height(), w * f.width(), h * f.height())

    def handles(self):
        r = self.crop_rect()
        xs = {"l": r.left(), "c": r.center().x(), "r": r.right()}
        ys = {"t": r.top(), "m": r.center().y(), "b": r.bottom()}
        return {v + h: QPointF(xs[h], ys[v]) for v in "tmb" for h in "lcr" if v + h != "mc"}

    def hit(self, pos):
        for name, pt in self.handles().items():
            if abs(pos.x() - pt.x()) <= HANDLE and abs(pos.y() - pt.y()) <= HANDLE:
                return name
        return "move" if self.crop_rect().contains(pos) else "new"

    CURSORS = {"tl": Qt.CursorShape.SizeFDiagCursor, "br": Qt.CursorShape.SizeFDiagCursor,
               "tr": Qt.CursorShape.SizeBDiagCursor, "bl": Qt.CursorShape.SizeBDiagCursor,
               "tc": Qt.CursorShape.SizeVerCursor, "bc": Qt.CursorShape.SizeVerCursor,
               "ml": Qt.CursorShape.SizeHorCursor, "mr": Qt.CursorShape.SizeHorCursor,
               "move": Qt.CursorShape.SizeAllCursor, "new": Qt.CursorShape.CrossCursor}

    def to_norm(self, pos):
        f = self.frame_rect()
        return (min(1.0, max(0.0, (pos.x() - f.x()) / f.width())),
                min(1.0, max(0.0, (pos.y() - f.y()) / f.height())))

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton or self.img is None:
            return
        kind = self.hit(e.position())
        self.drag = (kind, self.to_norm(e.position()), list(self.crop))

    def mouseMoveEvent(self, e):
        if not self.drag:
            self.setCursor(QCursor(self.CURSORS[self.hit(e.position())]))
            return
        kind, (sx, sy), (x, y, w, h) = self.drag
        nx, ny = self.to_norm(e.position())
        if kind == "move":
            dx = min(max(nx - sx, -x), 1 - x - w)
            dy = min(max(ny - sy, -y), 1 - y - h)
            self.crop = [x + dx, y + dy, w, h]
        else:
            if kind == "new":
                x0, y0, x1, y1 = min(sx, nx), min(sy, ny), max(sx, nx), max(sy, ny)
                anchor = ("l" if nx >= sx else "r", "t" if ny >= sy else "b")
            else:
                x0, y0, x1, y1 = x, y, x + w, y + h
                if "l" in kind[1]:
                    x0 = min(nx, x1 - 0.02)
                if "r" in kind[1]:
                    x1 = max(nx, x0 + 0.02)
                if kind[0] == "t":
                    y0 = min(ny, y1 - 0.02)
                if kind[0] == "b":
                    y1 = max(ny, y0 + 0.02)
                anchor = ("r" if "l" in kind[1] else "l", "b" if kind[0] == "t" else "t")
            self.crop = self.fit_aspect([x0, y0, x1 - x0, y1 - y0], anchor, kind)
        self.changed.emit()
        self.update()

    def mouseReleaseEvent(self, _):
        if self.drag and self.crop[2] < 0.02:  # щелчок без протягивания — рамку не теряем
            self.crop = self.drag[2]
        self.drag = None
        self.changed.emit()
        self.update()

    def fit_aspect(self, r, anchor=("l", "t"), kind="new"):
        """Подгоняет рамку под выбранное соотношение сторон, не вылезая за кадр."""
        x, y, w, h = r
        if self.aspect and self.img is not None:
            iw, ih = self.img.width(), self.img.height()
            k = self.aspect * ih / iw  # нужное w/h в долях кадра
            if kind in ("tc", "bc"):
                w = h * k
            elif kind in ("ml", "mr"):
                h = w / k
            elif w / max(h, 1e-6) > k:
                w = h * k
            else:
                h = w / k
            if w > 1:
                w, h = 1.0, 1.0 / k
            if h > 1:
                h, w = 1.0, k
            if anchor[0] == "r":
                x = r[0] + r[2] - w
            if anchor[1] == "b":
                y = r[1] + r[3] - h
        x = min(max(0.0, x), 1 - w)
        y = min(max(0.0, y), 1 - h)
        return [x, y, min(w, 1.0), min(h, 1.0)]

    def set_aspect(self, aspect):
        self.aspect = aspect
        if aspect:
            x, y, w, h = self.crop
            cx, cy = x + w / 2, y + h / 2
            r = self.fit_aspect([x, y, w, h])
            r[0] = min(max(0.0, cx - r[2] / 2), 1 - r[2])
            r[1] = min(max(0.0, cy - r[3] / 2), 1 - r[3])
            self.crop = r
        self.changed.emit()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        bg = QPainterPath()
        bg.addRoundedRect(QRectF(self.rect()), 14, 14)
        p.fillPath(bg, QColor(self.c["PANEL"]))
        if self.img is None:
            return
        f = self.frame_rect()
        # шахматка под прозрачными картинками
        p.save()
        p.setClipRect(f)
        cell = 10
        for yy in range(int(f.top()), int(f.bottom()) + cell, cell):
            for xx in range(int(f.left()), int(f.right()) + cell, cell):
                if (xx // cell + yy // cell) % 2:
                    p.fillRect(QRectF(xx, yy, cell, cell), QColor(255, 255, 255, 10))
        p.restore()
        p.drawImage(f, self.img)
        r = self.crop_rect()
        shade = QPainterPath()
        shade.addRect(f)
        hole = QPainterPath()
        hole.addRect(r)
        p.fillPath(shade.subtracted(hole), QColor(5, 5, 7, 170))
        p.setPen(QPen(QColor(255, 255, 255, 60), 1))
        for i in (1, 2):
            p.drawLine(QPointF(r.left() + r.width() * i / 3, r.top()), QPointF(r.left() + r.width() * i / 3, r.bottom()))
            p.drawLine(QPointF(r.left(), r.top() + r.height() * i / 3), QPointF(r.right(), r.top() + r.height() * i / 3))
        p.setPen(QPen(QColor(self.c["ACCENT"]), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#ffffff"))
        for name, pt in self.handles().items():
            if len(set(name) & set("lr")) and len(set(name) & set("tb")):
                p.drawRoundedRect(QRectF(pt.x() - 6, pt.y() - 6, 12, 12), 3, 3)
            else:
                p.drawRoundedRect(QRectF(pt.x() - 5, pt.y() - 5, 10, 10), 5, 5)


class TrimBar(QWidget):
    """Лента отрезка: две ручки (начало, конец) и бегунок проигрывания."""
    changed = pyqtSignal()
    seeked = pyqtSignal(float)

    def __init__(self, colors, duration):
        super().__init__()
        self.c = colors
        self.dur = max(0.05, duration)
        self.start, self.end, self.pos = 0.0, self.dur, 0.0
        self.drag = None
        self.setFixedHeight(46)
        self.setMouseTracking(True)

    def track(self):
        return QRectF(12, 14, self.width() - 24, 20)

    def x_of(self, t):
        tr = self.track()
        return tr.left() + tr.width() * t / self.dur

    def t_of(self, x):
        tr = self.track()
        return min(self.dur, max(0.0, (x - tr.left()) / tr.width() * self.dur))

    def hit(self, x):
        if abs(x - self.x_of(self.start)) < 10:
            return "start"
        if abs(x - self.x_of(self.end)) < 10:
            return "end"
        return "pos"

    def mousePressEvent(self, e):
        self.drag = self.hit(e.position().x())
        self.mouseMoveEvent(e)

    def mouseMoveEvent(self, e):
        x = e.position().x()
        if not self.drag:
            self.setCursor(Qt.CursorShape.SizeHorCursor if self.hit(x) != "pos" else Qt.CursorShape.PointingHandCursor)
            return
        t = self.t_of(x)
        gap = min(0.2, self.dur / 2)
        if self.drag == "start":
            self.start = min(t, self.end - gap)
            self.pos = self.start
            self.changed.emit()
        elif self.drag == "end":
            self.end = max(t, self.start + gap)
            self.pos = max(self.start, self.end - 0.05)
            self.changed.emit()
        else:
            self.pos = min(max(t, self.start), self.end)
        self.seeked.emit(self.pos)
        self.update()

    def mouseReleaseEvent(self, _):
        self.drag = None

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        tr = self.track()
        path = QPainterPath()
        path.addRoundedRect(tr, 6, 6)
        p.fillPath(path, QColor(self.c["PANEL_2"]))
        sel = QRectF(self.x_of(self.start), tr.top(), self.x_of(self.end) - self.x_of(self.start), tr.height())
        sp = QPainterPath()
        sp.addRoundedRect(sel, 6, 6)
        acc = QColor(self.c["ACCENT"])
        soft = QColor(acc)
        soft.setAlpha(50)
        p.fillPath(sp, soft)
        p.setPen(QPen(acc, 2))
        p.drawPath(sp)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#ffffff"))
        for t in (self.start, self.end):
            x = self.x_of(t)
            p.drawRoundedRect(QRectF(x - 4, tr.top() - 5, 8, tr.height() + 10), 3, 3)
        x = self.x_of(self.pos)
        p.setBrush(QColor(self.c["INK"]))
        p.drawRoundedRect(QRectF(x - 1, tr.top() - 8, 2, tr.height() + 16), 1, 1)
        p.drawEllipse(QPointF(x, tr.top() - 9), 4, 4)


class EditDialog(QDialog):
    def __init__(self, parent, path, name, edit, media_cls, segmented_cls, colors):
        super().__init__(parent)
        self.setWindowTitle(f"Обрезать маску — {name}")
        self.resize(1000, 700)
        self.setStyleSheet(f"QDialog {{ background: {colors['BG']}; }}")
        self.c = colors
        self.media = media_cls(path, None, side=1280)
        self.kind = self.media.kind
        if self.kind == "video":
            self.dur = (self.media.stop or self.media.count or 1) / self.media.fps
        elif self.kind == "anim":
            self.dur = float(self.media.ends[-1])
        else:
            self.dur = 0.0
        edit = edit or {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)
        head = QHBoxLayout()
        title = QLabel("Обрезать маску", objectName="title")
        head.addWidget(title)
        head.addStretch()
        self.info = QLabel("", objectName="mono")
        head.addWidget(self.info)
        lay.addLayout(head)

        self.view = CropView(colors)
        self.view.crop = list(edit.get("crop") or [0.0, 0.0, 1.0, 1.0])
        lay.addWidget(self.view, 1)

        self.trim = None
        self.play_btn = None
        if self.kind != "image":
            row = QHBoxLayout()
            row.setSpacing(10)
            self.play_btn = QPushButton("▶")
            self.play_btn.setFixedWidth(44)
            self.play_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.play_btn.setToolTip("Проиграть выбранный отрезок (пробел)")
            self.play_btn.clicked.connect(self.toggle_play)
            row.addWidget(self.play_btn)
            self.trim = TrimBar(colors, self.dur)
            self.trim.start = min(float(edit.get("start") or 0.0), self.dur)
            self.trim.end = min(float(edit.get("end") or self.dur), self.dur)
            self.trim.pos = self.trim.start
            self.trim.changed.connect(self.refresh_info)
            self.trim.seeked.connect(self.show_at)
            row.addWidget(self.trim, 1)
            lay.addLayout(row)
            hint = QLabel("Белые ручки на ленте — начало и конец отрезка. Рамку на кадре можно тянуть за углы и края.",
                          objectName="caption")
            lay.addWidget(hint)
        else:
            lay.addWidget(QLabel("Тяните рамку за углы и края или обведите нужную часть заново.", objectName="caption"))

        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        self.aspects = segmented_cls([("free", "Свободно"), ("1", "1:1"), ("0.75", "3:4"), ("1.7778", "16:9")])
        self.aspects.setFixedWidth(300)
        self.aspects.changed.connect(lambda v: self.view.set_aspect(None if v == "free" else float(v)))
        bottom.addWidget(self.aspects)
        reset = QPushButton("Сбросить")
        reset.setCursor(Qt.CursorShape.PointingHandCursor)
        reset.clicked.connect(self.reset)
        bottom.addWidget(reset)
        bottom.addStretch()
        cancel = QPushButton("Отмена")
        cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel.clicked.connect(self.reject)
        bottom.addWidget(cancel)
        done = QPushButton("Готово", objectName="primary")
        done.setCursor(Qt.CursorShape.PointingHandCursor)
        done.setDefault(True)
        done.clicked.connect(self.accept)
        bottom.addWidget(done)
        lay.addLayout(bottom)

        self.view.changed.connect(self.refresh_info)
        self.timer = QTimer(self, interval=33)
        self.timer.timeout.connect(self.tick)
        self.play_t0 = None
        self.show_at(self.trim.start if self.trim else 0.0)
        self.refresh_info()

    def show_at(self, t):
        self.view.set_image(to_qimage(self.media.frame(t)))

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("▶")
        else:
            self.play_t0 = time.monotonic() - (self.trim.pos - self.trim.start)
            self.timer.start()
            self.play_btn.setText("❚❚")

    def tick(self):
        length = max(0.05, self.trim.end - self.trim.start)
        t = self.trim.start + (time.monotonic() - self.play_t0) % length
        self.trim.pos = t
        self.trim.update()
        self.show_at(t)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space and self.trim:
            self.toggle_play()
            return
        super().keyPressEvent(e)

    def reset(self):
        self.view.crop = [0.0, 0.0, 1.0, 1.0]
        self.aspects.select(0)
        if self.trim:
            self.trim.start, self.trim.end, self.trim.pos = 0.0, self.dur, 0.0
            self.trim.update()
        self.view.update()
        self.refresh_info()

    def refresh_info(self):
        parts = []
        img = self.view.img
        if img is not None:
            # размер исходника: картинка в окне уменьшена до 1280 по большей стороне
            parts.append(f"{round(self.view.crop[2] * img.width())}×{round(self.view.crop[3] * img.height())}")
        if self.trim:
            parts.append(f"{fmt_time(self.trim.start)} – {fmt_time(self.trim.end)}  ({self.trim.end - self.trim.start:.1f} с)")
        self.info.setText("  ·  ".join(parts))

    def result_edit(self):
        x, y, w, h = self.view.crop
        crop = None if (w > 0.995 and h > 0.995) else [round(v, 5) for v in (x, y, w, h)]
        out = {}
        if crop:
            out["crop"] = crop
        if self.trim:
            if self.trim.start > 0.01:
                out["start"] = round(self.trim.start, 3)
            if self.trim.end < self.dur - 0.01:
                out["end"] = round(self.trim.end, 3)
        return out

    def done(self, r):
        self.timer.stop()
        self.media.close()
        super().done(r)
