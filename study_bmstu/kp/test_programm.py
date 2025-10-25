import sys
import math
import random
import cv2
import numpy as np
import torch
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from scipy import interpolate


class RadarImageGenerator:
    """Класс для генерации радиолокационных изображений по методу обратного проецирования"""

    def __init__(self):
        # Параметры сигнала
        self.Fc = 300e6  # центральная несущая частота ЛЧМ-импульса
        self.cc = 3e8  # скорость света
        self.BW = 7e6  # девиация частоты ЛЧМ
        self.Fs = self.BW * 1 / 10000  # частота дискретизации
        self.Tp = 50 / self.Fs  # длительность сигнала

    def generate_radar_image(self, N_RLD=4, point_coord=None):
        """Генерация РЛИ методом обратного проецирования"""
        if point_coord is None:
            # Координаты блестящей точки по умолчанию
            beta = -5 * np.pi / 180
            R_scene = 300
            point_coord = np.array([[R_scene * np.tan(beta), R_scene]])

        # Временная и частотная сетка
        t = np.arange(0, self.Tp, 1 / self.Fs)
        f = np.linspace(self.Fc - self.BW / 2, self.Fc + self.BW / 2, len(t))
        lambda_vals = self.cc / f

        # Параметры апертуры
        N = 8  # число излучателей в апертуре
        L = 9  # длина апертуры (метров)
        delta_x = L / (N - 1)
        k = np.linspace(-L / 2, L / 2, N)

        # Пространственная сетка
        massiv_range_x = 100
        massiv_range_y = 100
        massiv_step = 0.5
        R_scene = 300

        # Координаты центра сцены
        center_x = 0
        center_y = R_scene

        # Массив координат
        massiv_x = np.arange(center_x - massiv_range_x, center_x + massiv_range_x + massiv_step, massiv_step)
        massiv_y = np.arange(center_y - massiv_range_y, center_y + massiv_range_y + massiv_step, massiv_step)
        massiv_x, massiv_y = np.meshgrid(massiv_x, massiv_y)

        # Полярная система координат
        azimut = np.arctan2(massiv_y, massiv_x) * 180 / np.pi
        range_vals = np.sqrt(massiv_x ** 2 + massiv_y ** 2)

        # Координаты излучателей
        dist_ant_start = np.zeros((N, 2, 4))
        ugol = [0, 0, 0, 0]  # углы разворота АР

        for j in range(N):
            dist_ant_start[j, :, 0] = [k[j] * np.cos(ugol[0]), 0]
            dist_ant_start[j, :, 1] = [k[j] * np.cos(ugol[1]), 35]
            dist_ant_start[j, :, 2] = [k[j] * np.cos(ugol[2]) - 10, 15]
            dist_ant_start[j, :, 3] = [k[j] * np.cos(ugol[3]) + 23, 20]

        # Генерация изображений
        Image = np.zeros((massiv_x.shape[0], massiv_x.shape[1], N_RLD), dtype=complex)

        for r in range(N_RLD):
            dist_ant = dist_ant_start[:, :, r]
            Vsum = np.zeros((N, len(t)), dtype=complex)

            # Моделирование сигналов
            for i in range(N):
                for n in range(len(point_coord)):
                    rc0 = np.sqrt((point_coord[n, 0] - dist_ant[i, 0]) ** 2 +
                                  (point_coord[n, 1] - dist_ant[i, 1]) ** 2)
                    Vc = np.exp(-1j * 2 * np.pi * rc0 / lambda_vals)
                    Vsum[i, :] += Vc

            # Обратное проецирование
            S0 = np.zeros((massiv_x.shape[0], massiv_x.shape[1]), dtype=complex)
            for z in range(N):
                r_c = np.sqrt((massiv_x - dist_ant[z, 0]) ** 2 +
                              (massiv_y - dist_ant[z, 1]) ** 2)
                for k1 in range(len(t)):
                    S = Vsum[z, k1] * np.exp(1j * 2 * np.pi * r_c / lambda_vals[k1])
                    S0 += S

            Image[:, :, r] = S0

        # Нормализация
        Image_normalized = np.zeros_like(Image, dtype=np.float32)
        for r in range(N_RLD):
            img_abs = np.abs(Image[:, :, r])
            Image_normalized[:, :, r] = img_abs / np.max(img_abs)

        return Image_normalized, massiv_x, massiv_y


def cv_to_qpixmap(cv_img):
    """Преобразует изображения OpenCV в QPixmap, поддерживает 1-, 3- и 4-канальные изображения.
    Ожидает формат: Gray, BGR или BGRA (uint8).
    """
    if cv_img is None:
        return None
    h, w = cv_img.shape[:2]
    if cv_img.ndim == 2:
        bytes_per_line = w
        qimg = QtGui.QImage(cv_img.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8)
    elif cv_img.shape[2] == 3:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    elif cv_img.shape[2] == 4:
        # OpenCV uses BGRA; Qt expects RGBA ordering for Format_RGBA8888
        bgra = cv_img
        rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
        bytes_per_line = 4 * w
        qimg = QtGui.QImage(rgba.data, w, h, bytes_per_line, QtGui.QImage.Format_RGBA8888)
    else:
        return None
    return QtGui.QPixmap.fromImage(qimg)


def add_gaussian_noise(img, sigma):
    noisy = img.astype(np.float32)
    gauss = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy += gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper(img, amount):
    out = img.copy()
    h, w = img.shape[:2]
    num = int(amount * h * w)
    # salt
    ys = np.random.randint(0, h, num)
    xs = np.random.randint(0, w, num)
    out[ys, xs] = 255
    # pepper
    ys = np.random.randint(0, h, num)
    xs = np.random.randint(0, w, num)
    out[ys, xs] = 0
    return out


def cartesian_to_polar(img, center=None, radius=None, size=(512, 512)):
    if img is None:
        return None
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = min(center[0], center[1])
    dst = cv2.warpPolar(img, size, center, radius, flags=cv2.WARP_POLAR_LINEAR)
    return dst


def polar_to_cartesian(polar_img, out_size=(512, 512), center=None, maxRadius=None):
    if polar_img is None:
        return None
    h_out, w_out = out_size
    if center is None:
        center = (w_out // 2, h_out // 2)
    if maxRadius is None:
        maxRadius = min(center[0], center[1])
    cart = cv2.warpPolar(polar_img, (w_out, h_out), center, maxRadius,
                         flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)
    return cart


def calc_entropy(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    s = hist.sum()
    if s == 0:
        return 0.0
    p = hist / s
    ent = -float(np.sum([pp * math.log2(pp) for pp in p if pp > 0]))
    return ent


def draw_pixel_plane(w=60, h=60):
    """Пиксельный силуэт самолета"""
    plane = np.zeros((h, w, 4), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    # фюзеляж
    cv2.rectangle(plane, (cx - 2, cy - 12), (cx + 2, cy + 12), (0, 0, 0, 255), -1)
    # крылья
    pts = np.array([[cx - 15, cy], [cx + 15, cy], [cx, cy + 4]])
    cv2.fillPoly(plane, [pts], (0, 0, 0, 255))
    # хвост
    cv2.line(plane, (cx, cy - 12), (cx - 8, cy - 18), (0, 0, 0, 255), 3)
    return plane


def overlay_rgba(base_bgr, overlay_rgba, x, y):
    if base_bgr is None or overlay_rgba is None:
        return base_bgr
    bh, bw = base_bgr.shape[:2]
    oh, ow = overlay_rgba.shape[:2]
    if x >= bw or y >= bh:
        return base_bgr
    w = min(ow, bw - x)
    h = min(oh, bh - y)
    if w <= 0 or h <= 0:
        return base_bgr
    base = base_bgr.copy().astype(np.float32) / 255.0
    overlay = overlay_rgba[:h, :w].astype(np.float32) / 255.0
    alpha = overlay[..., 3:4]
    overlay_rgb = overlay[..., :3]
    base_region = base[y:y + h, x:x + w]
    comp = (1.0 - alpha) * base_region + alpha * overlay_rgb
    base[y:y + h, x:x + w] = comp
    return (base * 255).astype(np.uint8)


def make_circular_alpha(bgr_img, diameter=None):
    """Возвращает BGRA-изображение с circular alpha mask (прозрачный за пределами круга).
    Если вход уже имеет 4 канала, альфа будет заменена.
    """
    if bgr_img is None:
        return None
    h, w = bgr_img.shape[:2]
    if diameter is None:
        diameter = min(h, w)
    # центр и радиус
    cx, cy = w // 2, h // 2
    radius = diameter // 2
    # Создаем BGRA
    if bgr_img.shape[2] == 4:
        bgra = bgr_img.copy()
    else:
        bgra = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2BGRA)
    # Создаем маску круга
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    # Применяем маску как альфа-канал
    bgra[..., 3] = mask
    return bgra


class ImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self.setMinimumSize(120, 120)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # делаем фон прозрачным
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")

    def set_image(self, cv_img):
        if cv_img is None:
            self._pixmap = None
        else:
            self._pixmap = cv_to_qpixmap(cv_img)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        # НЕ заливаем фон — прозрачный
        if self._pixmap:
            target = self.rect()
            scaled = self._pixmap.scaled(target.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            tx = target.x() + (target.width() - scaled.width()) // 2
            ty = target.y() + (target.height() - scaled.height()) // 2
            painter.drawPixmap(tx, ty, scaled)
        else:
            painter.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200)))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "Нет изображения")


class RadarApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Радарная обработка изображений — прозрачный Polar сверху")
        self.resize(1400, 800)

        # Инициализация генератора РЛИ
        self.radar_generator = RadarImageGenerator()

        # image data (BGR or BGRA for polar)
        self.base_img = None
        self.processed_img = None
        self.overlay_img = None
        self.clean_generated_img = None
        self.generated_flag = False
        self.generated_has_noise = False
        self.polar_img = None
        self.model = None

        self.setup_ui()

    def setup_ui(self):
        # --- Top: Polar (прозрачный круг) ---
        polar_label = QtWidgets.QLabel("Polar (Полярная СК)")
        polar_label.setAlignment(QtCore.Qt.AlignCenter)
        self.polar_widget = ImageWidget()
        polar_layout = QtWidgets.QVBoxLayout()
        polar_layout.addWidget(polar_label)
        polar_layout.addWidget(self.polar_widget)
        polar_frame = QtWidgets.QFrame()
        polar_frame.setLayout(polar_layout)

        # --- Bottom: Cartesian (Декартова СК) ---
        cart_label = QtWidgets.QLabel("Cartesian (Декартова СК)")
        cart_label.setAlignment(QtCore.Qt.AlignCenter)
        self.cart_widget = ImageWidget()
        cart_layout = QtWidgets.QVBoxLayout()
        cart_layout.addWidget(cart_label)
        cart_layout.addWidget(self.cart_widget)
        cart_frame = QtWidgets.QFrame()
        cart_frame.setLayout(cart_layout)

        images_layout = QtWidgets.QVBoxLayout()
        # Полярное изображение сверху
        images_layout.addWidget(polar_frame, 1)
        images_layout.addWidget(cart_frame, 2)
        images_container = QtWidgets.QWidget()
        images_container.setLayout(images_layout)

        # --- Right: controls + params ---
        ctrl_layout = QtWidgets.QVBoxLayout()

        # File / generate controls
        load_btn = QtWidgets.QPushButton("Загрузить изображение")
        load_btn.clicked.connect(self.load_image)
        save_btn = QtWidgets.QPushButton("Сохранить изображение")
        save_btn.clicked.connect(self.save_image)

        # Кнопки для генерации РЛИ
        gen_simple_btn = QtWidgets.QPushButton("Сгенерировать РЛИ (простой)")
        gen_simple_btn.clicked.connect(self.generate_simple_image)

        gen_radar_btn = QtWidgets.QPushButton("Сгенерировать РЛИ (радар)")
        gen_radar_btn.clicked.connect(self.generate_radar_image)

        ctrl_layout.addWidget(load_btn)
        ctrl_layout.addWidget(save_btn)
        ctrl_layout.addWidget(gen_simple_btn)
        ctrl_layout.addWidget(gen_radar_btn)
        ctrl_layout.addSpacing(8)

        # Noise controls
        noise_label = QtWidgets.QLabel("Добавить шум:")
        self.noise_combo = QtWidgets.QComboBox()
        self.noise_combo.addItems(["Нет", "Гауссов", "Соль-и-перец"])
        self.noise_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.noise_slider.setRange(0, 100)
        self.noise_slider.setValue(12)
        self.noise_val_label = QtWidgets.QLabel("Интенсивность: 12")
        self.noise_slider.valueChanged.connect(lambda v: self.noise_val_label.setText(f"Интенсивность: {v}"))
        add_noise_btn = QtWidgets.QPushButton("Добавить шум")
        add_noise_btn.clicked.connect(self.apply_noise)

        remove_noise_btn = QtWidgets.QPushButton("Убрать сгенерированный шум")
        remove_noise_btn.clicked.connect(self.remove_generated_noise)

        ctrl_layout.addWidget(noise_label)
        ctrl_layout.addWidget(self.noise_combo)
        ctrl_layout.addWidget(self.noise_slider)
        ctrl_layout.addWidget(self.noise_val_label)
        ctrl_layout.addWidget(add_noise_btn)
        ctrl_layout.addWidget(remove_noise_btn)
        ctrl_layout.addSpacing(8)

        # Polar recompute
        polar_btn = QtWidgets.QPushButton("Пересчитать Polar")
        polar_btn.clicked.connect(self.update_polar)
        ctrl_layout.addWidget(polar_btn)
        ctrl_layout.addSpacing(8)

        # YOLO threshold slider
        yolo_label = QtWidgets.QLabel("YOLO confidence threshold:")
        self.yolo_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.yolo_slider.setRange(0, 100)
        self.yolo_slider.setValue(35)
        self.yolo_val_label = QtWidgets.QLabel("0.35")
        self.yolo_slider.valueChanged.connect(self._on_yolo_slider)

        detect_cart_btn = QtWidgets.QPushButton("Детектировать на Cartesian")
        detect_cart_btn.clicked.connect(lambda: self.detect_objects("cartesian"))

        detect_polar_btn = QtWidgets.QPushButton("Детектировать на Polar")
        detect_polar_btn.clicked.connect(lambda: self.detect_objects("polar"))

        ctrl_layout.addWidget(yolo_label)
        ctrl_layout.addWidget(self.yolo_slider)
        ctrl_layout.addWidget(self.yolo_val_label)
        ctrl_layout.addWidget(detect_cart_btn)
        ctrl_layout.addWidget(detect_polar_btn)
        ctrl_layout.addSpacing(8)

        # Detection categories
        ctrl_layout.addWidget(QtWidgets.QLabel("Категории:"))
        self.checkbox_plane = QtWidgets.QCheckBox("Самолёты")
        self.checkbox_car = QtWidgets.QCheckBox("Автомобили")
        self.checkbox_house = QtWidgets.QCheckBox("Дома")
        self.checkbox_plane.setChecked(True)
        ctrl_layout.addWidget(self.checkbox_plane)
        ctrl_layout.addWidget(self.checkbox_car)
        ctrl_layout.addWidget(self.checkbox_house)

        # Clear overlay
        clear_overlay_btn = QtWidgets.QPushButton("Очистить наложения")
        clear_overlay_btn.clicked.connect(self.clear_overlays)
        ctrl_layout.addWidget(clear_overlay_btn)
        ctrl_layout.addStretch(1)

        # Help
        help_label = QtWidgets.QLabel("Параметры изображения отображаются внизу справа")
        help_label.setWordWrap(True)
        ctrl_layout.addWidget(help_label)

        control_widget = QtWidgets.QWidget()
        control_widget.setLayout(ctrl_layout)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(images_container)
        splitter.addWidget(control_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        # Params area bottom-right
        params_label = QtWidgets.QLabel()
        params_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        params_label.setMinimumHeight(160)
        params_label.setWordWrap(True)
        params_label.setStyleSheet("background: #f6f6f6; border: 1px solid #ddd; padding: 6px;")
        self.params_label = params_label

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(splitter)
        params_hbox = QtWidgets.QHBoxLayout()
        params_hbox.addStretch(1)
        params_hbox.addWidget(self.params_label, 1)
        main_layout.addLayout(params_hbox)

        central = QtWidgets.QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def _on_yolo_slider(self, v):
        val = v / 100.0
        self.yolo_val_label.setText(f"{val:.2f}")
        if self.model is not None:
            try:
                self.model.conf = val
            except Exception:
                pass

    def load_image(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выбрать изображение", "", "Images (*.png *.jpg *.bmp)")
        if not fname:
            return
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение.")
            return
        max_side = 800
        h, w = img.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale != 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        # если изображение имеет альфу, сохраняем, иначе приводим к BGR
        if img.ndim == 3 and img.shape[2] == 4:
            self.base_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            # keep alpha separately for overlay if needed
        else:
            self.base_img = img.copy()
        self.processed_img = self.base_img.copy()
        self.overlay_img = None
        self.clean_generated_img = None
        self.generated_flag = False
        self.generated_has_noise = False
        self.update_cart_and_polar()
        self.update_params()

    def save_image(self):
        if self.processed_img is None:
            QtWidgets.QMessageBox.information(self, "Сохранение", "Нет изображения для сохранения.")
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить изображение", "", "PNG (*.png);;JPEG (*.jpg)")
        if not fname:
            return
        if not (fname.lower().endswith(".png") or fname.lower().endswith(".jpg") or fname.lower().endswith(".jpeg")):
            fname += ".png"
        cv2.imwrite(fname, self.processed_img)
        QtWidgets.QMessageBox.information(self, "Сохранение", f"Сохранено: {fname}")

    def generate_simple_image(self):
        """Генерация простого изображения с объектами в полярной СК и его конвертация в декартову с прозрачным фоном для Polar"""
        polar_h = 512
        polar_w = 512
        polar_canvas = np.ones((polar_h, polar_w, 3), dtype=np.uint8) * 255

        if self.checkbox_plane.isChecked():
            plane_icon = draw_pixel_plane(60, 60)
            for _ in range(random.randint(1, 4)):
                ang = random.randint(0, polar_w - plane_icon.shape[1])
                rpos = random.randint(20, polar_h - plane_icon.shape[0] - 1)
                overlay_bgr = polar_canvas.copy()
                overlay_bgr = overlay_rgba(overlay_bgr, plane_icon, ang, rpos)
                polar_canvas = overlay_bgr

        if self.checkbox_house.isChecked():
            for _ in range(random.randint(1, 4)):
                ang = random.randint(0, polar_w - 40)
                rpos = random.randint(30, polar_h - 40)
                cv2.rectangle(polar_canvas, (ang, rpos), (ang + 30, rpos + 30), (0, 0, 0), -1)
        if self.checkbox_car.isChecked():
            for _ in range(random.randint(1, 4)):
                ang = random.randint(0, polar_w - 40)
                rpos = random.randint(20, polar_h - 30)
                cv2.rectangle(polar_canvas, (ang, rpos + 6), (ang + 40, rpos + 18), (0, 0, 0), -1)
                cv2.circle(polar_canvas, (ang + 10, rpos + 22), 5, (0, 0, 0), -1)
                cv2.circle(polar_canvas, (ang + 30, rpos + 22), 5, (0, 0, 0), -1)

        # делаем круглый polar с прозрачным фоном (BGRA)
        polar_rgba = make_circular_alpha(polar_canvas, diameter=min(polar_h, polar_w))
        self.polar_img = polar_rgba.copy()

        # конвертируем в декартову (RGB/BGR) для базового отображения
        out_size = (512, 512)
        center = (out_size[1] // 2, out_size[0] // 2)
        maxRadius = min(center[0], center[1])
        cart = polar_to_cartesian(polar_canvas, out_size=out_size, center=center, maxRadius=maxRadius)

        self.base_img = cart.copy()
        self.processed_img = cart.copy()
        self.overlay_img = None
        self.clean_generated_img = cart.copy()
        self.generated_flag = True
        self.generated_has_noise = False
        self.update_cart_and_polar()
        self.update_params()

    def generate_radar_image(self):
        """Генерация реалистичного РЛИ методом обратного проецирования и представление Polar как прозрачного круга сверху"""
        try:
            # Генерация РЛИ
            Image_normalized, massiv_x, massiv_y = self.radar_generator.generate_radar_image(N_RLD=1)

            # Преобразование в формат для отображения (полярный вид)
            radar_img = (Image_normalized[:, :, 0] * 255).astype(np.uint8)
            radar_img_color = cv2.applyColorMap(radar_img, cv2.COLORMAP_JET)

            # Создадим circular polar (BGRA) — используем тот же радарный вид, центрируем в квадрат
            size = max(radar_img_color.shape[:2])
            square = np.ones((size, size, 3), dtype=np.uint8) * 0
            h, w = radar_img_color.shape[:2]
            y0 = (size - h) // 2
            x0 = (size - w) // 2
            square[y0:y0 + h, x0:x0 + w] = radar_img_color
            polar_rgba = make_circular_alpha(square, diameter=size)

            # Масштабирование до разумного размера
            scale = min(1.0, 512 / max(size, size))
            if scale != 1.0:
                new_size = (int(size * scale), int(size * scale))
                polar_rgba = cv2.resize(polar_rgba, new_size, interpolation=cv2.INTER_AREA)

            self.polar_img = polar_rgba.copy()

            # Декартов вид (цветной)
            radar_img_color_resized = cv2.resize(radar_img_color, (512, 512), interpolation=cv2.INTER_AREA)
            self.base_img = radar_img_color_resized.copy()
            self.processed_img = radar_img_color_resized.copy()
            self.overlay_img = None
            self.clean_generated_img = radar_img_color_resized.copy()
            self.generated_flag = True
            self.generated_has_noise = False

            self.update_cart_and_polar()
            self.update_params()

            QtWidgets.QMessageBox.information(self, "Генерация РЛИ",
                                              "Радарное изображение успешно сгенерировано методом обратного проецирования")

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Ошибка при генерации РЛИ: {str(e)}")

    def apply_noise(self):
        if self.processed_img is None:
            QtWidgets.QMessageBox.information(self, "Шум", "Нет изображения.")
            return
        typ = self.noise_combo.currentText()
        val = self.noise_slider.value()
        img = self.processed_img.copy()
        if typ == "Нет":
            QtWidgets.QMessageBox.information(self, "Шум", "Выбран 'Нет'.")
            return
        if typ == "Гауссов":
            sigma = val * 0.6
            img = add_gaussian_noise(img, sigma)
        elif typ == "Соль-и-перец":
            amount = val / 100.0 * 0.18
            img = add_salt_pepper(img, amount)
        if self.generated_flag and (self.clean_generated_img is not None):
            self.generated_has_noise = True
        self.processed_img = img
        self.overlay_img = None
        self.update_cart_and_polar()
        self.update_params()

    def remove_generated_noise(self):
        if not self.generated_flag or self.clean_generated_img is None:
            QtWidgets.QMessageBox.information(self, "Убрать шум",
                                              "Нет сгенерированного изображения для восстановления.")
            return
        self.processed_img = self.clean_generated_img.copy()
        self.overlay_img = None
        self.generated_has_noise = False
        self.update_cart_and_polar()
        self.update_params()

    def update_polar(self):
        # если polar_img есть — отображаем его (поддерживается BGRA)
        if self.polar_img is None:
            return
        self.polar_widget.set_image(self.polar_img)

    def update_cart_and_polar(self):
        if self.processed_img is None:
            self.cart_widget.set_image(None)
            self.polar_widget.set_image(None)
            return
        composed = self.processed_img.copy()
        if self.overlay_img is not None:
            if self.overlay_img.shape[:2] != composed.shape[:2]:
                ol = cv2.resize(self.overlay_img, (composed.shape[1], composed.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                ol = self.overlay_img
            composed = cv2.addWeighted(composed, 0.75, ol, 0.25, 0)
        self.cart_widget.set_image(composed)
        # polar отображается отдельно сверху, уже как BGRA прозрачное
        self.update_polar()

    def clear_overlays(self):
        self.overlay_img = None
        self.update_cart_and_polar()

    def update_params(self):
        if self.processed_img is None:
            self.params_label.setText("Нет изображения")
            return
        h, w = self.processed_img.shape[:2]
        gray = cv2.cvtColor(self.processed_img, cv2.COLOR_BGR2GRAY)
        mn = int(gray.min())
        mx = int(gray.max())
        mean = float(gray.mean())
        std = float(gray.std())
        ent = calc_entropy(gray)
        snr = mean / (std + 1e-6)
        center = (w // 2, h // 2)
        gen = "Да" if self.generated_flag else "Нет"
        gen_noise = "Да" if self.generated_has_noise else "Нет"
        params = (
            f"Размер: {w} × {h}\n"
            f"Яркость (min..max): {mn} .. {mx}\n"
            f"Mean: {mean:.2f}, Std: {std:.2f}\n"
            f"Энтропия: {ent:.2f}\n"
            f"SNR: {snr:.2f}\n"
        )
        self.params_label.setText(params)

    def detect_objects(self, image_type="cartesian"):
        if image_type == "cartesian" and self.processed_img is None:
            QtWidgets.QMessageBox.information(self, "Детектирование", "Нет Cartesian изображения.")
            return
        if image_type == "polar" and self.polar_img is None:
            QtWidgets.QMessageBox.information(self, "Детектирование", "Нет Polar изображения.")
            return

        target_classes = []
        if self.checkbox_plane.isChecked():
            target_classes.append("airplane")
        if self.checkbox_car.isChecked():
            target_classes.append("car")
        if self.checkbox_house.isChecked():
            target_classes.append("house")

        if self.model is None:
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                self.model.conf = self.yolo_slider.value() / 100.0
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "YOLO", f"Не удалось загрузить модель YOLO: {e}")
                return
        else:
            self.model.conf = self.yolo_slider.value() / 100.0

        try:
            if image_type == "cartesian":
                img_to_detect = self.processed_img
                display_img = self.processed_img.copy()
            else:
                # Для polar: используем RGB-версию без альфа чтобы детектор понял
                if self.polar_img.shape[2] == 4:
                    img_to_detect = cv2.cvtColor(self.polar_img, cv2.COLOR_BGRA2BGR)
                else:
                    img_to_detect = self.polar_img
                display_img = img_to_detect.copy()

            results = self.model(img_to_detect)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "YOLO", f"Ошибка при запуске модели: {e}")
            return

        df = results.pandas().xyxy[0]
        output = display_img.copy()
        found = False
        for _, row in df.iterrows():
            cls_name = row['name']
            if (len(target_classes) == 0) or (cls_name in target_classes):
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                conf = float(row['confidence'])
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output, f"{cls_name} {conf:.2f}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                found = True

        if not found:
            QtWidgets.QMessageBox.information(self, "YOLO",
                                              "Объекты не найдены (попробуйте понизить порог confidence).")

        if image_type == "cartesian":
            self.overlay_img = output
            self.update_cart_and_polar()
        else:
            # Рисуем результат вместо polar (убираем альфу во время отображения результатов)
            out_bgr = output
            out_bgra = make_circular_alpha(out_bgr, diameter=min(out_bgr.shape[:2]))
            self.polar_widget.set_image(out_bgra)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # Убираем фон окна у Qt (позволяет видеть прозрачность виджетов на некоторых системах)
    win = RadarApp()
    win.show()
    sys.exit(app.exec_())
