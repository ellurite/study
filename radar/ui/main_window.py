import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

# Теперь можно импортировать из корневого пакета
from radar.services import RadarService
from radar.ui.widgets import ImageWidget

class RadarApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.service = RadarService()
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Радарная обработка изображений")
        self.resize(1400, 800)

        # Создание виджетов изображений
        polar_label = QtWidgets.QLabel("Polar (Полярная СК)")
        polar_label.setAlignment(QtCore.Qt.AlignCenter)
        self.polar_widget = ImageWidget()

        cart_label = QtWidgets.QLabel("Cartesian (Декартова СК)")
        cart_label.setAlignment(QtCore.Qt.AlignCenter)
        self.cart_widget = ImageWidget()

        # Компоновка изображений
        images_layout = QtWidgets.QVBoxLayout()
        images_layout.addWidget(polar_label)
        images_layout.addWidget(self.polar_widget, 1)
        images_layout.addWidget(cart_label)
        images_layout.addWidget(self.cart_widget, 2)

        images_container = QtWidgets.QWidget()
        images_container.setLayout(images_layout)

        # Панель управления
        control_widget = self.create_control_panel()

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(images_container)
        splitter.addWidget(control_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        # Панель параметров
        self.params_label = QtWidgets.QLabel()
        self.params_label.setMinimumHeight(160)
        self.params_label.setStyleSheet("background: #f6f6f6; border: 1px solid #ddd; padding: 6px;")

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(splitter)

        params_hbox = QtWidgets.QHBoxLayout()
        params_hbox.addStretch(1)
        params_hbox.addWidget(self.params_label, 1)
        main_layout.addLayout(params_hbox)

        central = QtWidgets.QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def create_control_panel(self):
        ctrl_layout = QtWidgets.QVBoxLayout()

        # File controls
        load_btn = QtWidgets.QPushButton("Загрузить изображение")
        load_btn.clicked.connect(self.load_image)
        save_btn = QtWidgets.QPushButton("Сохранить изображение")
        save_btn.clicked.connect(self.save_image)

        # Generation controls
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

        # Detection controls
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

        # Categories
        ctrl_layout.addWidget(QtWidgets.QLabel("Категории:"))
        self.checkbox_plane = QtWidgets.QCheckBox("Самолёты")
        self.checkbox_car = QtWidgets.QCheckBox("Автомобили")
        self.checkbox_house = QtWidgets.QCheckBox("Дома")
        self.checkbox_plane.setChecked(True)
        ctrl_layout.addWidget(self.checkbox_plane)
        ctrl_layout.addWidget(self.checkbox_car)
        ctrl_layout.addWidget(self.checkbox_house)

        # Other controls
        polar_btn = QtWidgets.QPushButton("Пересчитать Polar")
        polar_btn.clicked.connect(self.update_polar)
        clear_overlay_btn = QtWidgets.QPushButton("Очистить наложения")
        clear_overlay_btn.clicked.connect(self.clear_overlays)

        ctrl_layout.addWidget(polar_btn)
        ctrl_layout.addWidget(clear_overlay_btn)
        ctrl_layout.addStretch(1)

        control_widget = QtWidgets.QWidget()
        control_widget.setLayout(ctrl_layout)
        return control_widget

    def _on_yolo_slider(self, v):
        val = v / 100.0
        self.yolo_val_label.setText(f"{val:.2f}")

    def load_image(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выбрать изображение", "", "Images (*.png *.jpg *.bmp)")
        if not fname:
            return
        if self.service.load_image(fname):
            self.update_display()
        else:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение.")

    def save_image(self):
        if self.service.image_data.processed_image is None:
            QtWidgets.QMessageBox.information(self, "Сохранение", "Нет изображения для сохранения.")
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить изображение", "", "PNG (*.png);;JPEG (*.jpg)")
        if not fname:
            return
        if not (fname.lower().endswith(".png") or fname.lower().endswith(".jpg") or fname.lower().endswith(".jpeg")):
            fname += ".png"
        cv2.imwrite(fname, self.service.image_data.processed_image)
        QtWidgets.QMessageBox.information(self, "Сохранение", f"Сохранено: {fname}")

    def generate_simple_image(self):
        show_planes = self.checkbox_plane.isChecked()
        show_houses = self.checkbox_house.isChecked()
        show_cars = self.checkbox_car.isChecked()
        self.service.generate_simple_image(show_planes, show_houses, show_cars)
        self.update_display()

    def generate_radar_image(self):
        if self.service.generate_radar_image():
            self.update_display()
            QtWidgets.QMessageBox.information(self, "Генерация РЛИ", "Радарное изображение успешно сгенерировано")
        else:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Ошибка при генерации РЛИ")

    def apply_noise(self):
        noise_type = self.noise_combo.currentText()
        intensity = self.noise_slider.value()
        if noise_type == "Нет":
            return
        if self.service.apply_noise(noise_type, intensity):
            self.update_display()

    def remove_generated_noise(self):
        if self.service.remove_noise():
            self.update_display()

    def update_polar(self):
        self.polar_widget.set_image(self.service.image_data.polar_image)

    def clear_overlays(self):
        self.service.image_data.overlay_image = None
        self.update_display()

    def update_display(self):
        """Обновление отображения изображений и статистики"""
        stats = self.service.get_image_stats()
        if stats:
            self.params_label.setText(
                f"Размер: {stats.get('width', 0)} × {stats.get('height', 0)}\n"
                f"Яркость: {stats.get('min_brightness', 0)} .. {stats.get('max_brightness', 0)}\n"
                f"Mean: {stats.get('mean_brightness', 0):.2f}, Std: {stats.get('std_brightness', 0):.2f}\n"
                f"Энтропия: {stats.get('entropy', 0):.2f}\n"
                f"SNR: {stats.get('snr', 0):.2f}\n"
                f"Сгенерировано: {'Да' if stats.get('is_generated', False) else 'Нет'}\n"
                f"Шум: {'Да' if stats.get('has_noise', False) else 'Нет'}"
            )
        else:
            self.params_label.setText("Нет изображения")

        # Отображение Cartesian
        composed = self.service.image_data.processed_image
        if composed is not None and self.service.image_data.overlay_image is not None:
            if self.service.image_data.overlay_image.shape[:2] != composed.shape[:2]:
                ol = cv2.resize(self.service.image_data.overlay_image,
                                (composed.shape[1], composed.shape[0]),
                                interpolation=cv2.INTER_AREA)
            else:
                ol = self.service.image_data.overlay_image
            composed = cv2.addWeighted(composed, 0.75, ol, 0.25, 0)

        self.cart_widget.set_image(composed)

        # Отображение Polar
        self.polar_widget.set_image(self.service.image_data.polar_image)

    def detect_objects(self, image_type: str):
        if image_type == "cartesian" and self.service.image_data.processed_image is None:
            QtWidgets.QMessageBox.information(self, "Детектирование", "Нет Cartesian изображения.")
            return
        if image_type == "polar" and self.service.image_data.polar_image is None:
            QtWidgets.QMessageBox.information(self, "Детектирование", "Нет Polar изображения.")
            return

        target_classes = []
        if self.checkbox_plane.isChecked():
            target_classes.append("airplane")
        if self.checkbox_car.isChecked():
            target_classes.append("car")
        if self.checkbox_house.isChecked():
            target_classes.append("house")

        confidence_threshold = self.yolo_slider.value() / 100.0

        try:
            if image_type == "cartesian":
                detections = self.service.detect_objects(confidence_threshold, target_classes)
                if detections:
                    output = self.service.image_data.processed_image.copy()
                    for detection in detections:
                        x1, y1, x2, y2 = detection.bbox
                        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(output, f"{detection.class_name} {detection.confidence:.2f}",
                                    (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1, cv2.LINE_AA)
                    self.service.image_data.overlay_image = output
                else:
                    QtWidgets.QMessageBox.information(self, "YOLO", "Объекты не найдены")

            else:  # polar
                # Для polar используем RGB-версию без альфа
                if self.service.image_data.polar_image.shape[2] == 4:
                    polar_bgr = cv2.cvtColor(self.service.image_data.polar_image, cv2.COLOR_BGRA2BGR)
                else:
                    polar_bgr = self.service.image_data.polar_image

                detections = self.service.detect_objects(confidence_threshold, target_classes)
                if detections:
                    output = polar_bgr.copy()
                    for detection in detections:
                        x1, y1, x2, y2 = detection.bbox
                        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(output, f"{detection.class_name} {detection.confidence:.2f}",
                                    (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1, cv2.LINE_AA)
                    # Конвертируем обратно в BGRA с круговой маской
                    polar_with_detections = self.service.image_processor.make_circular_alpha(output)
                    self.service.image_data.polar_image = polar_with_detections
                else:
                    QtWidgets.QMessageBox.information(self, "YOLO", "Объекты не найдены")

            self.update_display()

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "YOLO", f"Ошибка при детектировании: {e}")