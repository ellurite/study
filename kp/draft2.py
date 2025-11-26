
import time
import numpy as np
import cv2
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata


# ------------------- Target Detection -------------------
class TargetDetector:
    def __init__(self, brightness_thresh=0.7, min_area=5, max_aspect_ratio=2.0):
        self.brightness_thresh = brightness_thresh
        self.min_area = min_area
        self.max_aspect_ratio = max_aspect_ratio

    def detect(self, image):
        img_uint8 = (image * 255).astype(np.uint8)
        _, thresh = cv2.threshold(img_uint8, int(self.brightness_thresh * 255), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        targets = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect_ratio = max(w / h, h / w)
            if area >= self.min_area and aspect_ratio <= self.max_aspect_ratio:
                targets.append((x, y, w, h))
        return targets

# ------------------- Radar Generation -------------------
def generate_radar_images(N_RLD=1, update_status=None):
    t_total_start = time.time()
    if update_status: update_status("Creating grid...")
    t_grid_start = time.time()

    Fc = 300e6; cc = 3e8; BW = 7e6; Fs = BW / 10000; Tp = 50 / Fs
    t = np.arange(0, Tp, 1 / Fs)
    f = np.linspace(Fc - BW / 2, Fc + BW / 2, len(t))
    lambda_arr = cc / f

    N = 8; L = 9; k = np.linspace(-L / 2, L / 2, N); Na = len(k); Nf = len(t)

    massiv_range_x = 100; massiv_range_y = 100; massiv_step = 0.5
    R_scene = 300; center_x, center_y = 0, R_scene
    x_coords = np.arange(center_x - massiv_range_x, center_x + massiv_range_x + 1e-9, massiv_step)
    y_coords = np.arange(center_y - massiv_range_y, center_y + massiv_range_y + 1e-9, massiv_step)
    X, Y = np.meshgrid(x_coords, y_coords)
    x_size, y_size = X.shape[1], X.shape[0]

    t_grid_end = time.time()
    if update_status: update_status(f"Grid created in {t_grid_end - t_grid_start:.2f}s")

    # Polar grid
    azimut = np.degrees(np.arctan2(Y, X))
    rng = np.sqrt(X ** 2 + Y ** 2)
    massiv_azimut = np.linspace(azimut.min(), azimut.max(), x_size)
    massiv_range = np.linspace(rng.min(), rng.max(), y_size)
    AZ_grid, R_grid = np.meshgrid(massiv_azimut, massiv_range)
    X_polar = R_grid * np.cos(np.radians(AZ_grid))
    Y_polar = R_grid * np.sin(np.radians(AZ_grid))

    # Bright point
    beta = -5 * np.pi / 180
    point_x = R_scene * np.tan(beta)
    point_y = R_scene

    # Radar positions
    ugol = [1, 20, 60, 100]; ugol_rad = np.radians(ugol)
    dist_ant_start = np.zeros((Na, 2, N_RLD))
    for j in range(Na):
        if N_RLD >= 1: dist_ant_start[j, :, 0] = [k[j] * np.cos(ugol_rad[0]), 0]
        if N_RLD >= 2: dist_ant_start[j, :, 1] = [k[j] * np.cos(ugol_rad[1]), 35]
        if N_RLD >= 3: dist_ant_start[j, :, 2] = [k[j] * np.cos(ugol_rad[2]) - 10, 15]
        if N_RLD >= 4: dist_ant_start[j, :, 3] = [k[j] * np.cos(ugol_rad[3]) + 23, 20]

    # Backprojection
    if update_status: update_status("Starting backprojection...")
    Image = np.zeros((y_size, x_size, N_RLD), dtype=np.complex128)
    for r in range(N_RLD):
        Vsum = np.zeros((Na, Nf), dtype=np.complex128)
        for i in range(Na):
            dx = point_x - dist_ant_start[i, 0, r]
            dy = point_y - dist_ant_start[i, 1, r]
            rc0 = np.sqrt(dx ** 2 + dy ** 2)
            Vsum[i, :] = np.exp(-1j * 2 * np.pi * rc0 / lambda_arr)
        S0 = np.zeros((y_size, x_size), dtype=np.complex128)
        for z in range(Na):
            dx_grid = X - dist_ant_start[z, 0, r]
            dy_grid = Y - dist_ant_start[z, 1, r]
            r_c = np.sqrt(dx_grid ** 2 + dy_grid ** 2)
            S_temp = np.exp(1j * 2 * np.pi * r_c[..., None] / lambda_arr[None, None, :])
            S0 += np.sum(Vsum[z, :] * S_temp, axis=2)
        Image[:, :, r] = S0

    # Normalize
    Image_dekart = np.zeros_like(Image.real)
    for r in range(N_RLD):
        mag = np.abs(Image[:, :, r])
        if mag.max() > 0: Image_dekart[:, :, r] = mag / mag.max()

    # Polar
    Image_polar = np.zeros_like(Image_dekart)
    for r in range(N_RLD):
        if griddata is not None:
            pts = np.column_stack((X.ravel(), Y.ravel()))
            vals = Image_dekart[:, :, r].ravel()
            grid_z = griddata(pts, vals, (X_polar, Y_polar), method='linear')
            if np.any(np.isnan(grid_z)):
                grid_z_near = griddata(pts, vals, (X_polar, Y_polar), method='nearest')
                grid_z[np.isnan(grid_z)] = grid_z_near[np.isnan(grid_z)]
            Image_polar[:, :, r] = grid_z
        else:
            xi = ((X_polar - x_coords[0]) / massiv_step).round().astype(int)
            yi = ((Y_polar - y_coords[0]) / massiv_step).round().astype(int)
            xi = np.clip(xi, 0, x_size - 1)
            yi = np.clip(yi, 0, y_size - 1)
            Image_polar[:, :, r] = Image_dekart[yi, xi, r]

    if update_status: update_status(f"Radar generation done in {time.time()-t_total_start:.2f}s")
    return Image_dekart, Image_polar, {'x_coords': x_coords, 'y_coords': y_coords,
                                       'massiv_azimut': massiv_azimut, 'massiv_range': massiv_range}

# ------------------- GUI -------------------
class RadarWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radar Backprojection Viewer with Detection")
        self.resize(1800, 900)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Left: Images
        self.image_scroll = QtWidgets.QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_container = QtWidgets.QWidget()
        self.image_scroll.setWidget(self.image_container)
        self.image_layout = QtWidgets.QVBoxLayout(self.image_container)
        layout.addWidget(self.image_scroll, 3)

        # Right: Controls
        self.status_panel = QtWidgets.QWidget()
        self.status_layout = QtWidgets.QVBoxLayout(self.status_panel)
        layout.addWidget(self.status_panel, 1)

        self.spin_radars = QtWidgets.QSpinBox()
        self.spin_radars.setMinimum(1)
        self.spin_radars.setMaximum(4)
        self.spin_radars.setValue(1)
        self.status_layout.addWidget(QtWidgets.QLabel("Number of radars:"))
        self.status_layout.addWidget(self.spin_radars)

        self.btn_generate = QtWidgets.QPushButton("Generate")
        self.btn_generate.clicked.connect(self.on_generate)
        self.status_layout.addWidget(self.btn_generate)

        # Sliders
        self.status_layout.addWidget(QtWidgets.QLabel("Brightness threshold:"))
        self.slider_thresh = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_thresh.setRange(0, 100)
        self.slider_thresh.setValue(70)
        self.status_layout.addWidget(self.slider_thresh)

        self.status_layout.addWidget(QtWidgets.QLabel("Min target area:"))
        self.slider_area = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_area.setRange(1, 500)
        self.slider_area.setValue(5)
        self.status_layout.addWidget(self.slider_area)

        # Log
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.status_layout.addWidget(self.log_text)
        self.status_layout.addStretch()

        # Variables
        self.figures = []
        self.canvases = []
        self.current_image = None

        # Signals
        self.slider_thresh.valueChanged.connect(self.update_detection)
        self.slider_area.valueChanged.connect(self.update_detection)

    def log(self, msg):
        self.log_text.append(msg)
        QtWidgets.QApplication.processEvents()

    def clear_plots(self):
        for canvas in self.canvases:
            canvas.setParent(None)
            canvas.deleteLater()
        self.figures.clear()
        self.canvases.clear()

    def on_generate(self):
        N_RLD = int(self.spin_radars.value())
        self.log_text.clear()
        self.log(f"Starting generation for {N_RLD} radar(s)...")

        Image_dekart, Image_polar, meta = generate_radar_images(N_RLD, update_status=self.log)
        self.display_all_images(Image_dekart, Image_polar, meta, N_RLD)
        # Сохраняем первый Cartesian image для детекции
        self.current_image = Image_dekart[:, :, 0]
        self.update_detection()

    def display_all_images(self, Image_dekart, Image_polar, meta, N_RLD):
        self.clear_plots()
        x_coords, y_coords = meta['x_coords'], meta['y_coords']
        az, rg = meta['massiv_azimut'], meta['massiv_range']

        grid_widget = QtWidgets.QWidget()
        grid_layout = QtWidgets.QGridLayout(grid_widget)
        grid_layout.setSpacing(10)

        for r in range(N_RLD):
            # Cartesian
            fig_cart = Figure(figsize=(4, 3))
            canvas_cart = FigureCanvas(fig_cart)
            ax1 = fig_cart.add_subplot(111)
            im1 = ax1.imshow(Image_dekart[:, :, r],
                             extent=(x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]),
                             origin='lower', aspect='equal',cmap='jet')
            ax1.set_title(f'Радар {r + 1} - ДПСК')
            ax1.set_xlabel('x (m)');
            ax1.set_ylabel('y (m)')
            fig_cart.colorbar(im1, ax=ax1)
            fig_cart.tight_layout()

            # Polar
            fig_pol = Figure(figsize=(4, 3))
            canvas_pol = FigureCanvas(fig_pol)
            ax2 = fig_pol.add_subplot(111)
            im2 = ax2.imshow(Image_polar[:, :, r],
                             extent=(az[0], az[-1], rg[0], rg[-1]),
                             origin='lower', aspect='auto',cmap='jet')
            ax2.set_title(f'Радар {r + 1} - Полярная СК')
            ax2.set_xlabel('Угол (deg)');
            ax2.set_ylabel('Расстояние (m)')
            fig_pol.colorbar(im2, ax=ax2)
            fig_pol.tight_layout()

            # Поменяли местами
            grid_layout.addWidget(canvas_pol, 2 * r, 0)  # Полярная слева
            grid_layout.addWidget(canvas_cart, 2 * r, 1)  # ДПСК справа

            self.figures.extend([fig_cart, fig_pol])
            self.canvases.extend([canvas_cart, canvas_pol])

        self.image_layout.addWidget(grid_widget)
        self.image_container.updateGeometry()

    def update_detection(self):
        if self.current_image is None or not self.canvases:
            return
        thresh = self.slider_thresh.value() / 100.0
        min_area = self.slider_area.value()
        detector = TargetDetector(brightness_thresh=thresh, min_area=min_area)

        ax = self.canvases[0].figure.gca()  # Cartesian image для детекции

        # удаляем старые квадраты
        for patch in getattr(ax, 'patches', []):
            patch.remove()

        targets = detector.detect(self.current_image)

        # преобразуем пиксели в координаты осей
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ny, nx = self.current_image.shape

        for x, y, w, h in targets:
            x_coord = x0 + (x1 - x0) * x / nx
            y_coord = y0 + (y1 - y0) * y / ny
            size_coord = max(w, h) * (x1 - x0) / nx  # квадрат в координатах осей
            rect = Rectangle((x_coord, y_coord), size_coord, size_coord,
                             linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        self.log_text.append(f"Обнаружено целей: {len(targets)}")
        self.canvases[0].draw()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = RadarWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
