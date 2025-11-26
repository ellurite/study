import numpy as np
from typing import Optional, Tuple
from domain import RadarParameters

class RadarImageGenerator:
    """Класс для генерации радиолокационных изображений по методу обратного проецирования"""

    def __init__(self, parameters: Optional[RadarParameters] = None):
        self.params = parameters or RadarParameters()

    def generate_radar_image(self, n_rld: int = 4, point_coord: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Генерация РЛИ методом обратного проецирования"""
        if point_coord is None:
            beta = -5 * np.pi / 180
            r_scene = 300
            point_coord = np.array([[r_scene * np.tan(beta), r_scene]])

        t = np.arange(0, self.params.tp, 1 / self.params.fs)
        f = np.linspace(self.params.fc - self.params.bw / 2, self.params.fc + self.params.bw / 2, len(t))
        lambda_vals = self.params.cc / f

        n = 8
        l = 9
        delta_x = l / (n - 1)
        k = np.linspace(-l / 2, l / 2, n)

        massiv_range_x = 100
        massiv_range_y = 100
        massiv_step = 0.5
        r_scene = 300

        center_x = 0
        center_y = r_scene

        massiv_x = np.arange(center_x - massiv_range_x, center_x + massiv_range_x + massiv_step, massiv_step)
        massiv_y = np.arange(center_y - massiv_range_y, center_y + massiv_range_y + massiv_step, massiv_step)
        massiv_x, massiv_y = np.meshgrid(massiv_x, massiv_y)

        dist_ant_start = np.zeros((n, 2, 4))
        ugol = [0, 0, 0, 0]

        for j in range(n):
            dist_ant_start[j, :, 0] = [k[j] * np.cos(ugol[0]), 0]
            dist_ant_start[j, :, 1] = [k[j] * np.cos(ugol[1]), 35]
            dist_ant_start[j, :, 2] = [k[j] * np.cos(ugol[2]) - 10, 15]
            dist_ant_start[j, :, 3] = [k[j] * np.cos(ugol[3]) + 23, 20]

        image = np.zeros((massiv_x.shape[0], massiv_x.shape[1], n_rld), dtype=complex)

        for r in range(n_rld):
            dist_ant = dist_ant_start[:, :, r]
            vsum = np.zeros((n, len(t)), dtype=complex)

            for i in range(n):
                for n_point in range(len(point_coord)):
                    rc0 = np.sqrt((point_coord[n_point, 0] - dist_ant[i, 0]) ** 2 +
                                  (point_coord[n_point, 1] - dist_ant[i, 1]) ** 2)
                    vc = np.exp(-1j * 2 * np.pi * rc0 / lambda_vals)
                    vsum[i, :] += vc

            s0 = np.zeros((massiv_x.shape[0], massiv_x.shape[1]), dtype=complex)
            for z in range(n):
                r_c = np.sqrt((massiv_x - dist_ant[z, 0]) ** 2 +
                              (massiv_y - dist_ant[z, 1]) ** 2)
                for k1 in range(len(t)):
                    s = vsum[z, k1] * np.exp(1j * 2 * np.pi * r_c / lambda_vals[k1])
                    s0 += s

            image[:, :, r] = s0

        image_normalized = np.zeros_like(image, dtype=np.float32)
        for r in range(n_rld):
            img_abs = np.abs(image[:, :, r])
            image_normalized[:, :, r] = img_abs / np.max(img_abs)

        return image_normalized, massiv_x, massiv_y