import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import time

# Метод обратного проецирования

# Параметры сигнала:
Fc = 300e6  # центральная несущая частота ЛЧМ-импульса
cc = 3e8  # скорость света
BW = 7e6  # девиация частоты ЛЧМ
Fs = BW * 1 / 10000  # частота дискретизации
Tp = 50 / Fs  # длительность сигнала
t = np.arange(0, Tp, 1 / Fs)  # временная сетка
f = np.linspace(Fc - BW / 2, Fc + BW / 2, len(t))  # частотная сетка ЛЧМ
lambda_ = cc / f  # длина волны
N = 8  # число излучателей в апертуре
L = 9  # длина апертуры (метров)
delta_x = L / (N - 1)  # шаг между излучателями в апертуре
k = np.arange(-L / 2, L / 2 + delta_x, delta_x)  # массив синтезируемой апертуры
Na = len(k)  # размерность массива по азимуту (число излучателей в апертуре)
Nf = len(t)  # размерность массива по дальности

# Пространственная сетка:
massiv_range_x = 100  # расстояние от центра сцены до одного края сцены
massiv_range_y = 100
massiv_step = 0.5  # пространственный шаг сцены
R_scene = 300  # дальность до центра сцены

# Координаты центра сцены:
center_x = 0
center_y = R_scene

# Массив координат:
massiv_x = np.arange(center_x - massiv_range_x, center_x + massiv_range_x + massiv_step, massiv_step)
massiv_y = np.arange(center_y - massiv_range_y, center_y + massiv_range_y + massiv_step, massiv_step)
massiv_x, massiv_y = np.meshgrid(massiv_x, massiv_y)

# Пересчет в полярную систему координат:
azimut = np.arctan2(massiv_y, massiv_x) * 180 / np.pi
range_ = np.sqrt(massiv_x ** 2 + massiv_y ** 2)

# Установление эквидистантной сетки в полярной системе координат:
min_azimut = np.min(azimut)  # минимальное значение азимута в кадре
max_azimut = np.max(azimut)  # максимальное значение азимута в кадре
step_azimut = (max_azimut - min_azimut) / (massiv_x.shape[0] - 1)  # шаг сетки по азимуту
massiv_azimut = np.linspace(min_azimut, max_azimut, massiv_x.shape[1])  # массив азимута (одна строка)

min_range = np.min(range_)  # минимальное значение дальности в кадре
max_range = np.max(range_)  # максимальное значение дальности в кадре
step_range = (max_range - min_range) / (massiv_y.shape[0] - 1)  # шаг сетки по дальности
massiv_range = np.linspace(min_range, max_range, massiv_y.shape[0])  # массив дальности (один столбец)

massiv_azimut = np.tile(massiv_azimut, (massiv_x.shape[0], 1))  # массив азимута (целиком)
massiv_range = np.tile(massiv_range.reshape(-1, 1), (1, massiv_y.shape[1]))  # массив дальности (целиком)

# Пересчет эквидистантной сетки из полярной СК в декартову (вспомогат. этап):
massiv_x_polar = massiv_range * np.cos(massiv_azimut * np.pi / 180)  # координата x для всех точек эквидистантной сетки
massiv_y_polar = massiv_range * np.sin(massiv_azimut * np.pi / 180)  # координата y для всех точек эквидистантной сетки

# Моделируемые блестящие точки (БТ):
# координаты (1-я - азимут x, 2-я - дальность R):
beta = -5 * np.pi / 180  # азимутальный угол БТ относительно нулевого направления
point_coord = np.array([[R_scene * np.tan(beta), R_scene]])  # координаты БТ (x,y)
N_points = 1  # количество БТ

# Вид разноса (ВЫБРАТЬ):
v = 1  # 1 - вертикальный разнос; 2 - разнос по окружности
if v == 1:  # вертикальный разнос
    pass
elif v == 2:  # разнос по окружности
    pass

# Задаем координаты излучателей в каждой АР:
N_RLD = 4  # КОЛИЧЕСТВО РЛД (ИЗМЕНЯТЬ)
ds = 1  # 1 - задаем координаты излучателей отдельно, 0 - не задаем (разнос по умолчанию)
dist_ant_start = np.zeros((Na, 2, 4))
ugol_1 = 0 * np.pi / 180  # угол разворота 1-й АР (относительно вертикали)
ugol_2 = 0 * np.pi / 180  # угол разворота 2-й АР (относительно вертикали)
ugol_3 = 0 * np.pi / 180  # угол разворота 3-й АР (относительно вертикали)
ugol_4 = 0 * np.pi / 180  # угол разворота 4-й АР (относительно вертикали)
ugol = [ugol_1, ugol_2, ugol_3, ugol_4]
for j in range(Na):  # 1-я координата - азимут x, 2-я координата - дальность y
    dist_ant_start[j, 0:2, 0] = [k[j] * np.cos(ugol[0]), 0]
    dist_ant_start[j, 0:2, 1] = [k[j] * np.cos(ugol[1]), 35]
    dist_ant_start[j, 0:2, 2] = [k[j] * np.cos(ugol[2]) - 10, 15]
    dist_ant_start[j, 0:2, 3] = [k[j] * np.cos(ugol[3]) + 23, 20]

start_time = time.time()
x = massiv_x.shape[0]
y = massiv_x.shape[1]
Image = np.zeros((x, y, N_RLD), dtype=complex)
S0 = np.zeros((x, y, N_RLD), dtype=complex)
Vc = np.zeros((Na, Nf), dtype=complex)
ugol_point = np.zeros((1, point_coord.shape[0]))

for r in range(N_RLD):  # цикл по РЛД
    dist_ant = np.zeros((Na, 2))
    r_c = np.zeros((x, y, Na))
    S = np.zeros((x, y), dtype=complex)
    Vsum = np.zeros((len(k), len(t)), dtype=complex)

    for i in range(Na):
        if ds == 1:  # задаем координаты излучателей отдельно
            dist_ant[i, 0:2] = dist_ant_start[i, 0:2, r]  # берем координаты из заданных массивов

        # Метод обратных проекций:
        for n in range(N_points):
            # расстояние от точки апертуры до моделируемой точки:
            rc0 = np.sqrt((point_coord[n, 0] - dist_ant[i, 0]) ** 2 +
                          (point_coord[n, 1] - dist_ant[i, 1]) ** 2)
            Vc[i, :] = np.exp(-1j * 2 * np.pi * (rc0) / lambda_)
            Vsum[i, :] = Vsum[i, :] + Vc[i, :]

    for z in range(Na):
        r_c = np.sqrt((massiv_x - dist_ant[z, 0]) ** 2 + (massiv_y - dist_ant[z, 1]) ** 2)
        for k1 in range(Nf):
            S = Vsum[z, k1] * np.exp(1j * 2 * np.pi * r_c / lambda_[k1])
            S0[:, :, r] = S0[:, :, r] + S

    Image[:, :, r] = Image[:, :, r] + S0[:, :, r]

end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")

# Изображение с нормированной яркостью в декартовой СК:
Image_dekart = np.zeros_like(Image, dtype=complex)
Image_dekart[:, :, 0] = Image[:, :, 0] / np.max(np.abs(Image[:, :, 0]))
# После вычисления Image, перед визуализацией добавьте:
print("Максимумы изображений по РЛД:")
for r in range(N_RLD):
    max_val = np.max(np.abs(Image[:, :, r]))
    # Находим координаты максимума
    max_pos = np.unravel_index(np.argmax(np.abs(Image[:, :, r])), Image[:, :, r].shape)
    coord_x = massiv_x[max_pos]
    coord_y = massiv_y[max_pos]
    print(f"РЛД {r+1}: макс={max_val:.4f}, позиция=({coord_x:.1f}, {coord_y:.1f})")

# Также проверьте координаты антенн:
print("\nКоординаты первых антенн каждого РЛД:")
for r in range(N_RLD):
    print(f"РЛД {r+1}: {dist_ant_start[0, :, r]}")
plt.figure(1)
plt.imshow(np.abs(Image_dekart[:, :, 0]), extent=[massiv_x[0, 0], massiv_x[0, -1], massiv_y[-1, 0], massiv_y[0, 0]])
plt.grid(True)
plt.colorbar()
plt.xlabel('Координата x (метры)')
plt.ylabel('Координата y (метры)')
plt.gca().invert_yaxis()
plt.axis('square')
plt.title('РЛИ №1 в декартовой СК')
plt.show()

# Представление изображения в полярной СК:
points = np.column_stack((massiv_x.flatten(), massiv_y.flatten()))
values = np.abs(Image_dekart[:, :, 0]).flatten()
grid_points = np.column_stack((massiv_x_polar.flatten(), massiv_y_polar.flatten()))
Image_polar = np.zeros((x, y, N_RLD))
Image_polar[:, :, 0] = griddata(points, values, grid_points, method='linear').reshape(x, y)

plt.figure(2)
plt.imshow(np.abs(Image_polar[:, :, 0]),
           extent=[massiv_azimut[0, 0], massiv_azimut[0, -1], massiv_range[-1, 0], massiv_range[0, 0]])
plt.grid(True)
plt.colorbar()
plt.xlabel('Азимут (градусы)')
plt.ylabel('Дальность (метры)')
plt.gca().invert_yaxis()
plt.axis('square')
plt.title('РЛИ №1 в полярной СК')
plt.show()

if N_RLD > 1:  # два РЛИ для двух РЛД
    Image_dekart[:, :, 1] = Image[:, :, 1] / np.max(np.abs(Image[:, :, 1]))
    values = np.abs(Image_dekart[:, :, 1]).flatten()
    Image_polar[:, :, 1] = griddata(points, values, grid_points, method='linear').reshape(x, y)

    plt.figure(3)
    plt.imshow(np.abs(Image_dekart[:, :, 1]), extent=[massiv_x[0, 0], massiv_x[0, -1], massiv_y[-1, 0], massiv_y[0, 0]])
    plt.grid(True)
    plt.colorbar()
    plt.xlabel('Координата x (метры)')
    plt.ylabel('Координата y (метры)')
    plt.gca().invert_yaxis()
    plt.axis('square')
    plt.title('РЛИ №2 в декартовой СК')
    plt.show()

    plt.figure(4)
    plt.imshow(np.abs(Image_polar[:, :, 1]),
               extent=[massiv_azimut[0, 0], massiv_azimut[0, -1], massiv_range[-1, 0], massiv_range[0, 0]])
    plt.grid(True)
    plt.colorbar()
    plt.xlabel('Азимут (градусы)')
    plt.ylabel('Дальность (метры)')
    plt.gca().invert_yaxis()
    plt.axis('square')
    plt.title('РЛИ №2 в полярной СК')
    plt.show()

    if N_RLD > 2:
        Image_dekart[:, :, 2] = Image[:, :, 2] / np.max(np.abs(Image[:, :, 2]))
        values = np.abs(Image_dekart[:, :, 2]).flatten()
        Image_polar[:, :, 2] = griddata(points, values, grid_points, method='linear').reshape(x, y)

        plt.figure(5)
        plt.imshow(np.abs(Image_dekart[:, :, 2]),
                   extent=[massiv_x[0, 0], massiv_x[0, -1], massiv_y[-1, 0], massiv_y[0, 0]])
        plt.grid(True)
        plt.colorbar()
        plt.xlabel('Координата x (метры)')
        plt.ylabel('Координата y (метры)')
        plt.gca().invert_yaxis()
        plt.axis('square')
        plt.title('РЛИ №3 в декартовой СК')
        plt.show()

        plt.figure(6)
        plt.imshow(np.abs(Image_polar[:, :, 2]),
                   extent=[massiv_azimut[0, 0], massiv_azimut[0, -1], massiv_range[-1, 0], massiv_range[0, 0]])
        plt.grid(True)
        plt.colorbar()
        plt.xlabel('Азимут (градусы)')
        plt.ylabel('Дальность (метры)')
        plt.gca().invert_yaxis()
        plt.axis('square')
        plt.title('РЛИ №3 в полярной СК')
        plt.show()

        if N_RLD > 3:
            Image_dekart[:, :, 3] = Image[:, :, 3] / np.max(np.abs(Image[:, :, 3]))
            values = np.abs(Image_dekart[:, :, 3]).flatten()
            Image_polar[:, :, 3] = griddata(points, values, grid_points, method='linear').reshape(x, y)

            plt.figure(7)
            plt.imshow(np.abs(Image_dekart[:, :, 3]),
                       extent=[massiv_x[0, 0], massiv_x[0, -1], massiv_y[-1, 0], massiv_y[0, 0]])
            plt.grid(True)
            plt.colorbar()
            plt.xlabel('Координата x (метры)')
            plt.ylabel('Координата y (метры)')
            plt.gca().invert_yaxis()
            plt.axis('square')
            plt.title('РЛИ №4 в декартовой СК')
            plt.show()

            plt.figure(8)
            plt.imshow(np.abs(Image_polar[:, :, 3]),
                       extent=[massiv_azimut[0, 0], massiv_azimut[0, -1], massiv_range[-1, 0], massiv_range[0, 0]])
            plt.grid(True)
            plt.colorbar()
            plt.xlabel('Азимут (градусы)')
            plt.ylabel('Дальность (метры)')
            plt.gca().invert_yaxis()
            plt.axis('square')
            plt.title('РЛИ №4 в полярной СК')
            plt.show()

