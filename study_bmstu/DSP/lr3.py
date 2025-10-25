import matplotlib.pyplot as plt
import numpy as np

# Данные из таблицы
points = [
    # (I, Q, двоичный код, значение)
    (1, 1, '0000', 0),
    (1, 3, '0001', 1),
    (3, 1, '0010', 2),
    (3, 3, '0011', 3),
    (-1, 1, '0100', 4),
    (-1, 3, '0101', 5),
    (-3, 1, '0110', 6),
    (-3, 3, '0111', 7),
    (1, -1, '1000', 8),
    (1, -3, '1001', 9),
    (3, -1, '1010', 10),
    (3, -3, '1011', 11),
    (-1, -1, '1100', 12),
    (-1, -3, '1101', 13),
    (-3, -1, '1110', 14),
    (-3, -3, '1111', 15)
]

# Создаем график
plt.figure(figsize=(12, 10))

# Рисуем окружность
circle = plt.Circle((0, 0), np.sqrt(18), fill=False, color='gray', linestyle='--', alpha=0.7)
plt.gca().add_patch(circle)

# Рисуем оси
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Рисуем точки и подписи
for I, Q, binary, value in points:
    plt.scatter(I, Q, s=100, alpha=0.7)
    plt.annotate(f'{binary}\n(I={I}, Q={Q})\nval={value}',
                (I, Q),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

# Настройки графика
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlabel('In-phase (I) Component')
plt.ylabel('Quadrature (Q) Component')
plt.title('16-QAM Constellation Diagram\n(Диаграмма созвездия 16-QAM)')
plt.xlim(-4, 4)
plt.ylim(-4, 4)

# Добавляем информацию о радиусе
max_radius = np.sqrt(18)
plt.text(0.5, 3.8, f'Максимальный радиус: √18 ≈ {max_radius:.2f}',
         fontsize=10, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5))

plt.tight_layout()
plt.show()

# Дополнительный график с углами
plt.figure(figsize=(10, 8))

# Рисуем полярную диаграмму
ax = plt.subplot(111, projection='polar')

for I, Q, binary, value in points:
    # Вычисляем амплитуду и фазу
    amplitude = np.sqrt(I**2 + Q**2)
    phase = np.arctan2(Q, I)  # в радианах
    if phase < 0:
        phase += 2 * np.pi  # преобразуем в [0, 2π]

    ax.scatter(phase, amplitude, s=100, alpha=0.7)
    ax.annotate(f'{binary} (val={value})',
               (phase, amplitude),
               xytext=(5, 5),
               textcoords='offset points',
               fontsize=8)

ax.set_title('16-QAM Constellation Diagram (Polar Coordinates)\nДиаграмма в полярных координатах', pad=20)
ax.grid(True)

plt.tight_layout()
plt.show()