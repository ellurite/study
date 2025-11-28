import numpy as np
import matplotlib.pyplot as plt


def corr(Nel, d_element):
    output = np.arange(Nel) * d_element - ((Nel - 1) * d_element) / 2
    return output


def hamming(coord):
    L_ant = coord[len(coord) - 1] - coord[0]
    output = 0.54 + 0.46 * np.cos(2 * np.pi * coord / L_ant)
    return output


def convert(input, razr):
    norm = 2 ** razr
    output = np.round(input * norm)
    return output


def phase_dist(coord, lyambda, angle):
    k = 2 * np.pi / lyambda
    output = k * coord * np.sin(angle)
    return output


def antenna_array_pattern(amplitudes, phases, positions, wavelength, phi_scan, phi_array):
    k = 2 * np.pi / wavelength
    phase_shift = -k * positions * np.sin(phi_scan)

    F = np.zeros_like(phi_array, dtype=complex)
    for n in range(len(positions)):
        F += amplitudes[n] * np.exp(1j * (k * positions[n] * np.sin(phi_array) + phases[n] + phase_shift[n]))

    return F



def beamwidth(angles, P_db, center_angle=0, penalty_strength=0.1):
    angles = np.array(angles)
    P_db = np.array(P_db)

    penalty = -penalty_strength * np.abs(angles - center_angle)

    penalty = penalty * (np.max(P_db) - np.min(P_db)) / np.max(np.abs(penalty))

    penalized_power = P_db + penalty

    main_idx = np.argmax(penalized_power)
    peak_power = P_db[main_idx]
    level_3db = peak_power - 3

    left_idx = None
    right_idx = None

    # Двигаемся ВЛЕВО от максимума до уровня -3 дБ
    for i in range(main_idx, 0, -1):
        if P_db[i] >= level_3db and P_db[i - 1] < level_3db:
            x1, x2 = angles[i - 1], angles[i]
            y1, y2 = P_db[i - 1], P_db[i]
            if y2 != y1:
                left_idx = x1 + (x2 - x1) * (level_3db - y1) / (y2 - y1)
            else:
                left_idx = angles[i]
            break

    # Двигаемся ВПРАВО от максимума до уровня -3 дБ
    for i in range(main_idx, len(P_db) - 1):
        if P_db[i] >= level_3db and P_db[i + 1] < level_3db:
            x1, x2 = angles[i], angles[i + 1]
            y1, y2 = P_db[i], P_db[i + 1]
            if y2 != y1:
                right_idx = x1 + (x2 - x1) * (level_3db - y1) / (y2 - y1)
            else:
                right_idx = angles[i]
            break

    # Если не нашли пересечений, возвращаем None (а не границы диапазона!)
    if left_idx is None or right_idx is None:
        return None

    width = right_idx - left_idx
    return width


def max_sidelobe_level(P):
    P = np.array(P)
    main_idx = np.argmax(P)
    step = 5

    left = main_idx - step
    left_lobe = 0
    while left > step:
        if P[left] > P[left - step] and P[left] > P[left + step]:
            left_lobe = P[left]
            break
        left -= step

    right = main_idx + step
    right_lobe = 0
    while right < len(P) - step:
        if P[right] > P[right - step] and P[right] > P[right + step]:
            right_lobe = P[right]
            break
        right += step

    sidelobe = max(left_lobe, right_lobe)
    return 20 * np.log10(sidelobe)


def directivity(P_db, angles_deg):

    F_linear = 10 ** (P_db / 20)

    F_norm = F_linear / np.max(F_linear)

    angles_rad = np.radians(angles_deg)
    delta_phi = angles_rad[1] - angles_rad[0]

    integral_F2 = np.sum(F_norm ** 2) * delta_phi
    D = 2 * np.pi / integral_F2

    return 10 * np.log10(D)

if __name__ == '__main__':
    Nel = 16
    d_element = 8.7e-3
    wave_lenght = 8.575e-3
    razryadnost_ampl = 8
    razryadnost_phase = 12
    phi = 20 * np.pi / 180  # угол сканирования
    half_p = -3
    coordinats = corr(Nel, d_element)

    hamm_raw = hamming(coordinats)
    hamm_welldone = convert(hamm_raw, razryadnost_ampl)

    phase_raw = phase_dist(coordinats, wave_lenght, phi)
    phase_odnoznach = phase_raw - np.floor(phase_raw / (2 * np.pi)) * 2 * np.pi
    phase_welldone = convert(phase_odnoznach / (2 * np.pi), razryadnost_phase)

    amplitudes_hamming = hamm_welldone / 2 ** razryadnost_ampl
    phases_hamming = phase_welldone / 2 ** razryadnost_phase * 2 * np.pi

    amplitudes_uniform = np.ones(Nel)
    phases_uniform = np.zeros(Nel)
    phi_array = np.radians(np.linspace(-90, 90, 1000))

    F_uniform = antenna_array_pattern(amplitudes_uniform, phases_uniform, coordinats, wave_lenght, 0, phi_array)
    F_uniform_abs = np.abs(F_uniform)
    F_uniform_dB = 20 * np.log10(F_uniform_abs / np.max(F_uniform_abs))

    F_phase = antenna_array_pattern(amplitudes_uniform, phases_hamming, coordinats, wave_lenght, 0, phi_array)
    F_phase_abs = np.abs(F_phase)
    F_phase_dB = 20 * np.log10(F_phase_abs / np.max(F_phase_abs))

    F_hamming = antenna_array_pattern(amplitudes_hamming, phases_uniform, coordinats, wave_lenght,0, phi_array)
    F_hamming_abs = np.abs(F_hamming)
    F_hamming_dB = 20 * np.log10(F_hamming_abs / np.max(F_hamming_abs))

    angles_deg = np.degrees(phi_array)

    UBL_uniform = max_sidelobe_level(F_uniform_abs / np.max(F_uniform_abs))
    bw_uniform = beamwidth(angles_deg, F_uniform_dB)
    D_uniform = directivity(F_uniform_abs, angles_deg)

    print("=== РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ ===")
    print("КНД =", D_uniform, "дБ")
    print("Ширина главного лепестка:", bw_uniform, "градусов")
    print("УБЛ:", UBL_uniform, "дБ")
    print()

    # Вычисляем параметры для фазового распределения
    UBL_phase = max_sidelobe_level(F_phase_abs / np.max(F_phase_abs))
    bw_phase = beamwidth(angles_deg, F_phase_dB)
    D_phase = directivity(F_phase_abs, angles_deg)

    print("=== ФАЗОВОЕ РАСПРЕДЕЛЕНИЕ ===")
    print("КНД =", D_phase, "дБ")
    print("Ширина главного лепестка:", bw_phase, "градусов")
    print("УБЛ:", UBL_phase, "дБ")
    print()

    UBL_hamming = max_sidelobe_level(F_hamming_abs / np.max(F_hamming_abs))
    bw_hamming = beamwidth(angles_deg, F_hamming_dB)
    D_hamming = directivity(F_hamming_abs, angles_deg)

    print("=== РАСПРЕДЕЛЕНИЕ ХЭММИНГА ===")
    print("КНД =", D_hamming, "дБ")
    print("Ширина главного лепестка:", bw_hamming, "градусов")
    print("УБЛ:", UBL_hamming, "дБ")

    plt.figure(figsize=(10, 6))
    plt.plot(angles_deg, F_uniform_dB, 'black', linewidth=2,)
    plt.axhline(y=half_p, color='black', linestyle='--', label='уровень половинной мощности')
    plt.title('Равномерное распределение', fontsize=14)
    plt.xlabel('Угол, градусы', fontsize=14)
    plt.ylabel('Уровень, дБ', fontsize=14)
    plt.grid(True, alpha=0.7)
    plt.legend()
    plt.yticks(np.arange(-70, 5, 5))
    plt.xticks(np.arange(-100, 100, 5))
    plt.tight_layout()


    plt.figure(figsize=(10, 6))
    plt.plot(angles_deg, F_phase_dB, 'black', linewidth=2)
    plt.axhline(y=half_p, color='black', linestyle='--', label='уровень половинной мощности')
    plt.title('Фазовое распределение', fontsize=14)
    plt.xlabel('Угол, градусы', fontsize=14)
    plt.ylabel('Уровень, дБ', fontsize=14)
    plt.grid(True, alpha=0.7)
    plt.yticks(np.arange(-85, 5, 5))
    plt.legend()
    plt.xticks(np.arange(-100, 100, 5))
    plt.tight_layout()


    plt.figure(figsize=(10, 6))
    plt.plot(angles_deg, F_hamming_dB, 'black', linewidth=2)
    plt.axhline(y=half_p, color='black', linestyle='--', label='уровень половинной мощности')
    plt.title('Распределение Хэмминга', fontsize=14)
    plt.xlabel('Угол, градусы', fontsize=14)
    plt.ylabel('Уровень, дБ', fontsize=14)
    plt.grid(True, alpha=0.7)
    plt.yticks(np.arange(-95, 5, 5))
    plt.legend()
    plt.xticks(np.arange(-100, 100, 5))
    plt.tight_layout()
    plt.show()
