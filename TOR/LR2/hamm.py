import numpy as np
import matplotlib.pyplot as plt

def corr(Nel,d_element):
    output = np.arange(Nel) * d_element - ((Nel - 1) * d_element)/2
    return output

def hamming(coord):
    L_ant = coord[len(coord)-1] - coord[0]
    output = 0.54+0.46*np.cos(2*np.pi*coord/L_ant)
    return output

def convert(input,razr):
    norm = 2**razr
    output = np.round(input*norm)
    return output

def phase_dist(coord,lyambda,angle):
    k = 2*np.pi/lyambda
    output = k*coord*np.sin(angle)
    return output


def antenna_array_pattern(amplitudes, phases, positions, wavelength, phi_scan, phi_array):

    k = 2 * np.pi / wavelength  # волновое число
    # Фазовый сдвиг для электронного сканирования
    phase_shift = -k * positions * np.sin(phi_scan)

    # Вычисляем комплексную ДН
    F = np.zeros_like(phi_array, dtype=complex)
    for n in range(len(positions)):
        F += amplitudes[n] * np.exp(1j * (k * positions[n] * np.sin(phi_array) + phases[n] + phase_shift[n]))

    # Преобразование в дБ
    F_dB = 20 * np.log10(np.abs(F))

    return F_dB

if __name__ == '__main__':
    Nel = 16
    d_element = 8.7e-3/2
    wave_lenght = 8.575e-3
    razryadnost_ampl = 8
    razryadnost_phase = 12
    phi = -20*np.pi/180  # угол сканирования

    coordinats = corr(Nel,d_element)

    hamm_raw = hamming(coordinats)
    hamm_welldone = convert(hamm_raw, razryadnost_ampl)

    phase_raw = phase_dist(coordinats, wave_lenght, phi)
    phase_odnoznach = phase_raw - np.floor(phase_raw / (2*np.pi))*2*np.pi
    phase_welldone = convert(phase_odnoznach/(2*np.pi), razryadnost_phase)

    amplitudes_hamming = hamm_welldone / 2**razryadnost_ampl
    phases_hamming = phase_welldone / 2**razryadnost_phase * 2*np.pi

    amplitudes_uniform = np.ones(Nel)
    phases_uniform = np.zeros(Nel)
    phi_array = np.radians(np.linspace(-90, 90, 1000))

    F_dB_uniform = antenna_array_pattern(amplitudes_uniform, phases_uniform, coordinats, wave_lenght, 0, phi_array)
    F_dB_phase   = antenna_array_pattern(amplitudes_uniform, phases_hamming, coordinats, wave_lenght, 0, phi_array)
    F_dB_hamming = antenna_array_pattern(amplitudes_hamming, phases_hamming, coordinats, wave_lenght, phi, phi_array)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)  # 3 строки, 1 столбец, 1-й график
    plt.plot(np.degrees(phi_array), F_dB_uniform)
    plt.title("Равномерная амплитуда, фаза = 0")
    plt.xlabel("Угол, градусы")
    plt.ylabel("ДН, дБ")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(np.degrees(phi_array), F_dB_phase)
    plt.title("Равномерная амплитуда + фазовое распределение")
    plt.xlabel("Угол, градусы")
    plt.ylabel("ДН, дБ")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(np.degrees(phi_array), F_dB_hamming)
    plt.title("Хэмминг + фазовое распределение")
    plt.xlabel("Угол, градусы")
    plt.ylabel("ДН, дБ")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
