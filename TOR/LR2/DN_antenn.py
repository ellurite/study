import numpy as np
import matplotlib.pyplot as plt
import re

file_path = r"C:\Users\superpro2005\Desktop\study\python\TOR\LR2\uniform_dist.txt"

with open(file_path, 'r') as file:
    content = file.read()

pattern = r'D\d+=(\d+)'
matches = re.findall(pattern, content)
P = np.array([int(match) for match in matches])
P_normalize = P/np.max(P)
angles = np.linspace(-90, 90, len(P))
half_p = -3


def beamwidth(angles, P):
    angles = np.array(angles)
    P = np.array(P)

    mask = P >= 0.707

    angles_wide = angles[mask]

    angle_min = np.min(angles_wide)
    angle_max = np.max(angles_wide)

    width = angle_max - angle_min

    return width

def max_sidelobe_level(P):
    P = np.array(P)
    main_idx = np.argmax(P)
    step = 5

    left = main_idx - step
    left_lobe = 0
    while left > step:
        if P[left] > P[left-step] and P[left] > P[left+step]:
            left_lobe = P[left]
            break
        left -= step

    right = main_idx + step
    right_lobe = 0
    while right < len(P) - step:
        if P[right] > P[right-step] and P[right] > P[right+step]:
            right_lobe = P[right]
            break
        right += step

    sidelobe = max(left_lobe, right_lobe)
    return 20 * np.log10(sidelobe)

def directivity(P_db, angles_deg):
    P_db = np.array(P_db)
    angles_deg = np.array(angles_deg)

    F = 10 ** (P_db / 10)

    delta_phi = np.deg2rad(angles_deg[1] - angles_deg[0])

    D = 2 * np.pi / (np.sum(F**2) * delta_phi)

    D_db = 10 * np.log10(D)

    return D_db

UBL = max_sidelobe_level(P_normalize)
bw= beamwidth(angles,P_normalize)
D_db = directivity(20*np.log10(P_normalize), angles)
print("КНД =", D_db, "дБ")
print("Ширина главного лепестка:", bw)
print("УБЛ:", UBL, "дБ")

plt.figure(figsize=(10, 6))
text = (
    f"Ширина главного лепестка: {bw:.2f}°\n"
    f"УБЛ: {UBL:.2f} dB\n"
    f"КНД: {D_db:.2f} dB"
)
plt.text(0.60, 0.28, text, transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.7))

plt.plot(angles,20*np.log10(P_normalize),label = 'Диграмма направленности')
plt.axhline(y=half_p,color = 'red',linestyle='--',label = 'уровень половинной мощности')
plt.xlabel('Азимут, град.')
plt.ylabel('Мощность, ДБ')
plt.yticks(np.arange(-60, 5, 5))
plt.xticks(np.arange(-90, 100, 10))
plt.title('ДН по азимуту')
plt.legend()
plt.grid(True)
plt.show()