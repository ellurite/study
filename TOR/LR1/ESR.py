import numpy as np
import matplotlib.pyplot as  plt
from sympy.printing.pretty.pretty_symbology import line_width


def comma_dot(text):
    return text.replace(',','.')

data = np.loadtxt(r'/TOR/LR1/B2.txt', converters = comma_dot)
angles, values = data[:,0], data[:,1]

sector_start = 0
sector_end = 360

angles_rad = np.deg2rad(angles)

in_sector = (angles >= sector_start)&(angles<= sector_end)
sector_angles = angles_rad[in_sector]
sector_values = values[in_sector]

roc_mean = np.mean(sector_values)

plt.figure(figsize=(10,10))
ax = plt.subplot(projection='polar')
ax.plot(angles_rad,values,label = "Измеренные данные")


sector_rad = np.deg2rad([sector_start,sector_end])
theta = np.linspace(sector_rad[0],sector_rad[1],100)

ax.fill_between(theta, 0 ,np.max(values)*1.1,
                color = 'red', alpha = 0.2,
                label = 'Анализируемый спектр')
ax.plot(theta,[roc_mean]*len(theta),'r-',linewidth=3,
        label = f'Средняя ЭПР = {roc_mean:.2f}')

ax.set_title('Диаграмма обратного рассеяния')
ax.set_ylim(0,np.max(values)*1.1)
ax.legend(loc='best')

plt.tight_layout()
plt.show()