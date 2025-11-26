import numpy as np
import matplotlib.pyplot as plt
import re

file_path = r"C:\Users\User74\PycharmProjects\PythonProject\TOR\LR2\hamm_dist.txt"

with open(file_path, 'r') as file:
    content = file.read()

pattern = r'D\d+=(\d+)'
matches = re.findall(pattern, content)

P = np.array([int(match) for match in matches])
angles = np.linspace(-90, 90, len(P))
half_p = np.sqrt(2)/2

plt.figure(figsize=(10, 6))
plt.plot(angles, P / np.max(P),label = 'Диграмма направленности')
plt.axhline(y=half_p,color = 'red',linestyle='--',label = 'уровень половинной мощности')
plt.xlabel('Азимут, град.')
plt.ylabel('Мощность')
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xticks(np.arange(-100, 100, 10))
plt.title('ДН по азимуту')
plt.legend()
plt.grid(True)
plt.show()