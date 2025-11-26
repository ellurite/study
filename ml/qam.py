import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square, sawtooth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report


# --- Генерация сигналов ---
def generate_signals(n_samples=400, n_points=256, fs=1000):
    """
    Генерация сигналов 4 типов: синус, прямоугольный, пилообразный, белый шум
    """
    X, y = [], []
    t = np.linspace(0, 1, n_points)

    for _ in range(n_samples):
        f = np.random.uniform(10, 100)  # случайная частота
        phase = np.random.uniform(0, 2 * np.pi)  # случайная фаза
        noise_amp = np.random.uniform(0.1, 0.3)  # случайная амплитуда шума

        noise = np.random.normal(0, noise_amp, n_points)

        # 4 класса сигналов
        s1 = np.sin(2 * np.pi * f * t + phase) + noise  # Синус
        s2 = square(2 * np.pi * f * t + phase) + noise  # Прямоугольный
        s3 = sawtooth(2 * np.pi * f * t + phase) + noise  # Пилообразный
        s4 = np.random.normal(0, 1, n_points)  # Белый шум

        X.extend([s1, s2, s3, s4])
        y.extend([0, 1, 2, 3])

    return np.array(X), np.array(y)


# --- Улучшенные признаки ---
def extract_improved_features(X, n_points=256):

    feats = []
    for x in X:
        # Спектральные признаки
        spectrum = np.abs(np.fft.fft(x))[:n_points // 2]

        # Статистические признаки во временной области
        time_features = [
            np.mean(x),  # среднее значение
            np.std(x),  # стандартное отклонение
            np.max(x) - np.min(x),  # размах
            np.median(x),  # медиана
        ]

        # Статистические признаки в частотной области
        freq_features = [
            np.mean(spectrum),  # средняя амплитуда спектра
            np.std(spectrum),  # стандартное отклонение спектра
            np.argmax(spectrum),  # позиция максимума спектра (основная частота)
        ]

        # Объединяем все признаки
        combined_features = np.concatenate([
            spectrum[:20],  # первые 20 компонент спектра
            time_features,
            freq_features
        ])

        feats.append(combined_features)

    return np.array(feats)


# --- Основной код ---
np.random.seed(42)  # для воспроизводимости результатов

# Генерация данных
print("Генерация сигналов...")
X_time, y = generate_signals(n_samples=500, n_points=256)

# Извлечение признаков
print("Извлечение признаков...")
X_features = extract_improved_features(X_time)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42, stratify=y
)

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Размерность признаков: {X_features.shape[1]}")
print(f"Обучающая выборка: {X_train.shape[0]} samples")
print(f"Тестовая выборка: {X_test.shape[0]} samples")

# Обучение модели
print("Обучение модели...")
model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
model.fit(X_train_scaled, y_train)

# Предсказание и оценка
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"\nТочность классификации: {acc:.3f}")
print("\nОтчет по классификации:")
print(classification_report(y_test, y_pred, target_names=["sin", "square", "saw", "noise"]))

# --- Визуализация ---
plt.style.use('default')
fig = plt.figure(figsize=(15, 10))

# 1. Матрица ошибок
plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["sin", "square", "saw", "noise"])
disp.plot(cmap='viridis', ax=plt.gca())
plt.title(f'Матрица ошибок (Accuracy: {acc:.3f})')

# 2. Примеры сигналов во временной области
plt.subplot(2, 2, 2)
t = np.linspace(0, 1, 256)
for i, label in enumerate(["sin", "square", "saw", "noise"]):
    signal = X_time[y == i][0]  # первый сигнал каждого класса
    plt.plot(t, signal, label=label, alpha=0.7)
plt.xlabel('Время')
plt.ylabel('Амплитуда')
plt.title('Примеры сигналов (временная область)')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Спектры сигналов
plt.subplot(2, 2, 3)
freqs = np.fft.fftfreq(256, 1 / 1000)[:128]  # частотная ось
for i, label in enumerate(["sin", "square", "saw", "noise"]):
    signal = X_time[y == i][0]
    spectrum = np.abs(np.fft.fft(signal))[:128]
    plt.plot(freqs, spectrum, label=label, alpha=0.7)
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.title('Амплитудные спектры')
plt.legend()
plt.grid(True, alpha=0.3)

