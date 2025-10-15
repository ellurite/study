import customtkinter as ctk
from tkinter import messagebox
import cv2
import numpy as np
import random
from PIL import Image
import io


class ImageGenerator:
    def __init__(self):
        self.objects = {
            "самолет": self.draw_airplane_radar,
            "дом": self.draw_house_radar
        }
        self.generated_image = None
        self.placed_objects = []  # Сохраняем информацию о размещенных объектах

    def create_generator_window(self, parent_callback):
        """Создание окна генератора радарных изображений"""
        self.generator_window = ctk.CTkToplevel()
        self.generator_window.title("Генератор радарных изображений")
        self.generator_window.geometry("500x600")
        self.generator_window.transient()
        self.generator_window.grab_set()

        self.parent_callback = parent_callback
        self.setup_generator_ui()

    def setup_generator_ui(self):
        """Настройка интерфейса генератора"""
        # Заголовок
        title_label = ctk.CTkLabel(self.generator_window, text="Генератор радарных изображений",
                                   font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=20)

        # Настройки изображения
        settings_frame = ctk.CTkFrame(self.generator_window)
        settings_frame.pack(pady=10, padx=20, fill="x")

        ctk.CTkLabel(settings_frame, text="Настройки радарного изображения:").pack(pady=10)

        # Размер изображения
        size_frame = ctk.CTkFrame(settings_frame)
        size_frame.pack(pady=5, padx=10, fill="x")

        ctk.CTkLabel(size_frame, text="Ширина:").pack(side="left", padx=5)
        self.width_var = ctk.StringVar(value="512")
        width_entry = ctk.CTkEntry(size_frame, textvariable=self.width_var, width=80)
        width_entry.pack(side="left", padx=5)

        ctk.CTkLabel(size_frame, text="Высота:").pack(side="left", padx=5)
        self.height_var = ctk.StringVar(value="512")
        height_entry = ctk.CTkEntry(size_frame, textvariable=self.height_var, width=80)
        height_entry.pack(side="left", padx=5)

        # Выбор объектов
        objects_frame = ctk.CTkFrame(self.generator_window)
        objects_frame.pack(pady=10, padx=20, fill="x")

        ctk.CTkLabel(objects_frame, text="Количество объектов:").pack(pady=10)

        self.airplane_count_var = ctk.StringVar(value="2")
        self.house_count_var = ctk.StringVar(value="3")

        # Количество самолетов
        airplane_frame = ctk.CTkFrame(objects_frame)
        airplane_frame.pack(pady=5, padx=10, fill="x")

        ctk.CTkLabel(airplane_frame, text="Самолеты:").pack(side="left", padx=5)
        airplane_count = ctk.CTkEntry(airplane_frame, textvariable=self.airplane_count_var, width=50)
        airplane_count.pack(side="left", padx=5)

        # Количество домов
        house_frame = ctk.CTkFrame(objects_frame)
        house_frame.pack(pady=5, padx=10, fill="x")

        ctk.CTkLabel(house_frame, text="Дома:").pack(side="left", padx=5)
        house_count = ctk.CTkEntry(house_frame, textvariable=self.house_count_var, width=50)
        house_count.pack(side="left", padx=5)

        # Настройки сигнала
        signal_frame = ctk.CTkFrame(self.generator_window)
        signal_frame.pack(pady=10, padx=20, fill="x")

        ctk.CTkLabel(signal_frame, text="Интенсивность сигнала:").pack(pady=5)

        self.signal_intensity_var = ctk.StringVar(value="medium")
        signal_intensities = ["weak", "medium", "strong"]

        signal_menu = ctk.CTkOptionMenu(signal_frame, variable=self.signal_intensity_var, values=signal_intensities)
        signal_menu.pack(pady=5)

        # Шум
        noise_frame = ctk.CTkFrame(self.generator_window)
        noise_frame.pack(pady=10, padx=20, fill="x")

        ctk.CTkLabel(noise_frame, text="Уровень шума:").pack(pady=5)

        self.noise_var = ctk.DoubleVar(value=0.1)
        noise_slider = ctk.CTkSlider(noise_frame, variable=self.noise_var, from_=0, to=0.5)
        noise_slider.pack(pady=5, padx=10, fill="x")

        self.noise_label = ctk.CTkLabel(noise_frame, text="Шум: 0.10")
        self.noise_label.pack()

        # Кнопки управления
        buttons_frame = ctk.CTkFrame(self.generator_window)
        buttons_frame.pack(pady=20, padx=20, fill="x")

        generate_btn = ctk.CTkButton(buttons_frame, text="Сгенерировать радарное изображение",
                                     command=self.generate_and_display)
        generate_btn.pack(pady=10, fill="x")

        use_btn = ctk.CTkButton(buttons_frame, text="Использовать в анализаторе",
                                command=self.use_in_analyzer)
        use_btn.pack(pady=10, fill="x")

        # Превью
        preview_frame = ctk.CTkFrame(self.generator_window)
        preview_frame.pack(pady=10, padx=20, fill="both", expand=True)

        ctk.CTkLabel(preview_frame, text="Превью радарного изображения:").pack(pady=5)

        self.preview_label = ctk.CTkLabel(preview_frame, text="Изображение не сгенерировано")
        self.preview_label.pack(pady=10, fill="both", expand=True)

        # Информация о сгенерированных объектах
        self.info_label = ctk.CTkLabel(self.generator_window, text="", text_color="lightblue")
        self.info_label.pack(pady=5)

        # Обновление текста слайдера
        def update_noise_text(*args):
            self.noise_label.configure(text=f"Шум: {self.noise_var.get():.2f}")

        self.noise_var.trace('w', update_noise_text)

    def get_signal_intensity(self):
        """Получение интенсивности сигнала"""
        intensities = {
            "weak": 150,
            "medium": 200,
            "strong": 255
        }
        return intensities.get(self.signal_intensity_var.get(), 200)

    def generate_and_display(self):
        """Генерация и отображение радарного изображения"""
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            airplane_count = int(self.airplane_count_var.get())
            house_count = int(self.house_count_var.get())
            noise_level = self.noise_var.get()

            if width <= 0 or height <= 0:
                messagebox.showerror("Ошибка", "Размеры должны быть положительными числами")
                return

            # Создание списка объектов
            objects_list = ["самолет"] * airplane_count + ["дом"] * house_count
            random.shuffle(objects_list)

            # Генерация радарного изображения
            self.generated_image, self.placed_objects = self.generate_radar_image(
                width, height, objects_list, noise_level
            )

            # Отображение превью
            self.display_preview(self.generated_image)

            # Обновление информации
            self.info_label.configure(
                text=f"Сгенерировано: {airplane_count} самолетов, {house_count} домов\n"
                     f"Размер: {width}x{height} | Объекты готовы к анализу!"
            )

        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные числовые значения")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка генерации: {str(e)}")

    def generate_radar_image(self, width=512, height=512, objects_list=None, noise_level=0.1):
        """Генерация радарного изображения (черный фон с белыми метками)"""
        # Создание черного фона
        image_array = np.zeros((height, width), dtype=np.uint8)

        # Размещение объектов
        placed_objects = []
        for obj_type in objects_list:
            if obj_type in self.objects:
                # Случайная позиция с проверкой на перекрытие
                x, y = self.find_free_position(placed_objects, width, height, obj_type)
                if x is not None:
                    # Рисование объекта на массиве
                    self.objects[obj_type](image_array, x, y)
                    placed_objects.append({
                        'type': obj_type,
                        'x': x, 'y': y,
                        'width': 40, 'height': 40
                    })

        # Добавление шума
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 50, (height, width))
            image_array = np.clip(image_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Конвертация в BGR для отображения
        opencv_image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        return opencv_image, placed_objects

    def find_free_position(self, placed_objects, width, height, obj_type, max_attempts=100):
        """Поиск свободной позиции для объекта"""
        obj_size = 60 if obj_type == "самолет" else 40

        for _ in range(max_attempts):
            x = random.randint(obj_size, width - obj_size - 1)
            y = random.randint(obj_size, height - obj_size - 1)

            # Проверка на перекрытие с существующими объектами
            overlap = False
            for obj in placed_objects:
                existing_size = 60 if obj['type'] == "самолет" else 40
                if (abs(x - obj['x']) < obj_size + existing_size and
                        abs(y - obj['y']) < obj_size + existing_size):
                    overlap = True
                    break

            if not overlap:
                return x, y

        return None, None

    def draw_airplane_radar(self, image_array, x, y):
        """Рисование самолета в радарном представлении"""
        intensity = self.get_signal_intensity()

        # Основная линия (фюзеляж)
        cv2.line(image_array, (x - 15, y), (x + 15, y), intensity, 2)

        # Крылья
        cv2.line(image_array, (x - 8, y - 5), (x + 8, y - 5), intensity, 2)
        cv2.line(image_array, (x, y - 5), (x, y + 5), intensity, 2)

        # Хвостовое оперение
        cv2.line(image_array, (x - 12, y - 3), (x - 12, y + 3), intensity, 1)

        # Усиление сигнала в центре
        cv2.circle(image_array, (x, y), 3, intensity, -1)

        # Добавление рассеянных точек вокруг (эффект радара)
        for i in range(5):
            px = x + random.randint(-8, 8)
            py = y + random.randint(-8, 8)
            cv2.circle(image_array, (px, py), 1, intensity // 2, -1)

    def draw_house_radar(self, image_array, x, y):
        """Рисование дома в радарном представлении"""
        intensity = self.get_signal_intensity()

        # Прямоугольник основания
        cv2.rectangle(image_array, (x - 12, y - 8), (x + 12, y + 8), intensity, 1)

        # Крыша (треугольник)
        cv2.line(image_array, (x - 12, y - 8), (x, y - 15), intensity, 1)
        cv2.line(image_array, (x + 12, y - 8), (x, y - 15), intensity, 1)

        # Заполнение (более слабый сигнал)
        cv2.rectangle(image_array, (x - 10, y - 6), (x + 10, y + 6), intensity // 3, -1)

        # Точки внутри (внутренняя структура)
        for i in range(3):
            px = x + random.randint(-8, 8)
            py = y + random.randint(-4, 4)
            cv2.circle(image_array, (px, py), 1, intensity // 2, -1)

    def display_preview(self, image):
        """Отображение превью изображения"""
        if image is not None:
            # Масштабирование для превью
            preview_size = (200, 200)
            resized_image = cv2.resize(image, preview_size)

            # Конвертация для CustomTkinter
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image,
                                     size=preview_size)

            self.preview_label.configure(image=ctk_image, text="")

    def use_in_analyzer(self):
        """Использование сгенерированного изображения в анализаторе"""
        if self.generated_image is None:
            messagebox.showerror("Ошибка", "Сначала сгенерируйте изображение!")
            return

        # Передаем изображение И информацию о реальных объектах в основное приложение
        result_data = {
            'image': self.generated_image,
            'true_objects': self.placed_objects  # Реальные координаты объектов
        }

        # Передаем данные в основное приложение
        self.parent_callback(result_data)

        # Закрытие окна генератора
        self.generator_window.destroy()

        messagebox.showinfo("Успех", "Изображение загружено в анализатор!\n"
                                     "Теперь можно запустить анализ для обнаружения объектов.")