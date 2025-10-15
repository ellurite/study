import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image
import os
import sys
import traceback

# Добавляем путь к текущей директории для импортов
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from image_processing import ImageProcessor
    from neural_network import ObjectDetector
    from config import SUPPORTED_OBJECTS, APP_SETTINGS
    from image_generator import ImageGenerator
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    traceback.print_exc()


    # Создаем заглушки для тестирования
    class ImageProcessor:
        def polar_to_cartesian(self, img): return img

        def analyze_properties(self, img): return {}


    class ObjectDetector:
        def detect_objects(self, img, classes):
            return {'objects': [], 'annotated_image': img}


    SUPPORTED_OBJECTS = ["самолет", "дом"]
    APP_SETTINGS = {"model_confidence": 0.5}


class MainApplication:
    def __init__(self):
        print("Инициализация приложения...")

        # Настройка темы
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Анализатор радарных изображений")
        self.root.geometry("1200x700")

        # Защита от ошибок инициализации
        try:
            self.image_processor = ImageProcessor()
            self.object_detector = ObjectDetector()
        except Exception as e:
            print(f"Ошибка инициализации модулей: {e}")
            self.image_processor = ImageProcessor()
            self.object_detector = ObjectDetector()

        self.current_image = None
        self.true_objects = []  # Реальные объекты для сравнения

        print("Настройка интерфейса...")
        self.setup_ui()
        print("Приложение готово к работе!")

    def setup_ui(self):
        # Создание основных фреймов
        self.left_frame = ctk.CTkFrame(self.root, width=300)
        self.left_frame.pack(side="left", fill="y", padx=10, pady=10)

        self.right_frame = ctk.CTkFrame(self.root)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.setup_left_panel()
        self.setup_right_panel()

    def setup_left_panel(self):
        # Заголовок
        title_label = ctk.CTkLabel(self.left_frame, text="Анализатор радарных изображений",
                                   font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=20)

        # Кнопка загрузки изображения
        self.load_btn = ctk.CTkButton(self.left_frame, text="Загрузить изображение",
                                      command=self.load_image)
        self.load_btn.pack(pady=10, padx=20, fill="x")

        # Кнопка генерации изображения
        self.generate_btn = ctk.CTkButton(self.left_frame, text="Сгенерировать радарное изображение",
                                          command=self.open_generator)
        self.generate_btn.pack(pady=10, padx=20, fill="x")

        # Выбор объектов для поиска
        objects_label = ctk.CTkLabel(self.left_frame, text="Выберите объекты для поиска:")
        objects_label.pack(pady=(20, 10))

        self.object_vars = {}
        for obj in SUPPORTED_OBJECTS:
            var = ctk.BooleanVar(value=True)
            cb = ctk.CTkCheckBox(self.left_frame, text=obj, variable=var)
            cb.pack(pady=5, padx=20, anchor="w")
            self.object_vars[obj] = var

        # Кнопка анализа
        self.analyze_btn = ctk.CTkButton(self.left_frame, text="Начать анализ",
                                         command=self.analyze_image, state="disabled")
        self.analyze_btn.pack(pady=20, padx=20, fill="x")

        # Панель результатов
        self.results_frame = ctk.CTkFrame(self.left_frame)
        self.results_frame.pack(pady=10, padx=10, fill="x")

        self.results_label = ctk.CTkLabel(self.results_frame, text="Результаты:",
                                          font=ctk.CTkFont(weight="bold"))
        self.results_label.pack(pady=10)

        self.results_text = ctk.CTkTextbox(self.results_frame, height=150)
        self.results_text.pack(pady=10, padx=10, fill="both", expand=True)

    def setup_right_panel(self):
        # Вкладки для отображения изображений
        self.tabview = ctk.CTkTabview(self.right_frame)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        self.original_tab = self.tabview.add("Оригинал")
        self.processed_tab = self.tabview.add("Обработанное")
        self.detection_tab = self.tabview.add("Обнаружение")
        self.comparison_tab = self.tabview.add("Сравнение")  # Новая вкладка!

        # Настройка отображения изображений
        self.setup_image_display(self.original_tab, "original_image")
        self.setup_image_display(self.processed_tab, "processed_image")
        self.setup_image_display(self.detection_tab, "detection_image")
        self.setup_image_display(self.comparison_tab, "comparison_image")

    def setup_image_display(self, parent, attribute_name):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Метка для изображения
        label = ctk.CTkLabel(frame, text="Изображение не загружено",
                             font=ctk.CTkFont(size=16))
        label.pack(expand=True)

        setattr(self, attribute_name + "_label", label)
        setattr(self, attribute_name + "_frame", frame)

    def open_generator(self):
        """Открытие окна генератора изображений"""
        try:
            from image_generator import ImageGenerator
            generator = ImageGenerator()
            generator.create_generator_window(self.load_generated_image)
        except Exception as e:
            self.show_error(f"Ошибка открытия генератора: {str(e)}")

    def load_generated_image(self, result_data):
        """Загрузка сгенерированного изображения из памяти"""
        try:
            self.current_image = result_data['image']
            self.true_objects = result_data['true_objects']  # Сохраняем реальные объекты

            # Отображаем оригинальное изображение
            self.display_image(self.current_image, "original_image")

            # Показываем реальные объекты на отдельном изображении
            comparison_image = self.show_true_objects(self.current_image.copy())
            self.display_image(comparison_image, "comparison_image")

            self.analyze_btn.configure(state="normal")
            self.show_message("Сгенерированное изображение загружено!\nРеальные объекты отмечены красными квадратами.")

        except Exception as e:
            self.show_error(f"Ошибка загрузки: {str(e)}")

    def show_true_objects(self, image):
        """Показывает реальные объекты красными квадратами"""
        for obj in self.true_objects:
            x, y = obj['x'], obj['y']
            # Рисуем красный квадрат вокруг реального объекта
            cv2.rectangle(image,
                          (x - 20, y - 20),
                          (x + 20, y + 20),
                          (0, 0, 255), 2)  # Красный цвет

            # Подписываем тип объекта
            cv2.putText(image, obj['type'],
                        (x - 25, y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return image

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )

        if file_path:
            try:
                self.current_image = cv2.imread(file_path)
                self.true_objects = []  # Очищаем реальные объекты для загруженных изображений

                if self.current_image is not None:
                    self.display_image(self.current_image, "original_image")
                    self.analyze_btn.configure(state="normal")
                    self.show_message("Изображение загружено успешно!")
                else:
                    self.show_error("Не удалось загрузить изображение!")
            except Exception as e:
                self.show_error(f"Ошибка загрузки: {str(e)}")

    def display_image(self, image, target):
        """Отображение изображения в указанном месте"""
        if image is not None:
            try:
                # Конвертация для tkinter
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)

                # Получаем размеры фрейма
                frame = getattr(self, target + "_frame")
                frame.update()
                frame_width = frame.winfo_width() - 40
                frame_height = frame.winfo_height() - 40

                if frame_width > 1 and frame_height > 1:
                    # Масштабирование с сохранением пропорций
                    pil_image.thumbnail((frame_width, frame_height), Image.Resampling.LANCZOS)

                ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image,
                                         size=pil_image.size)

                label = getattr(self, target + "_label")
                label.configure(image=ctk_image, text="")

            except Exception as e:
                print(f"Ошибка отображения изображения: {e}")

    def analyze_image(self):
        if self.current_image is None:
            self.show_error("Сначала загрузите изображение!")
            return

        try:
            # Получение выбранных объектов
            selected_objects = [obj for obj, var in self.object_vars.items() if var.get()]

            if not selected_objects:
                self.show_error("Выберите хотя бы один объект для поиска!")
                return

            # Показываем сообщение о начале анализа
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", "Выполняется анализ...\n")
            self.root.update()

            # Преобразование координат (если нужно)
            processed_image = self.image_processor.polar_to_cartesian(self.current_image)
            self.display_image(processed_image, "processed_image")

            # Анализ параметров
            properties = self.image_processor.analyze_properties(processed_image)

            # Поиск объектов
            detection_result = self.object_detector.detect_objects(
                processed_image, selected_objects
            )

            # Отображение результатов
            self.show_results(properties, detection_result)

            # Отображение изображения с обнаруженными объектами
            if detection_result['annotated_image'] is not None:
                self.display_image(detection_result['annotated_image'], "detection_image")

        except Exception as e:
            self.show_error(f"Ошибка анализа: {str(e)}")

    def show_results(self, properties, detection):
        results_text = "=== ПАРАМЕТРЫ ИЗОБРАЖЕНИЯ ===\n"
        for key, value in properties.items():
            results_text += f"{key}: {value}\n"

        results_text += "\n=== ОБНАРУЖЕННЫЕ ОБЪЕКТЫ ===\n"
        if detection['objects']:
            for obj in detection['objects']:
                results_text += f"- {obj['class']}: уверенность {obj['confidence']:.2f}\n"
                if 'bbox' in obj:
                    results_text += f"  координаты: {obj['bbox']}\n"
        else:
            results_text += "Объекты не найдены\n"

        # Добавляем информацию о реальных объектах (если есть)
        if self.true_objects:
            results_text += f"\n=== РЕАЛЬНЫЕ ОБЪЕКТЫ ===\n"
            airplane_count = sum(1 for obj in self.true_objects if obj['type'] == 'самолет')
            house_count = sum(1 for obj in self.true_objects if obj['type'] == 'дом')
            results_text += f"Самолетов: {airplane_count}, Домов: {house_count}\n"
            results_text += f"Всего реальных объектов: {len(self.true_objects)}\n"
            results_text += f"Найдено объектов: {len(detection['objects'])}\n"

            # Простая оценка точности
            if len(self.true_objects) > 0:
                accuracy = len(detection['objects']) / len(self.true_objects) * 100
                results_text += f"Примерная точность: {accuracy:.1f}%"

        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", results_text)

    def show_message(self, message):
        messagebox.showinfo("Информация", message)

    def show_error(self, message):
        messagebox.showerror("Ошибка", message)

    def run(self):
        print("Запуск основного цикла...")
        self.root.mainloop()


if __name__ == "__main__":
    print("=== ЗАПУСК АНАЛИЗАТОРА РАДАРНЫХ ИЗОБРАЖЕНИЙ ===")
    try:
        app = MainApplication()
        app.run()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        traceback.print_exc()
        input("Нажмите Enter для выхода...")