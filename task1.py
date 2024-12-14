import cv2
import numpy as np

# Створення сітки
x = np.linspace(-2, 2, 400)  # Діапазон x
y = np.linspace(-2, 2, 400)  # Діапазон y
X, Y = np.meshgrid(x, y)     # Сітка значень

# Формули для поверхонь
z1 = np.sin(X**2 + Y**2)                # Поверхня 1
z2 = np.cos(X) * np.sin(Y)              # Поверхня 2
z3 = np.exp(-(X**2 + Y**2))             # Поверхня 3

# Нормалізація для візуалізації (0-255)
def normalize_surface(z):
    z_min, z_max = z.min(), z.max()
    return ((z - z_min) / (z_max - z_min) * 255).astype(np.uint8)

z1_norm = normalize_surface(z1)
z2_norm = normalize_surface(z2)
z3_norm = normalize_surface(z3)

# Розмір тексту та кольори
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5  # Зменшений розмір тексту
font_thickness = 1
text_color = (255, 255, 255)

# Додавання тексту в окремі області знизу
height, width = z1_norm.shape
text_height = 50  # Місце для тексту

# Створення порожнього фону для тексту
z1_with_text = np.zeros((height + text_height, width), dtype=np.uint8)
z2_with_text = np.zeros((height + text_height, width), dtype=np.uint8)
z3_with_text = np.zeros((height + text_height, width), dtype=np.uint8)

# Копіюємо зображення поверхні
z1_with_text[:height, :] = z1_norm
z2_with_text[:height, :] = z2_norm
z3_with_text[:height, :] = z3_norm

# Додаємо текст під поверхнями
cv2.putText(z1_with_text, 'Surface 1: sin(x^2 + y^2)', (10, height + 30), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
cv2.putText(z2_with_text, 'Surface 2: cos(x) * sin(y)', (10, height + 30), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
cv2.putText(z3_with_text, 'Surface 3: exp(-x^2 - y^2)', (10, height + 30), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

# Об'єднуємо поверхні з текстом в одне зображення
combined_image = np.hstack([
    cv2.resize(z1_with_text, (400, height + text_height)),
    cv2.resize(z2_with_text, (400, height + text_height)),
    cv2.resize(z3_with_text, (400, height + text_height))
])

# Відображення комбінованого зображення
cv2.imshow("Combined Surfaces", combined_image)

# Очікування натискання клавіші та закриття вікон
cv2.waitKey(0)
cv2.destroyAllWindows()
