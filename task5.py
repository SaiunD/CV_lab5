import numpy as np
import matplotlib.pyplot as plt

# Створення сітки
x = np.linspace(-2, 2, 400)  # Діапазон x
y = np.linspace(-2, 2, 400)  # Діапазон y
X, Y = np.meshgrid(x, y)     # Сітка значень

# Формули для поверхонь
z1 = np.sin(X**2 + Y**2)                # Поверхня 1
z2 = np.cos(X) * np.sin(Y)              # Поверхня 2
z3 = np.exp(-(X**2 + Y**2))             # Поверхня 3

# Обчислення часткових похідних за X та Y (градієнти)
def compute_gradients(Z, X, Y):
    # Часткові похідні по X та Y за допомогою чисельних методів (різниця)
    dz_dx = np.gradient(Z, axis=1)  # Похідна за X
    dz_dy = np.gradient(Z, axis=0)  # Похідна за Y
    return dz_dx, dz_dy

# Для кожної поверхні розраховуємо градієнти
dz_dx_1, dz_dy_1 = compute_gradients(z1, X, Y)
dz_dx_2, dz_dy_2 = compute_gradients(z2, X, Y)
dz_dx_3, dz_dy_3 = compute_gradients(z3, X, Y)

# Обчислення величини градієнта (модуль вектора градієнта)
grad_magnitude_1 = np.sqrt(dz_dx_1**2 + dz_dy_1**2)
grad_magnitude_2 = np.sqrt(dz_dx_2**2 + dz_dy_2**2)
grad_magnitude_3 = np.sqrt(dz_dx_3**2 + dz_dy_3**2)

# Функція для обчислення реконструйованої поверхні за допомогою Франкота-Челлапа
def frankot_chellappa(dz_dx, dz_dy):
    # Інтегрування градієнтів за X і Y
    surface_x = np.cumsum(dz_dx, axis=1)  # Інтеграція за X
    surface_y = np.cumsum(dz_dy, axis=0)  # Інтеграція за Y
    surface = surface_x + surface_y
    return surface

# Функція для реконструкції поверхні за допомогою Вея-Клетте
def way_klette(dz_dx, dz_dy, lambda_0=1, lambda_1=1, lambda_2=1):
    # Створення масок для регуляризації
    smoothing_term = lambda_0 * (np.gradient(dz_dx)[0] + np.gradient(dz_dy)[1])
    # Регуляризація
    surface = dz_dx + dz_dy + smoothing_term
    return surface

# Реконструкція поверхні для кожної з градієнтних мап
surface_frankot_chellappa_1 = frankot_chellappa(dz_dx_1, dz_dy_1)
surface_frankot_chellappa_2 = frankot_chellappa(dz_dx_2, dz_dy_2)
surface_frankot_chellappa_3 = frankot_chellappa(dz_dx_3, dz_dy_3)

surface_way_klette_1 = way_klette(dz_dx_1, dz_dy_1)
surface_way_klette_2 = way_klette(dz_dx_2, dz_dy_2)
surface_way_klette_3 = way_klette(dz_dx_3, dz_dy_3)

# Візуалізація реконструйованих поверхонь
fig = plt.figure(figsize=(12, 12))

# Франкот-Челлап (поверхня 1)
ax1 = fig.add_subplot(331, projection='3d')
ax1.plot_surface(X, Y, surface_frankot_chellappa_1, cmap='jet')
ax1.set_title('Frankot-Chellappa Reconstruction for Surface 1')

# Вей-Клетте (поверхня 1)
ax2 = fig.add_subplot(332, projection='3d')
ax2.plot_surface(X, Y, surface_way_klette_1, cmap='jet')
ax2.set_title('Way-Klette Reconstruction for Surface 1')

# Франкот-Челлап (поверхня 2)
ax3 = fig.add_subplot(333, projection='3d')
ax3.plot_surface(X, Y, surface_frankot_chellappa_2, cmap='jet')
ax3.set_title('Frankot-Chellappa Reconstruction for Surface 2')

# Вей-Клетте (поверхня 2)
ax4 = fig.add_subplot(334, projection='3d')
ax4.plot_surface(X, Y, surface_way_klette_2, cmap='jet')
ax4.set_title('Way-Klette Reconstruction for Surface 2')

# Франкот-Челлап (поверхня 3)
ax5 = fig.add_subplot(335, projection='3d')
ax5.plot_surface(X, Y, surface_frankot_chellappa_3, cmap='jet')
ax5.set_title('Frankot-Chellappa Reconstruction for Surface 3')

# Вей-Клетте (поверхня 3)
ax6 = fig.add_subplot(336, projection='3d')
ax6.plot_surface(X, Y, surface_way_klette_3, cmap='jet')
ax6.set_title('Way-Klette Reconstruction for Surface 3')

plt.subplots_adjust(hspace=0.5)

plt.tight_layout()
plt.show()
