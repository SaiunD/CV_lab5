import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Створення сітки
x = np.linspace(-2, 2, 400)  # Діапазон x
y = np.linspace(-2, 2, 400)  # Діапазон y
X, Y = np.meshgrid(x, y)  # Сітка значень

# Формули для поверхонь
z1 = np.sin(X ** 2 + Y ** 2)  # Поверхня 1
z2 = np.cos(X) * np.sin(Y)  # Поверхня 2
z3 = np.exp(-(X ** 2 + Y ** 2))  # Поверхня 3


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

# Додавання випадкового шуму до градієнтів
noise_level = 0.1  # Рівень шуму
dz_dx_1 += noise_level * np.random.randn(*dz_dx_1.shape)
dz_dy_1 += noise_level * np.random.randn(*dz_dy_1.shape)

dz_dx_2 += noise_level * np.random.randn(*dz_dx_2.shape)
dz_dy_2 += noise_level * np.random.randn(*dz_dy_2.shape)

dz_dx_3 += noise_level * np.random.randn(*dz_dx_3.shape)
dz_dy_3 += noise_level * np.random.randn(*dz_dy_3.shape)

# Візуалізація мап градієнтів з шумом
fig = plt.figure(figsize=(12, 12))

# Мапа градієнтів по X (поверхня 1)
ax1 = fig.add_subplot(331)
ax1.imshow(dz_dx_1, cmap='jet')
ax1.set_title('Noisy Gradient in X for Surface 1')

# Мапа градієнтів по Y (поверхня 1)
ax2 = fig.add_subplot(332)
ax2.imshow(dz_dy_1, cmap='jet')
ax2.set_title('Noisy Gradient in Y for Surface 1')

# Мапа величини градієнта (поверхня 1)
grad_magnitude_1 = np.sqrt(dz_dx_1 ** 2 + dz_dy_1 ** 2)
ax3 = fig.add_subplot(333)
ax3.imshow(grad_magnitude_1, cmap='jet')
ax3.set_title('Noisy Gradient Magnitude for Surface 1')

# Мапа градієнтів по X (поверхня 2)
ax4 = fig.add_subplot(334)
ax4.imshow(dz_dx_2, cmap='jet')
ax4.set_title('Noisy Gradient in X for Surface 2')

# Мапа градієнтів по Y (поверхня 2)
ax5 = fig.add_subplot(335)
ax5.imshow(dz_dy_2, cmap='jet')
ax5.set_title('Noisy Gradient in Y for Surface 2')

# Мапа величини градієнта (поверхня 2)
grad_magnitude_2 = np.sqrt(dz_dx_2 ** 2 + dz_dy_2 ** 2)
ax6 = fig.add_subplot(336)
ax6.imshow(grad_magnitude_2, cmap='jet')
ax6.set_title('Noisy Gradient Magnitude for Surface 2')

# Мапа градієнтів по X (поверхня 3)
ax7 = fig.add_subplot(337)
ax7.imshow(dz_dx_3, cmap='jet')
ax7.set_title('Noisy Gradient in X for Surface 3')

# Мапа градієнтів по Y (поверхня 3)
ax8 = fig.add_subplot(338)
ax8.imshow(dz_dy_3, cmap='jet')
ax8.set_title('Noisy Gradient in Y for Surface 3')

# Мапа величини градієнта (поверхня 3)
grad_magnitude_3 = np.sqrt(dz_dx_3 ** 2 + dz_dy_3 ** 2)
ax9 = fig.add_subplot(339)
ax9.imshow(grad_magnitude_3, cmap='jet')
ax9.set_title('Noisy Gradient Magnitude for Surface 3')

plt.tight_layout()
plt.show()


# Функція для реконструкції поверхні за допомогою двопрохідного методу
def reconstruct_surface(dz_dx, dz_dy, X, Y):
    # Перший прохід (по X)
    Z_x = np.cumsum(dz_dx, axis=1)

    # Другий прохід (по Y)
    Z = np.cumsum(dz_dy, axis=0) + Z_x

    return Z


# Реконструкція поверхонь
reconstructed_z1 = reconstruct_surface(dz_dx_1, dz_dy_1, X, Y)
reconstructed_z2 = reconstruct_surface(dz_dx_2, dz_dy_2, X, Y)
reconstructed_z3 = reconstruct_surface(dz_dx_3, dz_dy_3, X, Y)

# Візуалізація реконструйованих поверхонь
fig = plt.figure(figsize=(12, 12))

# Поверхня 1
ax1 = fig.add_subplot(331, projection='3d')
ax1.plot_surface(X, Y, reconstructed_z1, cmap='viridis')
ax1.set_title('Reconstructed Surface 1')

# Поверхня 2
ax2 = fig.add_subplot(332, projection='3d')
ax2.plot_surface(X, Y, reconstructed_z2, cmap='viridis')
ax2.set_title('Reconstructed Surface 2')

# Поверхня 3
ax3 = fig.add_subplot(333, projection='3d')
ax3.plot_surface(X, Y, reconstructed_z3, cmap='viridis')
ax3.set_title('Reconstructed Surface 3')

plt.tight_layout()
plt.show()
