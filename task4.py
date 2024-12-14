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
    dz_dx = np.gradient(Z, axis=1)  # Похідна за X
    dz_dy = np.gradient(Z, axis=0)  # Похідна за Y
    return dz_dx, dz_dy


# Для кожної поверхні розраховуємо градієнти
dz_dx_1, dz_dy_1 = compute_gradients(z1, X, Y)
dz_dx_2, dz_dy_2 = compute_gradients(z2, X, Y)
dz_dx_3, dz_dy_3 = compute_gradients(z3, X, Y)


# Функція для інтеграції градієнтів для відновлення поверхні
def reconstruct_surface(dz_dx, dz_dy, X, Y):
    # Початкові умови: нульова поверхня
    Z = np.zeros_like(X)

    # Інтеграція градієнтів по осі X і Y
    Z += np.cumsum(dz_dx, axis=1)  # Інтеграція за X
    Z += np.cumsum(dz_dy, axis=0)  # Інтеграція за Y
    Z = Z - np.min(Z)  # Зміщення, щоб поверхня була відносно 0
    return Z


# Реконструйовані поверхні
z_reconstructed_1 = reconstruct_surface(dz_dx_1, dz_dy_1, X, Y)
z_reconstructed_2 = reconstruct_surface(dz_dx_2, dz_dy_2, X, Y)
z_reconstructed_3 = reconstruct_surface(dz_dx_3, dz_dy_3, X, Y)

# Візуалізація реконструйованих поверхонь
fig = plt.figure(figsize=(12, 8))

# Візуалізація реконструйованої поверхні 1
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, z_reconstructed_1, cmap='viridis')
ax1.set_title('Reconstructed Surface 1: sin(x^2 + y^2)')

# Візуалізація реконструйованої поверхні 2
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, z_reconstructed_2, cmap='plasma')
ax2.set_title('Reconstructed Surface 2: cos(x) * sin(y)')

# Візуалізація реконструйованої поверхні 3
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, z_reconstructed_3, cmap='inferno')
ax3.set_title('Reconstructed Surface 3: exp(-x^2 - y^2)')

# Показати графіки
plt.tight_layout()
plt.show()
