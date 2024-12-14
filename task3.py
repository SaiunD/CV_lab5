import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Візуалізація мап градієнтів
fig = plt.figure(figsize=(12, 12))

# Мапа градієнтів по X (поверхня 1)
ax1 = fig.add_subplot(331)
ax1.imshow(dz_dx_1, cmap='jet')
ax1.set_title('Gradient in X for Surface 1')

# Мапа градієнтів по Y (поверхня 1)
ax2 = fig.add_subplot(332)
ax2.imshow(dz_dy_1, cmap='jet')
ax2.set_title('Gradient in Y for Surface 1')

# Мапа величини градієнта (поверхня 1)
ax3 = fig.add_subplot(333)
ax3.imshow(grad_magnitude_1, cmap='jet')
ax3.set_title('Gradient Magnitude for Surface 1')

# Мапа градієнтів по X (поверхня 2)
ax4 = fig.add_subplot(334)
ax4.imshow(dz_dx_2, cmap='jet')
ax4.set_title('Gradient in X for Surface 2')

# Мапа градієнтів по Y (поверхня 2)
ax5 = fig.add_subplot(335)
ax5.imshow(dz_dy_2, cmap='jet')
ax5.set_title('Gradient in Y for Surface 2')

# Мапа величини градієнта (поверхня 2)
ax6 = fig.add_subplot(336)
ax6.imshow(grad_magnitude_2, cmap='jet')
ax6.set_title('Gradient Magnitude for Surface 2')

# Мапа градієнтів по X (поверхня 3)
ax7 = fig.add_subplot(337)
ax7.imshow(dz_dx_3, cmap='jet')
ax7.set_title('Gradient in X for Surface 3')

# Мапа градієнтів по Y (поверхня 3)
ax8 = fig.add_subplot(338)
ax8.imshow(dz_dy_3, cmap='jet')
ax8.set_title('Gradient in Y for Surface 3')

# Мапа величини градієнта (поверхня 3)
ax9 = fig.add_subplot(339)
ax9.imshow(grad_magnitude_3, cmap='jet')
ax9.set_title('Gradient Magnitude for Surface 3')

plt.tight_layout()
plt.show()
