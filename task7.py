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

# Обчислення градієнтів
def compute_gradients(Z):
    dz_dx = np.gradient(Z, axis=1)  # Похідна за X
    dz_dy = np.gradient(Z, axis=0)  # Похідна за Y
    return dz_dx, dz_dy

dz_dx_1, dz_dy_1 = compute_gradients(z1)
dz_dx_2, dz_dy_2 = compute_gradients(z2)
dz_dx_3, dz_dy_3 = compute_gradients(z3)

# Додавання шуму
def add_noise(grad, noise_level=0.01):
    noise = noise_level * np.random.randn(*grad.shape)
    return grad + noise

dz_dx_1_noisy = add_noise(dz_dx_1)
dz_dy_1_noisy = add_noise(dz_dy_1)
dz_dx_2_noisy = add_noise(dz_dx_2)
dz_dy_2_noisy = add_noise(dz_dy_2)
dz_dx_3_noisy = add_noise(dz_dx_3)
dz_dy_3_noisy = add_noise(dz_dy_3)

# Франкота-Челлаппа
def frankot_chellappa(dz_dx, dz_dy):
    fz_x = np.fft.fft2(dz_dx)
    fz_y = np.fft.fft2(dz_dy)
    kx = np.fft.fftfreq(dz_dx.shape[1]).reshape(1, -1)
    ky = np.fft.fftfreq(dz_dx.shape[0]).reshape(-1, 1)
    kz = np.sqrt(kx**2 + ky**2)
    kz[0, 0] = 1  # Уникнення ділення на нуль
    fz = (1j * kx * fz_x + 1j * ky * fz_y) / (2 * np.pi * kz)
    fz[0, 0] = 0  # Встановлення DC-компоненти
    return np.real(np.fft.ifft2(fz))

# Вей-Клетте
def vey_klette(dz_dx, dz_dy, lambda_0=1, lambda_1=1, lambda_2=0.1):
    fx = dz_dx - np.roll(dz_dx, 1, axis=1)
    fy = dz_dy - np.roll(dz_dy, 1, axis=0)
    return lambda_0 * dz_dx + lambda_1 * dz_dy + lambda_2 * (fx + fy)

# Реконструкція поверхонь
reconstructed_z1_fc = frankot_chellappa(dz_dx_1_noisy, dz_dy_1_noisy)
reconstructed_z1_vk = vey_klette(dz_dx_1_noisy, dz_dy_1_noisy)
reconstructed_z2_fc = frankot_chellappa(dz_dx_2_noisy, dz_dy_2_noisy)
reconstructed_z2_vk = vey_klette(dz_dx_2_noisy, dz_dy_2_noisy)
reconstructed_z3_fc = frankot_chellappa(dz_dx_3_noisy, dz_dy_3_noisy)
reconstructed_z3_vk = vey_klette(dz_dx_3_noisy, dz_dy_3_noisy)

# Візуалізація
fig = plt.figure(figsize=(18, 18))  # Збільшуємо розмір фігури

# Поверхня 1 (Франкот-Челлаппа і Вей-Клетте)
ax1 = fig.add_subplot(331, projection='3d')
ax1.plot_surface(X, Y, reconstructed_z1_fc, cmap='viridis')
ax1.set_title('Surface 1 (Frankot-Chelappa)', fontsize=8)

ax2 = fig.add_subplot(332, projection='3d')
ax2.plot_surface(X, Y, reconstructed_z1_vk, cmap='viridis')
ax2.set_title('Surface 1 (Vey-Klette)', fontsize=8)

# Поверхня 2 (Франкот-Челлаппа і Вей-Клетте)
ax3 = fig.add_subplot(333, projection='3d')
ax3.plot_surface(X, Y, reconstructed_z2_fc, cmap='viridis')
ax3.set_title('Surface 2 (Frankot-Chelappa)', fontsize=8)

ax4 = fig.add_subplot(334, projection='3d')
ax4.plot_surface(X, Y, reconstructed_z2_vk, cmap='viridis')
ax4.set_title('Surface 2 (Vey-Klette)', fontsize=8)

# Поверхня 3 (Франкот-Челлаппа і Вей-Клетте)
ax5 = fig.add_subplot(335, projection='3d')
ax5.plot_surface(X, Y, reconstructed_z3_fc, cmap='viridis')
ax5.set_title('Surface 3 (Frankot-Chelappa)', fontsize=8)

ax6 = fig.add_subplot(336, projection='3d')
ax6.plot_surface(X, Y, reconstructed_z3_vk, cmap='viridis')
ax6.set_title('Surface 3 (Vey-Klette)', fontsize=8)

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()
