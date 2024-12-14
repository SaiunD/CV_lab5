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

# Камера (параметри)
focal_length = 1  # Фокусна відстань
camera_position = np.array([0, 0, 5])  # Позиція камери
z_plane = 1  # Задній план для глибини

# Функція для обчислення глибини (відстані від камери до точки)
def calculate_depth(X, Y, Z, camera_position):
    return np.sqrt((X - camera_position[0])**2 + (Y - camera_position[1])**2 + (Z - camera_position[2])**2)

# Функція для створення мапи висоти
def create_height_map(Z):
    return Z

# Функція для створення мапи відстані
def create_distance_map(X, Y, Z, camera_position):
    return calculate_depth(X, Y, Z, camera_position)

# Функція для створення мапи глибини
def create_depth_map(Z, z_plane):
    return np.abs(Z - z_plane)

# Обчислення мап
depth_map_1 = create_depth_map(z1, z_plane)
distance_map_1 = create_distance_map(X, Y, z1, camera_position)
height_map_1 = create_height_map(z1)

depth_map_2 = create_depth_map(z2, z_plane)
distance_map_2 = create_distance_map(X, Y, z2, camera_position)
height_map_2 = create_height_map(z2)

depth_map_3 = create_depth_map(z3, z_plane)
distance_map_3 = create_distance_map(X, Y, z3, camera_position)
height_map_3 = create_height_map(z3)

# Візуалізація результатів
fig = plt.figure(figsize=(12, 12))

# Мапа глибини
ax1 = fig.add_subplot(331)
ax1.imshow(depth_map_1, cmap='gray')
ax1.set_title('Depth Map for Surface 1')

ax2 = fig.add_subplot(332)
ax2.imshow(depth_map_2, cmap='gray')
ax2.set_title('Depth Map for Surface 2')

ax3 = fig.add_subplot(333)
ax3.imshow(depth_map_3, cmap='gray')
ax3.set_title('Depth Map for Surface 3')

# Мапа відстані
ax4 = fig.add_subplot(334)
ax4.imshow(distance_map_1, cmap='gray')
ax4.set_title('Distance Map for Surface 1')

ax5 = fig.add_subplot(335)
ax5.imshow(distance_map_2, cmap='gray')
ax5.set_title('Distance Map for Surface 2')

ax6 = fig.add_subplot(336)
ax6.imshow(distance_map_3, cmap='gray')
ax6.set_title('Distance Map for Surface 3')

# Мапа висоти
ax7 = fig.add_subplot(337)
ax7.imshow(height_map_1, cmap='gray')
ax7.set_title('Height Map for Surface 1')

ax8 = fig.add_subplot(338)
ax8.imshow(height_map_2, cmap='gray')
ax8.set_title('Height Map for Surface 2')

ax9 = fig.add_subplot(339)
ax9.imshow(height_map_3, cmap='gray')
ax9.set_title('Height Map for Surface 3')

plt.tight_layout()
plt.show()
