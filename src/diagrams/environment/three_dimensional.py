"""
Two-Ray Ground Reflection Model - 3D Visualization
For Thesis Figure 3.3
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform


class Arrow3D(FancyArrowPatch):
    """Custom 3D arrow for matplotlib"""

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


# Create figure
fig = plt.figure(figsize=(14, 10))

# ========================================
# SUBPLOT 1: 3D Geometry
# ========================================
ax1 = fig.add_subplot(221, projection='3d')

# Sensor position (origin)
sensor_x, sensor_y, sensor_z = 0, 0, 0

# UAV position (elevated)
uav_x, uav_y, uav_z = 30, 20, 50

# Plot sensor
ax1.scatter([sensor_x], [sensor_y], [sensor_z],
            c='blue', marker='o', s=300,
            edgecolors='black', linewidths=2,
            label='IoT Sensor', zorder=10)

# Plot UAV
ax1.scatter([uav_x], [uav_y], [uav_z],
            c='orange', marker='^', s=400,
            edgecolors='red', linewidths=2,
            label='UAV', zorder=10)

# Direct path (Line-of-Sight)
ax1.plot([sensor_x, uav_x],
         [sensor_y, uav_y],
         [sensor_z, uav_z],
         'g--', linewidth=2.5, label='Direct Path (LOS)', zorder=5)

# Ground-reflected path
# Reflection point (midpoint on ground)
reflect_x = (sensor_x + uav_x) / 2
reflect_y = (sensor_y + uav_y) / 2
reflect_z = 0

ax1.plot([sensor_x, reflect_x],
         [sensor_y, reflect_y],
         [sensor_z, reflect_z],
         'r:', linewidth=2, label='Reflected Path', zorder=5)
ax1.plot([reflect_x, uav_x],
         [reflect_y, uav_y],
         [reflect_z, uav_z],
         'r:', linewidth=2, zorder=5)

# Vertical line from UAV to ground (altitude)
ax1.plot([uav_x, uav_x],
         [uav_y, uav_y],
         [0, uav_z],
         'k--', linewidth=1.5, alpha=0.5, zorder=3)

# Horizontal distance line
ax1.plot([sensor_x, uav_x],
         [sensor_y, uav_y],
         [0, 0],
         'b--', linewidth=1.5, alpha=0.5, zorder=3)

# Ground plane (semi-transparent)
xx, yy = np.meshgrid(np.linspace(-5, 40, 10),
                     np.linspace(-5, 30, 10))
zz = np.zeros_like(xx)
ax1.plot_surface(xx, yy, zz, alpha=0.1, color='gray')

# Labels and annotations
ax1.text(sensor_x, sensor_y, sensor_z - 5, 'Sensor\n(x₁, y₁, 0)',
         fontsize=10, ha='center', fontweight='bold')
ax1.text(uav_x, uav_y, uav_z + 5, 'UAV\n(x₂, y₂, h)',
         fontsize=10, ha='center', fontweight='bold')
ax1.text(uav_x + 2, uav_y, uav_z / 2, f'h = {uav_z}m',
         fontsize=9, color='black', fontweight='bold')

# Calculate distances
d_h = np.sqrt((uav_x - sensor_x) ** 2 + (uav_y - sensor_y) ** 2)
d_3d = np.sqrt(d_h ** 2 + uav_z ** 2)

ax1.text((sensor_x + uav_x) / 2, (sensor_y + uav_y) / 2, uav_z / 2 + 5,
         f'd₃D = {d_3d:.1f}m',
         fontsize=9, color='green', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Axis labels
ax1.set_xlabel('X (meters)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Y (meters)', fontsize=11, fontweight='bold')
ax1.set_zlabel('Altitude (meters)', fontsize=11, fontweight='bold')
ax1.set_title('3D Propagation Geometry', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.set_box_aspect([1, 1, 1])

# ========================================
# SUBPLOT 2: Side View (2D Cross-Section)
# ========================================
ax2 = fig.add_subplot(222)

# Horizontal distance
d_horizontal = 50

# UAV
ax2.scatter([d_horizontal / 2], [50], c='orange', marker='^',
            s=400, edgecolors='red', linewidths=2, zorder=10)
ax2.text(d_horizontal / 2, 55, 'UAV', ha='center', fontsize=11, fontweight='bold')

# Sensor
ax2.scatter([0], [0], c='blue', marker='o', s=300,
            edgecolors='black', linewidths=2, zorder=10)
ax2.text(0, -5, 'Sensor', ha='center', fontsize=11, fontweight='bold')

# Direct path
ax2.plot([0, d_horizontal / 2], [0, 50], 'g--', linewidth=2.5, label='Direct LOS')

# Reflected path (via ground)
ax2.plot([0, d_horizontal / 2], [0, 0], 'r:', linewidth=2, label='Ground Reflection')
ax2.plot([d_horizontal / 2, d_horizontal / 2], [0, 50], 'r:', linewidth=2)

# Altitude line
ax2.plot([d_horizontal / 2, d_horizontal / 2], [0, 50], 'k--', linewidth=1.5, alpha=0.5)
ax2.text(d_horizontal / 2 + 3, 25, 'h = 50m', fontsize=10, rotation=90, va='center')

# Horizontal distance line
ax2.plot([0, d_horizontal / 2], [0, 0], 'b--', linewidth=1.5, alpha=0.5)
ax2.text(d_horizontal / 4, -3, f'dₕ', fontsize=10, ha='center')

# Ground line
ax2.axhline(0, color='brown', linewidth=3, alpha=0.3, label='Ground')

# Grid
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('Horizontal Distance (meters)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Altitude (meters)', fontsize=11, fontweight='bold')
ax2.set_title('Side View: Two-Ray Propagation', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.set_xlim(-5, 60)
ax2.set_ylim(-10, 60)
ax2.set_aspect('equal')

# ========================================
# SUBPLOT 3: Top View (Communication Footprint)
# ========================================
ax3 = fig.add_subplot(223)

# Grid
grid_size = 500
ax3.set_xlim(0, grid_size)
ax3.set_ylim(0, grid_size)

# UAV position (center)
uav_pos = (250, 250)
ax3.scatter([uav_pos[0]], [uav_pos[1]], c='orange', marker='^',
            s=400, edgecolors='red', linewidths=2, zorder=10)
ax3.text(uav_pos[0], uav_pos[1] + 20, 'UAV', ha='center',
         fontsize=11, fontweight='bold')

# Communication range circle
comm_range = 100  # meters
circle = plt.Circle(uav_pos, comm_range, color='green',
                    fill=False, linewidth=2, linestyle='--',
                    label=f'Range ({comm_range}m)', zorder=5)
ax3.add_patch(circle)

# Sensors (random positions)
np.random.seed(42)
sensor_positions = np.random.uniform(50, 450, (20, 2))

for i, (sx, sy) in enumerate(sensor_positions):
    distance = np.sqrt((sx - uav_pos[0]) ** 2 + (sy - uav_pos[1]) ** 2)

    if distance <= comm_range:
        color = 'green'
        marker = 'o'
        alpha = 0.8
    else:
        color = 'gray'
        marker = 'o'
        alpha = 0.4

    ax3.scatter([sx], [sy], c=color, marker=marker,
                s=150, edgecolors='black', linewidths=1.5,
                alpha=alpha, zorder=8)

# Grid lines
for i in range(0, grid_size + 1, 50):
    ax3.axhline(i, color='gray', linewidth=0.5, alpha=0.3)
    ax3.axvline(i, color='gray', linewidth=0.5, alpha=0.3)

ax3.set_xlabel('X (meters)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Y (meters)', fontsize=11, fontweight='bold')
ax3.set_title('Top View: Communication Footprint', fontsize=13, fontweight='bold')
ax3.set_aspect('equal')
ax3.legend(loc='upper right', fontsize=9)
# ========================================
# Overall title
# ========================================
fig.suptitle('Two-Ray Ground Reflection Model for UAV-IoT Communication\n'
             'Figure 3.3: Propagation Geometry and Channel Characteristics',
             fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save high-resolution figure
plt.savefig('two_ray_model_3d.png', dpi=300, bbox_inches='tight')
plt.savefig('two_ray_model_3d.pdf', dpi=300, bbox_inches='tight')  # For LaTeX

print("✓ Figure saved as:")
print(" - two_ray_model_3d.png (for presentations)")
print(" - two_ray_model_3d.pdf (for thesis)")

plt.show()
