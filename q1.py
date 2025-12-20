import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import os
from datetime import datetime
from matplotlib.lines import Line2D

# Configure fonts (prefer DejaVu Sans for broad glyph support)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Create output directory
output_dir = "battlefield_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generate timestamp for filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create figure with two subplots
fig = plt.figure(figsize=(16, 8))

# First subplot: panoramic view
ax1 = fig.add_subplot(121, projection='3d')

# Define coordinate data
fake_target = np.array([0, 0, 0])
real_target = np.array([0, 200, 0])

missiles = {
    'M1': np.array([20000, 0, 2000]),
    'M2': np.array([19000, 600, 2100]),
    'M3': np.array([18000, -600, 1900])
}

drones = {
    'FY1': np.array([17800, 0, 1800]),
    'FY2': np.array([12000, 1400, 1400]),
    'FY3': np.array([6000, -3000, 700]),
    'FY4': np.array([11000, 2000, 1800]),
    'FY5': np.array([13000, -2000, 1300])
}

# Draw cylinder
def draw_cylinder(ax, center, radius, height, color, alpha=0.6):
    circle_bottom = Circle((center[0], center[1]), radius, color=color, alpha=alpha)
    ax.add_patch(circle_bottom)
    art3d.pathpatch_2d_to_3d(circle_bottom, z=center[2], zdir="z")
    
    circle_top = Circle((center[0], center[1]), radius, color=color, alpha=alpha)
    ax.add_patch(circle_top)
    art3d.pathpatch_2d_to_3d(circle_top, z=center[2] + height, zdir="z")
    
    return circle_bottom

# Draw decoy target (red cylinder) - differentiated by height
fake_cylinder = draw_cylinder(ax1, fake_target, 10, 15, 'red', 0.9)
ax1.text(fake_target[0], fake_target[1], fake_target[2] + 20, 'Decoy Target', 
    color='red', fontsize=14, ha='center', fontweight='bold', 
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Draw real target (blue cylinder) - differentiated by height
real_cylinder = draw_cylinder(ax1, real_target, 10, 20, 'blue', 0.9)
ax1.text(real_target[0], real_target[1], real_target[2] + 25, 'Real Target', 
    color='blue', fontsize=14, ha='center', fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Draw missiles (red triangles)
for name, pos in missiles.items():
    ax1.scatter(pos[0], pos[1], pos[2], c='red', marker='^', s=300,
               edgecolors='black', linewidth=2)
    # Adjust label position to be closer to the marker
    label_offset = 500
    ax1.text(pos[0] + label_offset, pos[1] + label_offset, pos[2] + label_offset, name, 
            color='red', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

# Draw drones (green squares)
for name, pos in drones.items():
    ax1.scatter(pos[0], pos[1], pos[2], c='green', marker='s', s=200,
               edgecolors='black', linewidth=1.5)
    # Adjust label position to be closer to the marker
    label_offset = 500
    ax1.text(pos[0] + label_offset, pos[1] + label_offset, pos[2] + label_offset, name, 
            color='green', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

# Draw missile flight direction arrows
for name, pos in missiles.items():
    direction = fake_target - pos
    direction_norm = direction / np.linalg.norm(direction)
    arrow_length = 1000
    
    ax1.quiver(pos[0], pos[1], pos[2], 
              direction_norm[0] * arrow_length,
              direction_norm[1] * arrow_length,
              direction_norm[2] * arrow_length,
              color='orange', arrow_length_ratio=0.25,
              linewidth=3, alpha=0.9)

# Adjust view angle for ax1 and add a short legend text
ax1.view_init(elev=20, azim=-60)
ax1.text2D(0.02, 0.95, 
          "Situation:\n"
          "• Red cylinder: Decoy target\n"
          "• Blue cylinder: Real target\n"
          "• Red triangle: Missile\n"
          "• Green square: Drone\n"
          "• Orange arrow: Flight direction", 
          transform=ax1.transAxes, fontsize=10, verticalalignment='top',
          bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", 
                   alpha=0.8, edgecolor='gold'))

# Second subplot: target area close-up
ax2 = fig.add_subplot(122, projection='3d')

# Draw target area close-up
draw_cylinder(ax2, fake_target, 10, 15, 'red', 0.9)
draw_cylinder(ax2, real_target, 10, 20, 'blue', 0.9)

ax2.text(fake_target[0], fake_target[1], fake_target[2] + 18, 'Decoy Target', 
         color='red', fontsize=14, ha='center', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
ax2.text(real_target[0], real_target[1], real_target[2] + 23, 'Real Target', 
         color='blue', fontsize=14, ha='center', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Set unequal axis ranges for the close-up view
ax2.set_xlim([-50, 250])
ax2.set_ylim([-50, 250])
ax2.set_zlim([-10, 60])

ax2.set_xlabel('X coordinate (m)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Y coordinate (m)', fontsize=12, fontweight='bold')
ax2.set_zlabel('Z coordinate (m)', fontsize=12, fontweight='bold')
ax2.set_title('Target Area Close-up\n(Height difference between decoy and real targets)', fontsize=14, fontweight='bold')

ax2.grid(True, alpha=0.4)
ax2.view_init(elev=35, azim=-45)

# Add distance annotation
ax2.plot([0, 0], [0, 200], [15, 20], 'k--', alpha=0.6)
ax2.text(10, 100, 18, '200m', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

plt.tight_layout()

# Save images
save_path_png = os.path.join(output_dir, f"battlefield_3d_map_{timestamp}.png")
save_path_pdf = os.path.join(output_dir, f"battlefield_3d_map_{timestamp}.pdf")

plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')

print(f"Images saved to directory: {output_dir}")
print(f"PNG file: battlefield_3d_map_{timestamp}.png")
print(f"PDF file: battlefield_3d_map_{timestamp}.pdf")

# Show (may be non-interactive in headless backends)
plt.show()

# Save info
print(f"\nFiles saved!")
print(f"You can find generated images in folder '{output_dir}'")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define constants
g = 9.8
v_missile = 300
v_smoke_sink = 3
effective_radius = 10
effective_duration = 20

# Target positions
fake_target = np.array([0, 0, 0])
real_target = np.array([0, 200, 0])

# Missile M1 start position
M1_start = np.array([20000, 0, 2000])

# Drone FY1 start position
FY1_start = np.array([17800, 0, 1800])

# Drone speed
v_drone = 120
direction_to_target = fake_target[:2] - FY1_start[:2]
direction_to_target = direction_to_target / np.linalg.norm(direction_to_target)
v_drone_vector = np.array([v_drone * direction_to_target[0], 
                          v_drone * direction_to_target[1], 0])

print(f"Drone velocity vector: {v_drone_vector}")

# Time parameters
t_drop = 1.5
t_blast_delay = 3.6
t_blast = t_drop + t_blast_delay

# Compute positions
drop_position = FY1_start + v_drone_vector * t_drop
vertical_drop = 0.5 * g * t_blast_delay**2
blast_position = np.array([drop_position[0]+v_drone_vector[0]*t_blast_delay, drop_position[1], drop_position[2] - vertical_drop])

print(f"Drop position: {drop_position}")
print(f"Blast position: {blast_position}")

# Missile flight direction (towards decoy target)
missile_direction = fake_target - M1_start
missile_direction = missile_direction / np.linalg.norm(missile_direction)
v_missile_vector = v_missile * missile_direction
print(missile_direction, v_missile_vector)
print(f"Missile velocity vector: {v_missile_vector}")
print(f"Missile start position: {M1_start}")

# Calculate time for missile to reach decoy target
def time_to_target(position, velocity, target):
    t_x = (target[0] - position[0]) / velocity[0]
    return t_x

t_missile_to_target = time_to_target(M1_start, v_missile_vector, fake_target)
print(f"Missile time to decoy target: {t_missile_to_target:.2f} s")

# Compute missile position at time t
def missile_position(t):
    return M1_start + v_missile_vector * t

# Compute smoke cloud position at time t (since blast)
def smoke_position(t_smoke):
    return np.array([blast_position[0], blast_position[1], blast_position[2] - v_smoke_sink * t_smoke])

# Angle condition: determine if smoke is between missile and real target
def is_smoke_between(missile_pos, smoke_pos, target_pos):
    """
    Determine whether the smoke is between the missile and the target.
    If angle missile->smoke->target is acute, smoke lies between them.
    If the angle is obtuse, smoke is behind the target (invalid).
    """
    # Vector: missile to smoke
    missile_to_smoke = smoke_pos - missile_pos
    # Vector: missile to target
    missile_to_target = target_pos - missile_pos

    # Compute cosine of the angle
    dot_product = np.dot(missile_to_smoke, missile_to_target)
    cos_angle = dot_product / (np.linalg.norm(missile_to_smoke) * np.linalg.norm(missile_to_target))

    # If cos_angle > 0 the angle is acute and smoke lies between missile and target
    # If cos_angle < 0 the smoke is behind the target (not effective)
    return cos_angle > 0

# Distance from a point to a line
def distance_point_to_line(point, line_point, line_direction):
    ap = point - line_point
    projection = np.dot(ap, line_direction) / np.linalg.norm(line_direction)
    foot_point = line_point + projection * line_direction
    return np.linalg.norm(point - foot_point)

# Determine whether the missile crosses the smoke cloud
def is_missile_through_smoke(missile_pos, smoke_pos, missile_prev_pos, smoke_prev_pos):
    """
    Determine whether the missile trajectory segment intersects the smoke sphere.
    Uses relative motion (missile minus smoke) to simplify the intersection test.
    """
    # Smoke cloud radius
    radius = effective_radius

    # Missile movement vector
    missile_move = missile_pos - missile_prev_pos
    missile_move_length = np.linalg.norm(missile_move)

    if missile_move_length == 0:
        return False

    # Smoke movement vector
    smoke_move = smoke_pos - smoke_prev_pos

    # Use relative motion to simplify calculation
    relative_move = missile_move - smoke_move
    relative_move_length = np.linalg.norm(relative_move)

    if relative_move_length == 0:
        # Relative stationary: check current distance
        return np.linalg.norm(missile_pos - smoke_pos) <= radius

    # Compute missile motion relative to smoke
    relative_direction = relative_move / relative_move_length

    # Vector from smoke to missile (at previous time)
    smoke_to_missile = missile_prev_pos - smoke_prev_pos

    # Compute projection length
    projection = np.dot(smoke_to_missile, relative_direction)

    # Compute perpendicular distance to smoke
    perpendicular_dist = np.linalg.norm(smoke_to_missile - projection * relative_direction)

    # If perpendicular distance <= radius and projection within movement range, intersection possible
    if perpendicular_dist <= radius and 0 <= projection <= relative_move_length:
        return True

    # Check whether start or end points are inside the cloud
    if np.linalg.norm(missile_prev_pos - smoke_prev_pos) <= radius:
        return True
    if np.linalg.norm(missile_pos - smoke_pos) <= radius:
        return True

    return False

# Calculate effective concealment time
def calculate_effective_time():
    start_time = 0
    end_time = min(effective_duration, t_missile_to_target - t_blast)
    
    if end_time <= 0:
        return 0.0
    
    time_step = 0.01  # Increase time step slightly for performance
    total_effective_time = 0.0
    current_time = start_time
    
    # Store previous positions for crossing checks
    prev_missile_pos = missile_position(t_blast)
    prev_smoke_pos = smoke_position(0)
    
    while current_time <= end_time:
        t_missile = t_blast + current_time
        if t_missile > t_missile_to_target:
            break
            
        pos_missile = missile_position(t_missile)
        pos_smoke = smoke_position(current_time)
        
        # Compute distance
        missile_to_target_direction = real_target - pos_missile
        missile_to_target_direction = missile_to_target_direction / np.linalg.norm(missile_to_target_direction)
        distance = distance_point_to_line(pos_smoke, pos_missile, missile_to_target_direction)
        # Check angle condition: smoke must be between missile and target
        is_between = is_smoke_between(pos_missile, pos_smoke, real_target)

        # Check whether missile crosses the smoke cloud
        is_through = is_missile_through_smoke(pos_missile, pos_smoke, prev_missile_pos, prev_smoke_pos)

        # Three cases count as effective concealment:
        # 1. distance <= effective_radius and smoke between missile and target (standard concealment)
        # 2. missile crosses the smoke cloud (new condition)
        # 3. missile is directly inside the cloud
        in_smoke = np.linalg.norm(pos_missile - pos_smoke) <= effective_radius
        
        if (distance <= effective_radius and is_between) or is_through or in_smoke:
            total_effective_time += time_step
        
        # Update previous positions
        prev_missile_pos = pos_missile
        prev_smoke_pos = pos_smoke
        
        current_time += time_step
    
    return total_effective_time

# Compute and output results
effective_time = calculate_effective_time()
print(f"\n=== RESULTS ===")
print(f"Smoke drop time: {t_drop:.1f} s")
print(f"Smoke blast time: {t_blast:.1f} s")
print(f"Blast position: ({blast_position[0]:.1f}, {blast_position[1]:.1f}, {blast_position[2]:.1f})")
print(f"Missile time to decoy target: {t_missile_to_target:.2f} s")
print(f"Effective concealment duration: {effective_time:.3f} s")

# Detailed analysis of key time points
print(f"\n=== Key timepoint detailed analysis ===")
for t_check in [0,1,2,3]:
    if t_check <= min(effective_duration, t_missile_to_target - t_blast):
        t_missile_check = t_blast + t_check
        pos_missile = missile_position(t_missile_check)
        pos_smoke = smoke_position(t_check)
        
        # Compute distance
        missile_to_target_direction = real_target - pos_missile
        missile_to_target_direction = missile_to_target_direction / np.linalg.norm(missile_to_target_direction)
        distance = distance_point_to_line(pos_smoke, pos_missile, missile_to_target_direction)
        
        # Check angle condition
        is_between = is_smoke_between(pos_missile, pos_smoke, real_target)
        
        # Compute direct distance
        direct_distance = np.linalg.norm(pos_missile - pos_smoke)
        
        # Compute angle
        missile_to_smoke = pos_smoke - pos_missile
        missile_to_target = real_target - pos_missile
        dot_product = np.dot(missile_to_smoke, missile_to_target)
        angle_deg = np.degrees(np.arccos(dot_product / (np.linalg.norm(missile_to_smoke) * np.linalg.norm(missile_to_target))))
        
        print(f"{t_check}s after blast:")
        print(f"  Missile position: {pos_missile}")
        print(f"  Smoke position: {pos_smoke}")
        print(f"  Distance to LOS: {distance:.2f} m")
        print(f"  Direct distance: {direct_distance:.2f} m")
        print(f"  Smoke between missile and target: {'Yes' if is_between else 'No'}")
        print(f"  Missile-Smoke-Target angle: {angle_deg:.1f}°")
        print(f"  Inside cloud: {'Yes' if direct_distance <= effective_radius else 'No'}")
        print(f"  Effective concealment: {'Yes' if (distance <= effective_radius and is_between) or direct_distance <= effective_radius else 'No'}")

# Visualization
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectories
missile_times = np.linspace(0, t_missile_to_target, 100)
missile_traj = np.array([missile_position(t) for t in missile_times])
ax.plot(missile_traj[:, 0], missile_traj[:, 1], missile_traj[:, 2], 'r-', label='Missile trajectory', linewidth=2)

# Plot smoke descent trajectory
smoke_times = np.linspace(0, min(effective_duration, t_missile_to_target - t_blast), 50)
smoke_traj = np.array([smoke_position(t) for t in smoke_times])
ax.plot(smoke_traj[:, 0], smoke_traj[:, 1], smoke_traj[:, 2], 'b-', label='Smoke descent trajectory', linewidth=2)

# Mark key points
ax.scatter(*fake_target, color='red', s=200, label='Decoy Target', marker='x')
ax.scatter(*real_target, color='blue', s=200, label='Real Target', marker='o')
ax.scatter(*M1_start, color='orange', s=100, label='Missile start')
ax.scatter(*blast_position, color='green', s=100, label='Blast point')

# Draw smoke cloud at key times
for t in [0, 10, 20]:
    if t < len(smoke_times):
        smoke_pos = smoke_position(t)
        # Draw sphere to represent smoke cloud
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = smoke_pos[0] + effective_radius * np.outer(np.cos(u), np.sin(v))
        y = smoke_pos[1] + effective_radius * np.outer(np.sin(u), np.sin(v))
        z = smoke_pos[2] + effective_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='cyan', alpha=0.2)

# Set view and labels
ax.view_init(elev=20, azim=45)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Smoke Screen Interference Analysis for Missile M1 (including crossing detection)')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()
