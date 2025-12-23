import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_smoke_obscuration(drone_direction, drone_speed, drop_time, blast_delay, 
                                visualize=True, verbose=True):
    """
    Calculate effective concealment time of a smoke screen for missile M1.

    Parameters:
    drone_direction: drone flight direction vector (3D)
    drone_speed: drone speed (m/s)
    drop_time: time after task start when smoke is dropped (s)
    blast_delay: delay from drop to blast (s)
    visualize: whether to plot results
    verbose: whether to print detailed info

    Returns:
    effective_time: effective concealment time (s)
    """
    
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
    
    # Normalize drone direction vector
    drone_direction = np.array(drone_direction)
    if np.linalg.norm(drone_direction) > 0:
        drone_direction = drone_direction / np.linalg.norm(drone_direction)
    
    # Drone velocity vector
    v_drone_vector = drone_speed * drone_direction
    
    if verbose:
        print(f"Drone velocity vector: {v_drone_vector}")

    # Time parameters
    t_drop = drop_time
    t_blast_delay = blast_delay
    t_blast = t_drop + t_blast_delay

    # Compute positions
    drop_position = FY1_start + v_drone_vector * t_drop
    vertical_drop = 0.5 * g * t_blast_delay**2
    blast_position = np.array([drop_position[0] + v_drone_vector[0] * t_blast_delay, 
                              drop_position[1] + v_drone_vector[1] * t_blast_delay, 
                              drop_position[2] - vertical_drop])

    if verbose:
        print(f"Drop position: {drop_position}")
        print(f"Blast position: {blast_position}")

    # Missile flight direction (towards decoy target)
    missile_direction = fake_target - M1_start
    missile_direction = missile_direction / np.linalg.norm(missile_direction)
    v_missile_vector = v_missile * missile_direction
    
    if verbose:
        print(f"Missile velocity vector: {v_missile_vector}")
        print(f"Missile start position: {M1_start}")

    # Calculate time for missile to reach decoy target
    def time_to_target(position, velocity, target):
        t_x = (target[0] - position[0]) / velocity[0]
        return t_x

    t_missile_to_target = time_to_target(M1_start, v_missile_vector, fake_target)
    
    if verbose:
        print(f"Missile time to decoy target: {t_missile_to_target:.2f} s")

    # Compute missile position at time t
    def missile_position(t):
        return M1_start + v_missile_vector * t

    # Compute smoke cloud position at time t (since blast)
    def smoke_position(t_smoke):
        return np.array([blast_position[0], blast_position[1], blast_position[2] - v_smoke_sink * t_smoke])

    # Angle condition: determine if smoke is between missile and real target
    def is_smoke_between(missile_pos, smoke_pos, target_pos):
        missile_to_smoke = smoke_pos - missile_pos
        missile_to_target = target_pos - missile_pos
        
        dot_product = np.dot(missile_to_smoke, missile_to_target)
        cos_angle = dot_product / (np.linalg.norm(missile_to_smoke) * np.linalg.norm(missile_to_target))
        
        return cos_angle > 0

    # Distance from a point to a line
    def distance_point_to_line(point, line_point, line_direction):
        ap = point - line_point
        projection = np.dot(ap, line_direction) / np.linalg.norm(line_direction)
        foot_point = line_point + projection * line_direction
        return np.linalg.norm(point - foot_point)

    # Determine whether                                                  the missile crosses the smoke cloud
    def is_missile_through_smoke(missile_pos, smoke_pos, missile_prev_pos, smoke_prev_pos):
        radius = effective_radius
        
        missile_move = missile_pos - missile_prev_pos
        missile_move_length = np.linalg.norm(missile_move)
        
        if missile_move_length == 0:
            return False
        
        smoke_move = smoke_pos - smoke_prev_pos
        relative_move = missile_move - smoke_move
        relative_move_length = np.linalg.norm(relative_move)
        
        if relative_move_length == 0:
            return np.linalg.norm(missile_pos - smoke_pos) <= radius
        
        relative_direction = relative_move / relative_move_length
        smoke_to_missile = missile_prev_pos - smoke_prev_pos
        projection = np.dot(smoke_to_missile, relative_direction)
        perpendicular_dist = np.linalg.norm(smoke_to_missile - projection * relative_direction)
        
        if perpendicular_dist <= radius and 0 <= projection <= relative_move_length:
            return True
        
        if np.linalg.norm(missile_prev_pos - smoke_prev_pos) <= radius:
            return True
        if np.linalg.norm(missile_pos - smoke_pos) <= radius:
            return True
        
        return False

    # Calculate effective concealment time
    start_time = 0
    end_time = min(effective_duration, t_missile_to_target - t_blast)
    
    if end_time <= 0:
        return 0.0
    
    time_step = 0.01
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
        
        # Check angle condition
        is_between = is_smoke_between(pos_missile, pos_smoke, real_target)

        # Check whether missile crosses the smoke cloud
        is_through = is_missile_through_smoke(pos_missile, pos_smoke, prev_missile_pos, prev_smoke_pos)

        # Check whether missile is inside the cloud
        in_smoke = np.linalg.norm(pos_missile - pos_smoke) <= effective_radius
        
        if (distance <= effective_radius and is_between) or is_through or in_smoke:
            total_effective_time += time_step
        
        # Update previous positions
        prev_missile_pos = pos_missile
        prev_smoke_pos = pos_smoke
        
        current_time += time_step
    
    if verbose:
        print(f"\n=== RESULTS ===")
        print(f"Smoke drop time: {t_drop:.1f} s")
        print(f"Smoke blast time: {t_blast:.1f} s")
        print(f"Blast position: ({blast_position[0]:.1f}, {blast_position[1]:.1f}, {blast_position[2]:.1f})")
        print(f"Missile time to decoy target: {t_missile_to_target:.2f} s")
        print(f"Effective concealment duration: {total_effective_time:.3f} s")

        # Detailed analysis of key time points
        print(f"\n=== Key timepoint detailed analysis ===")
        for t_check in [0, 1, 2, 3, 4, 5]:
            if t_check <= min(effective_duration, t_missile_to_target - t_blast):
                t_missile_check = t_blast + t_check
                pos_missile = missile_position(t_missile_check)
                pos_smoke = smoke_position(t_check)

                missile_to_target_direction = real_target - pos_missile
                missile_to_target_direction = missile_to_target_direction / np.linalg.norm(missile_to_target_direction)
                distance = distance_point_to_line(pos_smoke, pos_missile, missile_to_target_direction)

                is_between = is_smoke_between(pos_missile, pos_smoke, real_target)
                direct_distance = np.linalg.norm(pos_missile - pos_smoke)

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

    if visualize:
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
        ax.scatter(*M1_start, color='orange', s=100, label='Missile start',)
        ax.scatter(*blast_position, color='green', s=100, label='Blast point')

        # Set view and labels
        ax.view_init(elev=20, azim=45)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Smoke Screen Interference Analysis for Missile M1\nEffective concealment time: {total_effective_time:.3f} s')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    return total_effective_time


effective_time = calculate_smoke_obscuration(
        drone_direction=[-1, 0, 0],  # towards decoy target
        drone_speed=120,
        drop_time=1.5,
        blast_delay=3.6,
        visualize=True,
        verbose=True
)
import numpy as np
from scipy.optimize import basinhopping, minimize
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Objective function (returns negative effective time because we minimize)
def objective(params):
    theta, s, t_drop, t_delay = params
    u = np.cos(theta/10)
    v = np.sin(theta/10)
    dir_vec = [u, v, 0]
    
    effective_time = calculate_smoke_obscuration(
        drone_direction=dir_vec,
        drone_speed=s,
        drop_time=t_drop/10,
        blast_delay=t_delay/10,
        visualize=False,
        verbose=False
    )
    
    return -effective_time  # return negative for minimization

# Custom callback to record optimization progress
class OptimizationTracker:
    def __init__(self):
        self.history = []  # record all function values
        self.params_history = []  # record all parameters
        self.best_history = []  # record best function value
        self.best_params_history = []  # record best parameters
        self.best_value = float('inf')
        self.best_params = None
    
    def __call__(self, x, f, accepted):
        self.history.append(f)
        self.params_history.append(x.copy())
        
        if f < self.best_value:
            self.best_value = f
            self.best_params = x.copy()
        
        self.best_history.append(self.best_value)
        self.best_params_history.append(self.best_params.copy())
        
        if len(self.history) % 10 == 0:
            print(f"Iteration {len(self.history)}: Current value = {-f:.4f}, Best value = {-self.best_value:.4f}")

# Variable bounds
bounds = [(np.pi*9, np.pi*11), (70, 140), (0, 50), (0.1, 50)]

# Initial parameters
initial_params = np.array([np.pi*10, 120, 5, 35])

# Create tracker
tracker = OptimizationTracker()

# Basinhopping optimization
print("Starting basinhopping optimization...")
minimizer_kwargs = {
    "method": "L-BFGS-B", 
    "bounds": bounds,
    "options": {"maxiter": 100}
}

result_sa = basinhopping(
    objective, 
    initial_params, 
    niter=200, 
    minimizer_kwargs=minimizer_kwargs, 
    stepsize=0.5,
    accept_test=None,
    callback=tracker
)

# Extract best parameters
best_params_sa = result_sa.x
best_value_sa = -result_sa.fun  # convert to positive effective time

print("\nBasinhopping optimization results:")
print(f"Best direction angle (theta): {best_params_sa[0]:.4f} rad")
print(f"Best speed: {best_params_sa[1]:.2f} m/s")
print(f"Best drop time: {best_params_sa[2]:.2f} s")
print(f"Best blast delay: {best_params_sa[3]:.2f} s")
print(f"Maximum effective concealment time: {best_value_sa:.4f} s")

# Plot convergence curves
plt.figure(figsize=(15, 5))

# 1. Loss curve
plt.subplot(1, 3, 1)
iterations = range(1, len(tracker.history) + 1)
plt.plot(iterations, [-x for x in tracker.history], 'b-', alpha=0.7, label='Current')
plt.plot(iterations, [-x for x in tracker.best_history], 'r-', linewidth=2, label='Best')
plt.xlabel('Iterations')
plt.ylabel('Effective concealment time (s)')
plt.title('Optimization - Loss Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

# 2. Parameter change curves (normalized)
plt.subplot(1, 3, 2)

# Use parameter history from tracker
theta_norm = [(p[0] - bounds[0][0]) / (bounds[0][1] - bounds[0][0]) for p in tracker.params_history]
speed_norm = [(p[1] - bounds[1][0]) / (bounds[1][1] - bounds[1][0]) for p in tracker.params_history]
drop_norm = [(p[2] - bounds[2][0]) / (bounds[2][1] - bounds[2][0]) for p in tracker.params_history]
delay_norm = [(p[3] - bounds[3][0]) / (bounds[3][1] - bounds[3][0]) for p in tracker.params_history]

plt.plot(iterations, theta_norm, label='theta', alpha=0.7)
plt.plot(iterations, speed_norm, label='speed', alpha=0.7)
plt.plot(iterations, drop_norm, label='drop_time', alpha=0.7)
plt.plot(iterations, delay_norm, label='delay', alpha=0.7)

plt.xlabel('Iterations')
plt.ylabel('Normalized parameter value')
plt.title('Parameter optimization process')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

# 3. Final results comparison
plt.subplot(1, 3, 3)
initial_value = -objective(initial_params)
plt.bar(['Initial', 'After optimization'], [initial_value, best_value_sa], alpha=0.7)
plt.ylabel('Effective concealment time (s)')
plt.title('Before vs After Optimization')
plt.grid(True, alpha=0.3)

# Add numeric labels on bars
for i, v in enumerate([initial_value, best_value_sa]):
    plt.text(i, v + 0.1, f'{v:.2f}s', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Print detailed best parameter information
print("\nDetailed best parameter info:")
print(f"Direction vector: [{np.cos(best_params_sa[0]/10):.4f}, {np.sin(best_params_sa[0]/10):.4f}, 0]")
print(f"Speed: {best_params_sa[1]:.2f} m/s (range: 70-140 m/s)")
print(f"Drop time: {best_params_sa[2]/10:.2f} s (range: 0-5 s)")
print(f"Blast delay: {best_params_sa[3]/10:.2f} s (range: 0.1-5 s)")
print(f"Improvement: {(best_value_sa - initial_value):.2f} s ({((best_value_sa - initial_value)/initial_value*100):.1f}%)")

# Print best strategy
print("\nBest strategy:")
print(f"Drone flight direction: angle {best_params_sa[0]:.4f} rad (approx {np.degrees(best_params_sa[0]):.2f}°)")
print(f"Drone flight speed: {best_params_sa[1]:.2f} m/s")
print(f"Smoke drop time: {best_params_sa[2]:.2f} s after task start")
print(f"Smoke blast delay: {best_params_sa[3]:.2f} s after drop")
print(f"Estimated effective concealment time: {best_value_sa:.2f} s")


