import numpy as np
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from q2 import calculate_smoke_obscuration

def calculate_multiple_smoke_obscuration(drone_direction, drone_speed, drop_times, blast_delays, 
                                        verbose=False):
    """
    Calculate the total effective obscuration time of multiple smoke bombs on missile M1
    """
    total_time = 0
    individual_times = []
    
    for i, (drop_time, blast_delay) in enumerate(zip(drop_times, blast_delays)):
        eff_time = calculate_smoke_obscuration(
            drone_direction=drone_direction,
            drone_speed=drone_speed,
            drop_time=drop_time,
            blast_delay=blast_delay,
            visualize=False,
            verbose=verbose and i == 0
        )
        individual_times.append(eff_time)
        total_time += eff_time
    
    if verbose:
        print(f"\n=== Multiple bombs effect analysis ===")
        for i, (drop, delay, time) in enumerate(zip(drop_times, blast_delays, individual_times)):
            print(f"Bomb {i+1}: Drop={drop:.2f}s, Delay={delay:.2f}s, Effective={time:.3f}s")
        print(f"Total effective obscuration time: {total_time:.3f}s")
    
    return total_time

def check_time_constraints(drop_times, min_interval=1.0):
    """Check drop time interval constraints"""
    if len(drop_times) <= 1:
        return True
    sorted_times = np.sort(drop_times)
    intervals = np.diff(sorted_times)
    return np.all(intervals >= min_interval)

def objective_multiple_bombs(params, num_bombs=3):
    """Multiple bombs objective function"""
    theta = params[0]
    s = params[1]
    
    # Extract drop times and blast delays
    drop_times = [params[2 + 2*i] for i in range(num_bombs)]
    blast_delays = [params[3 + 2*i] for i in range(num_bombs)]
    
    # Check time interval constraints
    if not check_time_constraints(drop_times):
        # Return a large penalty value, but ensure it's not fixed
        penalty = 100.0 + abs(np.random.normal(0, 10))  # Add randomness to avoid local optima
        return penalty
    
    u = np.cos(theta)
    v = np.sin(theta)
    dir_vec = [u, v, 0]
    
    try:
        total_time = calculate_multiple_smoke_obscuration(
            drone_direction=dir_vec,
            drone_speed=s,
            drop_times=drop_times,
            blast_delays=blast_delays,
            verbose=False
        )
        
        # Ensure positive return (minimize negative effective time)
        return -total_time
        
    except Exception as e:
        print(f"Calculation error: {e}")
        return 100.0

class MultipleBombsOptimizationTracker:
    def __init__(self, num_bombs=3):
        self.num_bombs = num_bombs
        self.history = []
        self.params_history = []
        self.best_history = []
        self.best_value = float('inf')
        self.best_params = None
    
    def __call__(self, x, f, accepted):
        self.history.append(f)
        self.params_history.append(x.copy())
        
        if f < self.best_value:
            self.best_value = f
            self.best_params = x.copy()
        
        self.best_history.append(self.best_value)
        
        if len(self.history) % 5 == 0:
            theta = x[0]
            s = x[1]
            drop_times = [x[2 + 2*i] for i in range(self.num_bombs)]
            blast_delays = [x[3 + 2*i] for i in range(self.num_bombs)]
            
            # Check if it's a valid solution (not a penalty value)
            if f < 50:  # Assume less than 50 is a valid solution
                print(f"Iteration {len(self.history)}: Current = {-f:.3f}s, Best = {-self.best_value:.3f}s")
                print(f"  Theta: {np.degrees(theta):.1f}°, Speed: {s:.1f} m/s")
                for i in range(self.num_bombs):
                    print(f"  Bomb {i+1}: Drop={drop_times[i]:.2f}s, Delay={blast_delays[i]:.2f}s")
            else:
                print(f"Iteration {len(self.history)}: Invalid solution (constraint violation)")

# Set up 3-bomb optimization
num_bombs = 3

# Variable bounds - adjusted based on single bomb optimal solution, ensure time intervals
bounds = [
    (3.12, 3.16),    # theta (based on single bomb optimal angle 178.9° ≈ 3.124rad)
    (115, 125),      # speed (based on single bomb optimal speed 121.32)
]

# Add drop time and blast delay bounds for each bomb, ensure time intervals
for i in range(num_bombs):
    bounds.append((i * 1.5, i * 1.5 + 2.0))  # Ensure at least 1.5 second interval
    bounds.append((2, 5))                    # blast_delay

# Set initial parameters based on single bomb optimal solution, ensure time intervals
single_opt_theta = 3.124
single_opt_speed = 121.32

# Redesign initial parameters to ensure time interval ≥1 second
initial_params = np.array([
    single_opt_theta,      # theta
    single_opt_speed,      # speed
    # Bomb 1: earlier drop
    0.5, 3.0,
    # Bomb 2: medium time drop (ensure interval ≥1s with bomb 1)
    2.0, 3.5,
    # Bomb 3: later drop (ensure interval ≥1s with bomb 2)
    3.5, 4.0
])

# Create tracker
tracker = MultipleBombsOptimizationTracker(num_bombs=num_bombs)

# Test initial parameters
print("Testing initial parameters...")
print(f"Based on single bomb optimal solution: theta={np.degrees(single_opt_theta):.1f}°, speed={single_opt_speed:.2f} m/s")

# Check initial parameters time intervals
initial_drop_times = [initial_params[2], initial_params[4], initial_params[6]]
print(f"Initial drop times: {initial_drop_times}")
print(f"Time interval check: {check_time_constraints(initial_drop_times)}")

initial_time = -objective_multiple_bombs(initial_params, num_bombs)
print(f"Initial 3-bomb total effective time: {initial_time:.3f}s")

# Show initial parameters detailed effect
print("\nInitial parameters detailed effect:")
calculate_multiple_smoke_obscuration(
    drone_direction=[np.cos(single_opt_theta), np.sin(single_opt_theta), 0],
    drone_speed=single_opt_speed,
    drop_times=initial_drop_times,
    blast_delays=[initial_params[3], initial_params[5], initial_params[7]],
    verbose=True
)

# Simulated annealing optimization
print("\nStarting 3-bomb simulated annealing optimization...")
minimizer_kwargs = {
    "method": "L-BFGS-B", 
    "bounds": bounds,
    "options": {"maxiter": 30, "ftol": 1e-3}
}

# Set random seed
np.random.seed(42)

result_sa = basinhopping(
    lambda x: objective_multiple_bombs(x, num_bombs),
    initial_params, 
    niter=50,
    minimizer_kwargs=minimizer_kwargs, 
    stepsize=0.3,
    T=1.5,
    callback=tracker,
    niter_success=10  # Stop if no improvement for 10 consecutive iterations
)

# Extract best parameters
if tracker.best_params is not None and tracker.best_value < 50:  # Ensure it's a valid solution
    best_params = tracker.best_params
    best_time = -tracker.best_value
else:
    best_params = initial_params
    best_time = initial_time
    print("Optimization failed, using initial parameters")

print(f"\n=== Optimization completed ===")
print(f"Best total effective time: {best_time:.3f}s")

theta = best_params[0]
speed = best_params[1]
drop_times = [best_params[2 + 2*i] for i in range(num_bombs)]
blast_delays = [best_params[3 + 2*i] for i in range(num_bombs)]

print(f"Best direction angle: {np.degrees(theta):.1f}°")
print(f"Best speed: {speed:.2f} m/s")
for i in range(num_bombs):
    print(f"Bomb {i+1}: Drop time={drop_times[i]:.3f}s, Blast delay={blast_delays[i]:.3f}s")

# Verify time interval constraints
print(f"\nTime interval check:")
sorted_drop_times = np.sort(drop_times)
intervals = np.diff(sorted_drop_times)
for i, interval in enumerate(intervals):
    status = "OK" if interval >= 1.0 else "FAIL"
    print(f"Bomb {i+1} and Bomb {i+2} interval: {interval:.3f}s {status}")

# Verify best strategy effect
print("\n=== Best strategy detailed effect ===")
final_time = calculate_multiple_smoke_obscuration(
    drone_direction=[np.cos(theta), np.sin(theta), 0],
    drone_speed=speed,
    drop_times=drop_times,
    blast_delays=blast_delays,
    verbose=True
)

# Calculate single bomb optimal time for comparison
single_bomb_time = calculate_smoke_obscuration(
    drone_direction=[np.cos(single_opt_theta), np.sin(single_opt_theta), 0],
    drone_speed=single_opt_speed,
    drop_time=0.37,  # Single bomb optimal drop time
    blast_delay=3.48,  # Single bomb optimal blast delay
    verbose=False
)

print(f"\n=== Performance comparison ===")
print(f"Single bomb optimal time: {single_bomb_time:.3f}s")
print(f"3-bomb optimized time: {final_time:.3f}s")
print(f"3-bomb improvement over single bomb: {final_time - single_bomb_time:.3f}s "
    f"({(final_time - single_bomb_time)/single_bomb_time*100:.1f}%)")
print(f"3-bomb improvement over initial: {final_time - initial_time:.3f}s "
    f"({(final_time - initial_time)/initial_time*100:.1f}%)")

# Plot convergence curve (only valid solutions)
plt.figure(figsize=(15, 5))

# Filter out penalty values
valid_indices = [i for i, val in enumerate(tracker.history) if val < 50]
valid_iterations = [i+1 for i in valid_indices]
valid_history = [-tracker.history[i] for i in valid_indices]
valid_best_history = [-tracker.best_history[i] for i in valid_indices]

plt.subplot(1, 3, 1)
if valid_iterations:
    plt.plot(valid_iterations, valid_history, 'b-', alpha=0.7, label='Current value')
    plt.plot(valid_iterations, valid_best_history, 'r-', linewidth=2, label='Best value')
    plt.axhline(y=single_bomb_time, color='g', linestyle='--', label='Single bomb optimal')
    plt.xlabel('Iteration count')
    plt.ylabel('Total effective obscuration time (s)')
    plt.title('3-bomb optimization convergence curve (valid solutions)')
    plt.legend()
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'No valid solutions', ha='center', va='center')

# Output final strategy
print("\n=== Final 3-bomb deployment strategy ===")
print(f"Drone flight direction: {np.degrees(theta):.1f}°")
print(f"Drone flight speed: {speed:.2f} m/s")
print(f"Direction vector: [{np.cos(theta):.4f}, {np.sin(theta):.4f}, 0.000]")

for i in range(num_bombs):
    blast_time = drop_times[i] + blast_delays[i]
    print(f"\nBomb {i+1}:")
    print(f"  Drop time: {drop_times[i]:.3f} s after mission start")
    print(f"  Blast delay: {blast_delays[i]:.3f} s")
    print(f"  Estimated blast time: {blast_time:.3f} s")

print(f"\nEstimated total effective obscuration time: {final_time:.3f} s")
print(f"Improvement over single bomb strategy: {final_time - single_bomb_time:.3f} s "
    f"({(final_time - single_bomb_time)/single_bomb_time*100:.1f}%)")

plt.tight_layout()
plt.show()
# Improved simulated annealing
# Hybrid optimization strategy: combining genetic algorithm (global search) and simulated annealing (local optimization)

# Adaptive step size: automatically adjust step size as iteration progresses

# Correlated parameter perturbation: coordinated perturbation of related parameters (drop times)

# Improved tracker: display more statistical information, including valid solution ratio

# Intelligent penalty function: calculate penalty value based on violation degree

# Multi-stage optimization: coarse search first, then fine optimization

# Visualization improvement: display valid solution ratio and parameter change trends
import numpy as np
from scipy.optimize import basinhopping, differential_evolution
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def calculate_multiple_smoke_obscuration(drone_direction, drone_speed, drop_times, blast_delays, 
                                        verbose=False):
    """
    Calculate the total effective obscuration time of multiple smoke bombs on missile M1
    """
    total_time = 0
    individual_times = []
    
    for i, (drop_time, blast_delay) in enumerate(zip(drop_times, blast_delays)):
        eff_time = calculate_smoke_obscuration(
            drone_direction=drone_direction,
            drone_speed=drone_speed,
            drop_time=drop_time,
            blast_delay=blast_delay,
            visualize=False,
            verbose=verbose and i == 0
        )
        individual_times.append(eff_time)
        total_time += eff_time
    
    if verbose:
        print(f"\n=== Multiple bombs effect analysis ===")
        for i, (drop, delay, time) in enumerate(zip(drop_times, blast_delays, individual_times)):
            print(f"Bomb {i+1}: Drop={drop:.2f}s, Delay={delay:.2f}s, Effective={time:.3f}s")
        print(f"Total effective obscuration time: {total_time:.3f}s")
    
    return total_time

def check_time_constraints(drop_times, min_interval=1.0):
    """Check drop time interval constraints"""
    if len(drop_times) <= 1:
        return True
    sorted_times = np.sort(drop_times)
    intervals = np.diff(sorted_times)
    return np.all(intervals >= min_interval)

def objective_multiple_bombs(params, num_bombs=3):
    """Multiple bombs objective function"""
    theta = params[0]
    s = params[1]
    
    # Extract drop times and blast delays
    drop_times = [params[2 + 2*i] for i in range(num_bombs)]
    blast_delays = [params[3 + 2*i] for i in range(num_bombs)]
    
    # Check time interval constraints
    if not check_time_constraints(drop_times):
        # Calculate penalty value based on violation degree
        sorted_times = np.sort(drop_times)
        intervals = np.diff(sorted_times)
        violation = max(0, 1.0 - np.min(intervals)) if len(intervals) > 0 else 1.0
        penalty = 50.0 + 50.0 * violation  # Penalty proportional to violation degree
        return penalty
    
    u = np.cos(theta)
    v = np.sin(theta)
    dir_vec = [u, v, 0]
    
    try:
        total_time = calculate_multiple_smoke_obscuration(
            drone_direction=dir_vec,
            drone_speed=s,
            drop_times=drop_times,
            blast_delays=blast_delays,
            verbose=False
        )
        
        return -total_time
        
    except Exception as e:
        print(f"Calculation error: {e}")
        return 100.0

class AdvancedOptimizationTracker:
    def __init__(self, num_bombs=3):
        self.num_bombs = num_bombs
        self.history = []
        self.params_history = []
        self.best_history = []
        self.best_value = float('inf')
        self.best_params = None
        self.valid_solutions = 0
    
    def __call__(self, x, f, accepted):
        self.history.append(f)
        self.params_history.append(x.copy())
        
        is_valid = f < 50  # Valid solution judgment
        
        if is_valid:
            self.valid_solutions += 1
            
        if f < self.best_value:
            self.best_value = f
            self.best_params = x.copy()
        
        self.best_history.append(self.best_value)
        
        if len(self.history) % 5 == 0:
            theta = x[0]
            s = x[1]
            drop_times = [x[2 + 2*i] for i in range(self.num_bombs)]
            
            if is_valid:
                status = "Valid"
                time_info = f"Current = {-f:.3f}s, Best = {-self.best_value:.3f}s"
            else:
                status = "Invalid (constraint violation)"
                time_info = f"Penalty value = {f:.1f}"
                
            print(f"Iteration {len(self.history)}: {status}, {time_info}")
            print(f"  Theta: {np.degrees(theta):.1f}°, Speed: {s:.1f} m/s")
            print(f"  Drop times: {[f'{t:.2f}' for t in drop_times]}s")
            
            # Display detailed statistics every 20 iterations
            if len(self.history) % 20 == 0:
                valid_ratio = self.valid_solutions / len(self.history) * 100
                print(f"  Valid solution ratio: {valid_ratio:.1f}%")

def adaptive_step_size(x, iteration, max_iterations):
    """Adaptive step size adjustment"""
    base_step = 0.3
    # Gradually decrease step size as iteration progresses
    decay_factor = 1.0 - (iteration / max_iterations) * 0.8
    return base_step * decay_factor

def custom_take_step(bounds, max_iterations):
    "Custom step size generation function"
    def take_step(x):
        iteration = len(take_step.counter) if hasattr(take_step, 'counter') else 0
        step_size = adaptive_step_size(x, iteration, max_iterations)
        
        # Set different perturbation strategies for different parameter types
        new_x = x.copy()
        
        # Angle parameter - small perturbation
        new_x[0] += np.random.normal(0, step_size * 0.1)
        
        # Speed parameter - medium perturbation
        new_x[1] += np.random.normal(0, step_size * 0.5)
        
        # Drop time parameters - correlated perturbation (maintain intervals)
        for i in range(2, len(x), 2):
            # Correlated perturbation of drop times
            base_perturb = np.random.normal(0, step_size * 0.3)
            new_x[i] += base_perturb
            
            # Smaller perturbation for blast delays
            new_x[i+1] += np.random.normal(0, step_size * 0.2)
        
        # Ensure parameters are within bounds
        for i in range(len(new_x)):
            new_x[i] = max(bounds[i][0], min(bounds[i][1], new_x[i]))
        
        # Record iteration count
        if not hasattr(take_step, 'counter'):
            take_step.counter = []
        take_step.counter.append(1)
        
        return new_x
    return take_step

def hybrid_optimization(objective_func, initial_params, bounds, num_bombs=3, max_iterations=100):
    """Hybrid optimization strategy: genetic algorithm + simulated annealing"""
    
    print("First stage: using genetic algorithm for global search...")
    
    # Genetic algorithm for coarse search
    ga_result = differential_evolution(
        objective_func,
        bounds,
        strategy='best1bin',
        maxiter=20,
        popsize=15,
        tol=1e-4,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True
    )
    
    print(f"Genetic algorithm result: {-ga_result.fun:.3f}s")
    
    # Use genetic algorithm result as initial point for simulated annealing
    sa_initial = ga_result.x
    
    print("Second stage: using improved simulated annealing for fine optimization...")
    
    # Create improved tracker
    tracker = AdvancedOptimizationTracker(num_bombs=num_bombs)
    
    # Custom step size function
    take_step = custom_take_step(bounds, max_iterations)
    
    # Improved simulated annealing
    minimizer_kwargs = {
        "method": "L-BFGS-B", 
        "bounds": bounds,
        "options": {"maxiter": 15, "ftol": 1e-4}
    }
    
    result_sa = basinhopping(
        objective_func,
        sa_initial, 
        niter=max_iterations,
        minimizer_kwargs=minimizer_kwargs, 
        take_step=take_step,
        T=2.0,
        stepsize=0.3,
        callback=tracker,
        niter_success=15,
        interval=10
    )
    
    return result_sa, tracker

# Set up 3-bomb optimization
num_bombs = 3

# Variable bounds
bounds = [
    (3.0, 3.3),      # theta (172°-189°)
    (70, 140),       # speed
]

# Add drop time and blast delay bounds for each bomb
for i in range(num_bombs):
    bounds.append((i * 1.0, i * 1.0 + 5.0))  # Drop time, ensure overlapping space
    bounds.append((1, 8))                    # blast_delay

# Set initial parameters based on single bomb optimal solution
single_opt_theta = 3.124
single_opt_speed = 121.32

# Initial parameters - ensure time intervals
initial_params = np.array([
    single_opt_theta,
    single_opt_speed,
    1.0, 3.0,    # Bomb 1
    3.0, 4.0,    # Bomb 2 (2 seconds after Bomb 1)
    5.0, 5.0     # Bomb 3 (2 seconds after Bomb 2)
])

print("Starting hybrid optimization...")
result, tracker = hybrid_optimization(
    lambda x: objective_multiple_bombs(x, num_bombs),
    initial_params,
    bounds,
    num_bombs=num_bombs,
    max_iterations=80
)

# Extract best parameters
if tracker.best_value < 50:  # Valid solution
    best_params = tracker.best_params
    best_time = -tracker.best_value
    print(f"Optimization successful! Best time: {best_time:.3f}s")
else:
    best_params = initial_params
    best_time = -objective_multiple_bombs(initial_params, num_bombs)
    print("Optimization did not find better solution, using initial parameters")

theta = best_params[0]
speed = best_params[1]
drop_times = [best_params[2 + 2*i] for i in range(num_bombs)]
blast_delays = [best_params[3 + 2*i] for i in range(num_bombs)]

print(f"\n=== Final results ===")
print(f"Best direction angle: {np.degrees(theta):.1f}°")
print(f"Best speed: {speed:.2f} m/s")
for i in range(num_bombs):
    print(f"Bomb {i+1}: Drop={drop_times[i]:.3f}s, Delay={blast_delays[i]:.3f}s")

# Verify time intervals
sorted_times = np.sort(drop_times)
intervals = np.diff(sorted_times)
print(f"Time intervals: {intervals}")

# Verify effect
print("\nVerifying final strategy effect:")
final_time = calculate_multiple_smoke_obscuration(
    drone_direction=[np.cos(theta), np.sin(theta), 0],
    drone_speed=speed,
    drop_times=drop_times,
    blast_delays=blast_delays,
    verbose=True
)

# Plot optimization process
plt.figure(figsize=(15, 5))

# 1. Convergence curve
valid_mask = np.array(tracker.history) < 50
valid_indices = np.where(valid_mask)[0]
if len(valid_indices) > 0:
    plt.subplot(1, 3, 1)
    plt.plot(valid_indices + 1, -np.array(tracker.history)[valid_indices], 'b-', alpha=0.7, label='Current value')
    plt.plot(valid_indices + 1, -np.array(tracker.best_history)[valid_indices], 'r-', linewidth=2, label='Best value')
    plt.xlabel('Iteration count')
    plt.ylabel('Effective obscuration time (s)')
    plt.title('Optimization convergence curve')
    plt.legend()
    plt.grid(True)

# 2. Parameter changes
plt.subplot(1, 3, 2)
theta_vals = [np.degrees(p[0]) for p in tracker.params_history]
speed_vals = [p[1] for p in tracker.params_history]
plt.plot(theta_vals, 'g-', alpha=0.7, label='Angle (°)')
plt.plot(speed_vals, 'b-', alpha=0.7, label='Speed (m/s)')
plt.xlabel('Iteration count')
plt.ylabel('Parameter value')
plt.title('Main parameter changes')
plt.legend()
plt.grid(True)

# 3. Valid solution ratio
plt.subplot(1, 3, 3)
window_size = 10
valid_ratios = []
for i in range(len(tracker.history)):
    start = max(0, i - window_size + 1)
    window = tracker.history[start:i+1]
    valid_count = sum(1 for val in window if val < 50)
    valid_ratios.append(valid_count / len(window) * 100)

plt.plot(valid_ratios, 'm-', linewidth=2)
plt.xlabel('Iteration count')
plt.ylabel('Valid solution ratio (%)')
plt.title('Valid solution ratio (sliding window)')
plt.grid(True)
plt.ylim(0, 100)

plt.tight_layout()
plt.show()

print(f"\nFinal total effective obscuration time: {final_time:.3f}s")

