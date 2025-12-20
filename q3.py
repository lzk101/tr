import numpy as np
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from q2 import calculate_smoke_obscuration

def check_time_constraints(drop_times, min_interval=1.0):
    """Check drop time interval constraints"""
    if len(drop_times) <= 1:
        return True
    sorted_times = np.sort(drop_times)
    intervals = np.diff(sorted_times)
    return np.all(intervals >= min_interval)

def calculate_multiple_smoke_obscuration(drone_direction, drone_speed, drop_times, blast_delays,
                                        verbose=False):
    """
    Compute total effective concealment time of multiple smoke bombs for missile M1
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
        print(f"\n=== Multi-bomb Analysis ===")
        for i, (drop, delay, time) in enumerate(zip(drop_times, blast_delays, individual_times)):
            print(f"Bomb {i+1}: Drop={drop:.2f}s, Delay={delay:.2f}s, Effective={time:.3f}s")
        print(f"Total effective concealment time: {total_time:.3f}s")
    
    return total_time

def objective_multiple_bombs(params, num_bombs=3):
    """Objective function for multiple bombs optimization"""
    theta = params[0]
    s = params[1]
    
    # extract drop times and blast delays
    drop_times = [params[2 + 2*i] for i in range(num_bombs)]
    blast_delays = [params[3 + 2*i] for i in range(num_bombs)]
    
    # check time interval constraints
    if not check_time_constraints(drop_times):
        # return a large penalty value with randomness to avoid local minima
        penalty = 100.0 + abs(np.random.normal(0, 10))
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
        
        # return negative total_time for minimization
        return -total_time
        
    except Exception as e:
        print(f"Computation error: {e}")
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
            
            # Check if solution is valid (not a penalty)
            if f < 50:  # assume values <50 are valid
                print(f"Iteration {len(self.history)}: Current = {-f:.3f}s, Best = {-self.best_value:.3f}s")
                print(f"  Theta: {np.degrees(theta):.1f}°, Speed: {s:.1f} m/s")
                for i in range(self.num_bombs):
                    print(f"  Bomb {i+1}: Drop={drop_times[i]:.2f}s, Delay={blast_delays[i]:.2f}s")
            else:
                print(f"Iteration {len(self.history)}: Invalid solution (violates constraints)")

# 设置3弹优化
num_bombs = 3

# Variable bounds - adjusted from single-bomb optimum, ensure time intervals
bounds = [
    (3.12, 3.16),    # theta (based on single-bomb optimal angle 178.9° ≈ 3.124 rad)
    (115, 125),      # speed (based on single-bomb optimal speed 121.32)
]

# Add drop time and blast delay bounds for each bomb, ensure spacing
for i in range(num_bombs):
    bounds.append((i * 1.5, i * 1.5 + 2.0))  # ensure at least 1.5s spacing
    bounds.append((2, 5))                    # blast_delay

# 基于单弹最优解设置初始参数，确保时间间隔
single_opt_theta = 3.124
single_opt_speed = 121.32

# Redesign initial parameters, ensure intervals >= 1s
initial_params = np.array([
    single_opt_theta,      # theta
    single_opt_speed,      # speed
    # 弹1: 较早投放
    0.5, 3.0,
    # 弹2: 中等时间投放（确保与弹1间隔≥1秒）
    2.0, 3.5,
    # 弹3: 较晚投放（确保与弹2间隔≥1秒）
    3.5, 4.0
])

# Create tracker
tracker = MultipleBombsOptimizationTracker(num_bombs=num_bombs)

# 测试初始参数
print("Testing initial parameters...")
print(f"Based on single-bomb optimum: theta={np.degrees(single_opt_theta):.1f}°, speed={single_opt_speed:.2f} m/s")

# Check initial parameter time intervals
initial_drop_times = [initial_params[2], initial_params[4], initial_params[6]]
print(f"Initial drop times: {initial_drop_times}")
print(f"Interval check: {check_time_constraints(initial_drop_times)}")

initial_time = -objective_multiple_bombs(initial_params, num_bombs)
print(f"Initial total effective time (3 bombs): {initial_time:.3f}s")

# Show detailed effects for initial parameters
print("\nDetailed effects for initial parameters:")
calculate_multiple_smoke_obscuration(
    drone_direction=[np.cos(single_opt_theta), np.sin(single_opt_theta), 0],
    drone_speed=single_opt_speed,
    drop_times=initial_drop_times,
    blast_delays=[initial_params[3], initial_params[5], initial_params[7]],
    verbose=True
)

# 模拟退火优化
print("\nStarting 3-bomb basinhopping optimization...")
minimizer_kwargs = {
    "method": "L-BFGS-B", 
    "bounds": bounds,
    "options": {"maxiter": 30, "ftol": 1e-3}
}

# 设置随机种子
np.random.seed(42)

result_sa = basinhopping(
    lambda x: objective_multiple_bombs(x, num_bombs),
    initial_params, 
    niter=50,
    minimizer_kwargs=minimizer_kwargs, 
    stepsize=0.3,
    T=1.5,
    callback=tracker,
    niter_success=10  # stop after 10 consecutive iterations without improvement
)

# 提取最佳参数
if tracker.best_params is not None and tracker.best_value < 50:  # ensure valid solution
    best_params = tracker.best_params
    best_time = -tracker.best_value
else:
    best_params = initial_params
    best_time = initial_time
    print("Optimization failed, using initial parameters")

print(f"\n=== Optimization complete ===")
print(f"Best total effective time: {best_time:.3f}s")

theta = best_params[0]
speed = best_params[1]
drop_times = [best_params[2 + 2*i] for i in range(num_bombs)]
blast_delays = [best_params[3 + 2*i] for i in range(num_bombs)]

print(f"Best direction angle: {np.degrees(theta):.1f}°")
print(f"Best speed: {speed:.2f} m/s")
for i in range(num_bombs):
    print(f"Bomb {i+1}: Drop time={drop_times[i]:.3f}s, Blast delay={blast_delays[i]:.3f}s")

# 验证时间间隔约束
print(f"\nInterval check:")
sorted_drop_times = np.sort(drop_times)
intervals = np.diff(sorted_drop_times)
for i, interval in enumerate(intervals):
    status = "✓" if interval >= 1.0 else "✗"
    print(f"Interval between Bomb {i+1} and Bomb {i+2}: {interval:.3f}s {status}")

# 验证最佳策略效果
print("\n=== Best strategy detailed effects ===")
final_time = calculate_multiple_smoke_obscuration(
    drone_direction=[np.cos(theta), np.sin(theta), 0],
    drone_speed=speed,
    drop_times=drop_times,
    blast_delays=blast_delays,
    verbose=True
)

# 计算单弹最优时间用于对比
single_bomb_time = calculate_smoke_obscuration(
    drone_direction=[np.cos(single_opt_theta), np.sin(single_opt_theta), 0],
    drone_speed=single_opt_speed,
    drop_time=0.37,  # 单弹最优投放时间
    blast_delay=3.48,  # 单弹最优起爆延迟
    verbose=False
)

print(f"\n=== Performance comparison ===")
print(f"Single-bomb optimum time: {single_bomb_time:.3f}s")
print(f"3-bomb optimized time: {final_time:.3f}s")
print(f"Improvement vs single-bomb: {final_time - single_bomb_time:.3f}s "
    f"({(final_time - single_bomb_time)/single_bomb_time*100:.1f}%)")
print(f"Improvement vs initial: {final_time - initial_time:.3f}s "
    f"({(final_time - initial_time)/initial_time*100:.1f}%)")

# 绘制收敛曲线（只绘制有效解）
plt.figure(figsize=(15, 5))

# 过滤掉惩罚值
valid_indices = [i for i, val in enumerate(tracker.history) if val < 50]
valid_iterations = [i+1 for i in valid_indices]
valid_history = [-tracker.history[i] for i in valid_indices]
valid_best_history = [-tracker.best_history[i] for i in valid_indices]

plt.subplot(1, 3, 1)
if valid_iterations:
    plt.plot(valid_iterations, valid_history, 'b-', alpha=0.7, label='Current')
    plt.plot(valid_iterations, valid_best_history, 'r-', linewidth=2, label='Best')
    plt.axhline(y=single_bomb_time, color='g', linestyle='--', label='Single-bomb optimum')
    plt.xlabel('Iterations')
    plt.ylabel('Total effective concealment time (s)')
    plt.title('3-bomb optimization convergence (valid solutions)')
    plt.legend()
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'No valid solutions', ha='center', va='center')

# 输出最终策略
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

print(f"\nEstimated total effective concealment time: {final_time:.3f} s")
print(f"Improvement vs single-bomb strategy: {final_time - single_bomb_time:.3f} s "
    f"({(final_time - single_bomb_time)/single_bomb_time*100:.1f}%)")

plt.tight_layout()
plt.show()
# Improved basinhopping
# Hybrid strategy: combine genetic algorithm (global search) and basinhopping (local optimization)

# Adaptive step size: gradually reduce step size with iterations

# Correlated parameter perturbation: coordinate perturbations for related parameters (drop times)

# Enhanced tracker: show additional statistics including valid-solution ratio

# Smart penalty function: compute penalty based on violation degree

# Multi-stage optimization: coarse global search then fine local optimization

# Visualization improvements: show valid-solution ratio and parameter trends
import numpy as np
from scipy.optimize import basinhopping, differential_evolution
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def calculate_multiple_smoke_obscuration(drone_direction, drone_speed, drop_times, blast_delays, 
                                        verbose=False):
    """
    Compute total effective concealment time of multiple smoke bombs for missile M1
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
        print(f"\n=== Multi-bomb Analysis ===")
        for i, (drop, delay, time) in enumerate(zip(drop_times, blast_delays, individual_times)):
            print(f"Bomb {i+1}: Drop={drop:.2f}s, Delay={delay:.2f}s, Effective={time:.3f}s")
        print(f"Total effective concealment time: {total_time:.3f}s")
    
    return total_time

def check_time_constraints(drop_times, min_interval=1.0):
    """Check drop time interval constraints"""
    if len(drop_times) <= 1:
        return True
    sorted_times = np.sort(drop_times)
    intervals = np.diff(sorted_times)
    return np.all(intervals >= min_interval)

def objective_multiple_bombs(params, num_bombs=3):
    """Objective function for multiple bombs optimization"""
    theta = params[0]
    s = params[1]
    
    # extract drop times and blast delays
    drop_times = [params[2 + 2*i] for i in range(num_bombs)]
    blast_delays = [params[3 + 2*i] for i in range(num_bombs)]
    
    # check time interval constraints
    if not check_time_constraints(drop_times):
        # compute penalty based on violation degree
        sorted_times = np.sort(drop_times)
        intervals = np.diff(sorted_times)
        violation = max(0, 1.0 - np.min(intervals)) if len(intervals) > 0 else 1.0
        penalty = 50.0 + 50.0 * violation  # penalty proportional to violation
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
        print(f"Computation error: {e}")
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
        
        is_valid = f < 50  # valid-solution check
        
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
                status = "Invalid (violates constraints)"
                time_info = f"Penalty = {f:.1f}"

            print(f"Iteration {len(self.history)}: {status}, {time_info}")
            print(f"  Theta: {np.degrees(theta):.1f}°, Speed: {s:.1f} m/s")
            print(f"  Drop times: {[f'{t:.2f}' for t in drop_times]}s")

            # show detailed stats every 20 iterations
            if len(self.history) % 20 == 0:
                valid_ratio = self.valid_solutions / len(self.history) * 100
                print(f"  Valid-solution ratio: {valid_ratio:.1f}%")

def adaptive_step_size(x, iteration, max_iterations):
    """Adaptive step size adjustment"""
    base_step = 0.3
    # gradually reduce step size with iterations
    decay_factor = 1.0 - (iteration / max_iterations) * 0.8
    return base_step * decay_factor

def custom_take_step(bounds, max_iterations):
    """Custom step generator"""
    def take_step(x):
        iteration = len(take_step.counter) if hasattr(take_step, 'counter') else 0
        step_size = adaptive_step_size(x, iteration, max_iterations)

        # set different perturbation strategies for different parameter types
        new_x = x.copy()

        # angle parameter - small perturbation
        new_x[0] += np.random.normal(0, step_size * 0.1)

        # speed parameter - moderate perturbation
        new_x[1] += np.random.normal(0, step_size * 0.5)

        # drop time parameters - correlated perturbations (preserve spacing)
        for i in range(2, len(x), 2):
            # correlated perturbation for drop time
            base_perturb = np.random.normal(0, step_size * 0.3)
            new_x[i] += base_perturb

            # small perturbation for blast delay
            new_x[i+1] += np.random.normal(0, step_size * 0.2)

        # ensure parameters are within bounds
        for i in range(len(new_x)):
            new_x[i] = max(bounds[i][0], min(bounds[i][1], new_x[i]))

        # record iteration count
        if not hasattr(take_step, 'counter'):
            take_step.counter = []
        take_step.counter.append(1)

        return new_x
    return take_step

def hybrid_optimization(objective_func, initial_params, bounds, num_bombs=3, max_iterations=100):
    """Hybrid optimization strategy: differential evolution (global) + basinhopping (local)"""

    print("Phase 1: running differential evolution for global search...")

    # differential evolution for coarse global search
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
    
    print(f"GA result: {-ga_result.fun:.3f}s")

    # use GA result as initial point for basinhopping
    sa_initial = ga_result.x
    
    print("Phase 2: running improved basinhopping for fine optimization...")

    # create improved tracker
    tracker = AdvancedOptimizationTracker(num_bombs=num_bombs)

    # custom step function
    take_step = custom_take_step(bounds, max_iterations)

    # improved basinhopping
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

# 设置3弹优化
num_bombs = 3

# 变量边界
bounds = [
    (3.0, 3.3),      # theta (172°-189°)
    (70, 140),       # speed
]

# 为每个弹添加投放时间和起爆延迟边界
for i in range(num_bombs):
    bounds.append((i * 1.0, i * 1.0 + 5.0))  # 投放时间，确保有重叠空间
    bounds.append((1, 8))                    # blast_delay

# 基于单弹最优解设置初始参数
single_opt_theta = 3.124
single_opt_speed = 121.32

# 初始参数 - 确保时间间隔
initial_params = np.array([
    single_opt_theta,
    single_opt_speed,
    1.0, 3.0,    # 弹1
    3.0, 4.0,    # 弹2（与弹1间隔2秒）
    5.0, 5.0     # 弹3（与弹2间隔2秒）
])

print("开始混合优化...")
result, tracker = hybrid_optimization(
    lambda x: objective_multiple_bombs(x, num_bombs),
    initial_params,
    bounds,
    num_bombs=num_bombs,
    max_iterations=80
)

# 提取最佳参数
if tracker.best_value < 50:  # 有效解
    best_params = tracker.best_params
    best_time = -tracker.best_value
    print(f"优化成功！最佳时间: {best_time:.3f}s")
else:
    best_params = initial_params
    best_time = -objective_multiple_bombs(initial_params, num_bombs)
    print("优化未找到更好解，使用初始参数")

theta = best_params[0]
speed = best_params[1]
drop_times = [best_params[2 + 2*i] for i in range(num_bombs)]
blast_delays = [best_params[3 + 2*i] for i in range(num_bombs)]

print(f"\n=== 最终结果 ===")
print(f"最佳方向角度: {np.degrees(theta):.1f}°")
print(f"最佳速度: {speed:.2f} m/s")
for i in range(num_bombs):
    print(f"弹 {i+1}: 投放={drop_times[i]:.3f}s, 延迟={blast_delays[i]:.3f}s")

# 验证时间间隔
sorted_times = np.sort(drop_times)
intervals = np.diff(sorted_times)
print(f"时间间隔: {intervals}")

# 验证效果
print("\n验证最终策略效果:")
final_time = calculate_multiple_smoke_obscuration(
    drone_direction=[np.cos(theta), np.sin(theta), 0],
    drone_speed=speed,
    drop_times=drop_times,
    blast_delays=blast_delays,
    verbose=True
)

# 绘制优化过程
plt.figure(figsize=(15, 5))

# 1. 收敛曲线
valid_mask = np.array(tracker.history) < 50
valid_indices = np.where(valid_mask)[0]
if len(valid_indices) > 0:
    plt.subplot(1, 3, 1)
    plt.plot(valid_indices + 1, -np.array(tracker.history)[valid_indices], 'b-', alpha=0.7, label='当前值')
    plt.plot(valid_indices + 1, -np.array(tracker.best_history)[valid_indices], 'r-', linewidth=2, label='最佳值')
    plt.xlabel('迭代次数')
    plt.ylabel('有效遮蔽时间 (s)')
    plt.title('优化收敛曲线')
    plt.legend()
    plt.grid(True)

# 2. 参数变化
plt.subplot(1, 3, 2)
theta_vals = [np.degrees(p[0]) for p in tracker.params_history]
speed_vals = [p[1] for p in tracker.params_history]
plt.plot(theta_vals, 'g-', alpha=0.7, label='角度 (°)')
plt.plot(speed_vals, 'b-', alpha=0.7, label='速度 (m/s)')
plt.xlabel('迭代次数')
plt.ylabel('参数值')
plt.title('主要参数变化')
plt.legend()
plt.grid(True)

# 3. 有效解比例
plt.subplot(1, 3, 3)
window_size = 10
valid_ratios = []
for i in range(len(tracker.history)):
    start = max(0, i - window_size + 1)
    window = tracker.history[start:i+1]
    valid_count = sum(1 for val in window if val < 50)
    valid_ratios.append(valid_count / len(window) * 100)

plt.plot(valid_ratios, 'm-', linewidth=2)
plt.xlabel('迭代次数')
plt.ylabel('有效解比例 (%)')
plt.title('有效解比例（滑动窗口）')
plt.grid(True)
plt.ylim(0, 100)

plt.tight_layout()
plt.show()

print(f"\n最终总有效遮蔽时间: {final_time:.3f}s")

