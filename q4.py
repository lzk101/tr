import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_smoke_obscuration(drone_direction, drone_speed, drop_time, blast_delay, 
                                visualize=False, verbose=False):
    """
    Calculate the effective concealment time of smoke screen interference on missile M1
    
    Parameters:
    drone_direction: drone flight direction vector (3D vector)
    drone_speed: drone speed (m/s)
    drop_time: time after task assignment when smoke is dropped (s)
    blast_delay: delay from drop to detonation (s)
    visualize: whether to visualize results
    verbose: whether to output detailed information
    
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

    # Missile M1 initial position
    M1_start = np.array([20000, 0, 2000])

    # Drone FY1 initial position
    FY1_start = np.array([12000,1400,1400])
    
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

    # Calculate positions
    drop_position = FY1_start + v_drone_vector * t_drop
    vertical_drop = 0.5 * g * t_blast_delay**2
    blast_position = np.array([drop_position[0] + v_drone_vector[0] * t_blast_delay, 
                              drop_position[1] + v_drone_vector[1] * t_blast_delay, 
                              drop_position[2] - vertical_drop])

    if verbose:
        print(f"Drop position: {drop_position}")
        print(f"Blast position: {blast_position}")

    # Missile flight direction (towards fake target)
    missile_direction = fake_target - M1_start
    missile_direction = missile_direction / np.linalg.norm(missile_direction)
    v_missile_vector = v_missile * missile_direction
    
    if verbose:
        print(f"Missile velocity vector: {v_missile_vector}")
        print(f"Missile initial position: {M1_start}")

    # Calculate time for missile to reach fake target
    def time_to_target(position, velocity, target):
        t_x = (target[0] - position[0]) / velocity[0]
        return t_x

    t_missile_to_target = time_to_target(M1_start, v_missile_vector, fake_target)
    
    if verbose:
        print(f"Missile time to fake target: {t_missile_to_target:.2f} s")

    # Calculate missile position at time t
    def missile_position(t):
        return M1_start + v_missile_vector * t

    # Calculate smoke cloud position at time t (since blast)
    def smoke_position(t_smoke):
        return np.array([blast_position[0], blast_position[1], blast_position[2] - v_smoke_sink * t_smoke])

    # Angle condition: determine if smoke is between missile and real target
    def is_smoke_between(missile_pos, smoke_pos, target_pos):
        missile_to_smoke = smoke_pos - missile_pos
        missile_to_target = target_pos - missile_pos
        
        dot_product = np.dot(missile_to_smoke, missile_to_target)
        cos_angle = dot_product / (np.linalg.norm(missile_to_smoke) * np.linalg.norm(missile_to_target))
        
        return cos_angle > 0

    # Calculate distance from point to line
    def distance_point_to_line(point, line_point, line_direction):
        ap = point - line_point
        projection = np.dot(ap, line_direction) / np.linalg.norm(line_direction)
        foot_point = line_point + projection * line_direction
        return np.linalg.norm(point - foot_point)

    # Determine if missile passes through smoke cloud
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
    
    # Store previous positions for crossing detection
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
        
        # Check if missile passes through smoke cloud
        is_through = is_missile_through_smoke(pos_missile, pos_smoke, prev_missile_pos, prev_smoke_pos)
        
        # Check if missile is inside the cloud
        in_smoke = np.linalg.norm(pos_missile - pos_smoke) <= effective_radius
        
        if (distance <= effective_radius and is_between) or is_through or in_smoke:
            total_effective_time += time_step
        
        # Update previous positions
        prev_missile_pos = pos_missile
        prev_smoke_pos = pos_smoke
        
        current_time += time_step
    
    if verbose:
        print(f"\n=== Calculation Results ===")
        print(f"Smoke bomb drop time: {t_drop:.1f} s")
        print(f"Smoke bomb blast time: {t_blast:.1f} s")
        print(f"Blast position: ({blast_position[0]:.1f}, {blast_position[1]:.1f}, {blast_position[2]:.1f})")
        print(f"Missile time to fake target: {t_missile_to_target:.2f} s")
        print(f"Effective concealment duration: {total_effective_time:.3f} seconds")

        # Detailed analysis of key time points
        print(f"\n=== Detailed Analysis of Key Time Points ===")
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
                print(f"  Cloud position: {pos_smoke}")
                print(f"  Distance to LOS: {distance:.2f}m")
                print(f"  Direct distance: {direct_distance:.2f}m")
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
        ax.scatter(*fake_target, color='red', s=200, label='Fake target', marker='x')
        ax.scatter(*real_target, color='blue', s=200, label='Real target', marker='o')
        ax.scatter(*M1_start, color='orange', s=100, label='Missile start')
        ax.scatter(*blast_position, color='green', s=100, label='Blast point')

        # Set view and labels
        ax.view_init(elev=20, azim=45)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Smoke screen interference analysis for missile M1\nEffective concealment time: {total_effective_time:.3f}s')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    return total_effective_time


effective_time = calculate_smoke_obscuration(
        drone_direction=[-0.5, -0.5, 0],  # towards fake target
        drone_speed=110,
        drop_time=20,
        blast_delay=7,
        visualize=True,
        verbose=False
    )
import numpy as np
import tqdm
import random


# Custom genetic algorithm implementation
class GeneticAlgorithm:
    def __init__(self, pop_size=100, crossover_rate=0.7, mutation_rate=0.2, generations=50):
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        
        # Parameter ranges
        self.param_ranges = [
            (40, 50),   # i range
            (80, 120),  # j range
            (5, 20),    # q range
            (0, 15)     # s range
        ]
    
    def create_individual(self):
        # Create an individual (set of parameters)
        individual = []
        for param_range in self.param_ranges:
            individual.append(random.randint(param_range[0], param_range[1] - 1))
        return individual
    
    def create_population(self):
        # Create initial population
        return [self.create_individual() for _ in range(self.pop_size)]
    
    def evaluate(self, individual):
        # Evaluate individual's fitness
        i, j, q, s = individual
        result = calculate_smoke_obscuration(
            drone_direction=[np.cos(i/10), np.sin(i/10), 0],
            drone_speed=j,
            drop_time=q,
            blast_delay=s,
            visualize=False,
            verbose=False
        )
        return result  # Return fitness value
    
    def select(self, population, fitnesses):
        # Tournament selection
        selected = []
        for _ in range(self.pop_size):
            # Randomly select 3 individuals for competition
            candidates = random.sample(list(zip(population, fitnesses)), 3)
            # Select the one with highest fitness
            winner = max(candidates, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected
    
    def crossover(self, parent1, parent2):
        # Single-point crossover
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1, parent2
    
    def mutate(self, individual):
        # Uniform mutation
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                low, high = self.param_ranges[i]
                mutated[i] = random.randint(low, high - 1)
        return mutated
    
    def run(self):
        # Create initial population
        population = self.create_population()
        
        # Store all valid solutions
        solutions = []
        
        print("Starting genetic algorithm optimization...")
        for gen in tqdm.tqdm(range(self.generations)):
            # Evaluate all individuals in population
            fitnesses = [self.evaluate(ind) for ind in population]
            
            # Record valid solutions
            for i, fit in enumerate(fitnesses):
                if fit != 0:
                    solutions.append((population[i], fit))
            
            # Selection
            selected = self.select(population, fitnesses)
            
            # Crossover and mutation
            next_population = []
            for i in range(0, self.pop_size, 2):
                parent1 = selected[i]
                parent2 = selected[i+1] if i+1 < self.pop_size else selected[0]
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                next_population.extend([child1, child2])
            
            # Ensure population size remains constant
            population = next_population[:self.pop_size]
        
        # Remove duplicate solutions
        unique_solutions = []
        seen = set()
        for sol, fit in solutions:
            key = tuple(sol)
            if key not in seen:
                seen.add(key)
                unique_solutions.append((sol, fit))
        
        return unique_solutions

def main():
    # Create genetic algorithm instance and run
    ga = GeneticAlgorithm(pop_size=100, generations=50)
    solutions = ga.run()
    
    # Output results
    if solutions:
        print(f"\nFound {len(solutions)} valid solutions:")
        for i, (sol, fit) in enumerate(solutions, 1):
            print(f"Solution {i}: i={sol[0]}, j={sol[1]}, q={sol[2]}, s={sol[3]}, Result={fit}")
    else:
        print("\nNo valid solutions found")

if __name__ == "__main__":
    main()

