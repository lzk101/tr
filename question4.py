import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_smoke_obscuration(drone_direction1, drone_speed1, drop_time1, blast_delay1, drone_direction2, drone_speed2, drop_time2, blast_delay2, drone_direction3, drone_speed3, drop_time3, blast_delay3, 
                                visualize=False, verbose=False):
    """
    Compute effective concealment time from smoke screens for missile M1.

    Parameters:
    drone_direction*: direction vectors for drones (3D vectors)
    drone_speed*: drone speeds (m/s)
    drop_time*: time after task assignment to drop smoke (s)
    blast_delay*: delay from drop to detonation (s)
    visualize: whether to plot results
    verbose: whether to print detailed debug information

    Returns:
    effective_time: effective concealment time (s)
    """
    
    # Constants
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

    # Drone start positions
    FY1_start = np.array([17800,0,1800])
    FY2_start = np.array([12000,1400,1400])
    FY3_start = np.array([6000,-3000,700])
    
    # Normalize drone direction vectors
    drone_direction1 = np.array(drone_direction1)
    if np.linalg.norm(drone_direction1) > 0:
        drone_direction1 = drone_direction1 / np.linalg.norm(drone_direction1)
    drone_direction2 = np.array(drone_direction2)  
    if np.linalg.norm(drone_direction2) > 0:
        drone_direction2 = drone_direction2 / np.linalg.norm(drone_direction2)
    drone_direction3 = np.array(drone_direction3)  
    if np.linalg.norm(drone_direction3) > 0:
        drone_direction3 = drone_direction3 / np.linalg.norm(drone_direction3)
    
    # Drone velocity vectors
    v_drone_vector1 = drone_speed1 * drone_direction1
    v_drone_vector2 = drone_speed2 * drone_direction2
    v_drone_vector3 = drone_speed3 * drone_direction3
    
    if verbose:
        print(f"Drone1 velocity vector: {v_drone_vector1}")
        print(f"Drone2 velocity vector: {v_drone_vector2}")
        print(f"Drone3 velocity vector: {v_drone_vector3}")

    # Time parameters
    t_drop1 = drop_time1
    t_blast_delay1 = blast_delay1
    t_blast1 = t_drop1 + t_blast_delay1
    t_drop2 = drop_time2
    t_blast_delay2 = blast_delay2
    t_blast2 = t_drop2 + t_blast_delay2
    t_drop3 = drop_time3
    t_blast_delay3 = blast_delay3
    t_blast3 = t_drop3 + t_blast_delay3

    # Compute drop and blast positions
    drop_position1 = FY1_start + v_drone_vector1 * t_drop1
    vertical_drop1 = 0.5 * g * t_blast_delay1**2
    blast_position1 = np.array([drop_position1[0] + v_drone_vector1[0] * t_blast_delay1, 
                                drop_position1[1] + v_drone_vector1[1] * t_blast_delay1, 
                                drop_position1[2] - vertical_drop1])
    drop_position2 = FY2_start + v_drone_vector2 * t_drop2
    vertical_drop2 = 0.5 * g * t_blast_delay2**2
    blast_position2 = np.array([drop_position2[0] + v_drone_vector2[0] * t_blast_delay2, 
                                drop_position2[1] + v_drone_vector2[1] * t_blast_delay2, 
                                drop_position2[2] - vertical_drop2])
    drop_position3 = FY3_start + v_drone_vector3 * t_drop3
    vertical_drop3 = 0.5 * g * t_blast_delay3**2
    blast_position3 = np.array([drop_position3[0] + v_drone_vector3[0] * t_blast_delay3, 
                                drop_position3[1] + v_drone_vector3[1] * t_blast_delay3, 
                                drop_position3[2] - vertical_drop3])

    if verbose:
        print(f"Drop position1: {drop_position1}")
        print(f"Blast position1: {blast_position1}")
        print(f"Drop position2: {drop_position2}")
        print(f"Blast position2: {blast_position2}")
        print(f"Drop position3: {drop_position3}")
        print(f"Blast position3: {blast_position3}")

    # Missile flight direction (towards fake target)
    missile_direction = fake_target - M1_start
    missile_direction = missile_direction / np.linalg.norm(missile_direction)
    v_missile_vector = v_missile * missile_direction
    
    if verbose:
        print(f"Missile velocity vector: {v_missile_vector}")
        print(f"Missile start position: {M1_start}")

    # Compute time for missile to reach fake target
    def time_to_target(position, velocity, target):
        t_x = (target[0] - position[0]) / velocity[0]
        return t_x

    t_missile_to_target = time_to_target(M1_start, v_missile_vector, fake_target)
    
    if verbose:
        print(f"Missile time to fake target: {t_missile_to_target:.2f} s")

    # Missile position at time t
    def missile_position(t):
        return M1_start + v_missile_vector * t

    # Compute smoke cloud position at time t (since blast)
    def smoke1_position(t_smoke):
        return np.array([blast_position1[0], blast_position1[1], blast_position1[2] - v_smoke_sink * t_smoke])
    def smoke2_position(t_smoke):
        return np.array([blast_position2[0], blast_position2[1], blast_position2[2] - v_smoke_sink * t_smoke])
    def smoke3_position(t_smoke):
        return np.array([blast_position3[0], blast_position3[1], blast_position3[2] - v_smoke_sink * t_smoke])

    # Angle condition: determine if smoke is between missile and real target
    def is_smoke1_between(missile_pos, smoke_pos, target_pos):
        missile_to_smoke = smoke_pos - missile_pos
        missile_to_target = target_pos - missile_pos
        
        dot_product = np.dot(missile_to_smoke, missile_to_target)
        
        cos_angle = dot_product / (np.linalg.norm(missile_to_smoke) * np.linalg.norm(missile_to_target))
        
        return cos_angle > 0
    def is_smoke2_between(missile_pos, smoke_pos, target_pos):
        missile_to_smoke = smoke_pos - missile_pos
        missile_to_target = target_pos - missile_pos
        
        dot_product = np.dot(missile_to_smoke, missile_to_target)
        
        cos_angle = dot_product / (np.linalg.norm(missile_to_smoke) * np.linalg.norm(missile_to_target))
        
        return cos_angle > 0
    def is_smoke3_between(missile_pos, smoke_pos, target_pos):
        missile_to_smoke = smoke_pos - missile_pos
        missile_to_target = target_pos - missile_pos
        
        dot_product = np.dot(missile_to_smoke, missile_to_target)
        
        cos_angle = dot_product / (np.linalg.norm(missile_to_smoke) * np.linalg.norm(missile_to_target))
        
        return cos_angle > 0

    # Distance from point to line
    def distance_point_to_line(point, line_point, line_direction):
        ap = point - line_point
        projection = np.dot(ap, line_direction) / np.linalg.norm(line_direction)
        foot_point = line_point + projection * line_direction
        return np.linalg.norm(point - foot_point)

    # Determine if missile passes through a smoke cloud
    def is_missile_through_smoke1(missile_pos, smoke_pos, missile_prev_pos, smoke_prev_pos):
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
    def is_missile_through_smoke2(missile_pos, smoke_pos, missile_prev_pos, smoke_prev_pos):
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
    def is_missile_through_smoke3(missile_pos, smoke_pos, missile_prev_pos, smoke_prev_pos):
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

    # Compute effective concealment time
    start_time = 0
    end_time = min(effective_duration, t_missile_to_target - t_blast1 - t_blast2 - t_blast3)
    
    if end_time <= 0:
        return 0.0
    
    time_step = 0.01
    total_effective_time = 0.0
    current_time = start_time
    
    # Store previous positions for crossing detection
    prev_missile_pos = missile_position(t_blast1 + t_blast2 + t_blast3 )
    prev_smoke1_pos = smoke1_position(0)
    prev_smoke2_pos = smoke2_position(0)
    prev_smoke3_pos = smoke3_position(0)
    
    while current_time <= end_time:
        t_missile = t_blast1 + t_blast2 + t_blast3 + current_time
        if t_missile > t_missile_to_target:
            break
            
        pos_missile = missile_position(t_missile)
        pos_smoke1 = smoke1_position(current_time)
        pos_smoke2 = smoke2_position(current_time)
        pos_smoke3 = smoke3_position(current_time)
        
        # Compute distance
        missile_to_target_direction = real_target - pos_missile
        missile_to_target_direction = missile_to_target_direction / np.linalg.norm(missile_to_target_direction)
        distance1 = distance_point_to_line(pos_smoke1, pos_missile, missile_to_target_direction)
        distance2 = distance_point_to_line(pos_smoke2, pos_missile, missile_to_target_direction)
        distance3 = distance_point_to_line(pos_smoke3, pos_missile, missile_to_target_direction)
        
        # Check angle condition (any smoke between missile and target)
        is_between = is_smoke1_between(pos_missile, pos_smoke1, real_target) + is_smoke2_between(pos_missile, pos_smoke2, real_target) + is_smoke3_between(pos_missile, pos_smoke3, real_target)
        
        # Check if missile passes through smoke cloud
        is_through = is_missile_through_smoke1(pos_missile, pos_smoke1, prev_missile_pos, prev_smoke1_pos) + is_missile_through_smoke2(pos_missile, pos_smoke2, prev_missile_pos, prev_smoke2_pos) + is_missile_through_smoke3(pos_missile, pos_smoke3, prev_missile_pos, prev_smoke3_pos)
        
        # Check if missile is inside any cloud
        in_smoke1 = np.linalg.norm(pos_missile - pos_smoke1) <= effective_radius
        in_smoke2 = np.linalg.norm(pos_missile - pos_smoke2) <= effective_radius
        in_smoke3 = np.linalg.norm(pos_missile - pos_smoke3) <= effective_radius
        in_smoke = in_smoke1 + in_smoke2 + in_smoke3
        
        if (distance1 <= effective_radius and is_between) or (distance2 <= effective_radius and is_between) or (distance3 <= effective_radius and is_between) or is_through or in_smoke:
            total_effective_time += time_step
        
        # Update previous positions
        prev_missile_pos = pos_missile
        prev_smoke1_pos = pos_smoke1
        prev_smoke2_pos = pos_smoke2
        prev_smoke3_pos = pos_smoke3
        
        current_time += time_step
    
    if verbose:
        print(f"\n=== Results ===")
        print(f"Smoke1 drop time: {t_drop1:.1f} s")
        print(f"Smoke1 blast time: {t_blast1:.1f} s")
        print(f"Blast1 position: ({blast_position1[0]:.1f}, {blast_position1[1]:.1f}, {blast_position1[2]:.1f})")
        print(f"Smoke2 drop time: {t_drop2:.1f} s")
        print(f"Smoke2 blast time: {t_blast2:.1f} s")
        print(f"Blast2 position: ({blast_position2[0]:.1f}, {blast_position2[1]:.1f}, {blast_position2[2]:.1f})")
        print(f"Smoke3 drop time: {t_drop3:.1f} s")
        print(f"Smoke3 blast time: {t_blast3:.1f} s")
        print(f"Blast3 position: ({blast_position3[0]:.1f}, {blast_position3[1]:.1f}, {blast_position3[2]:.1f})")
        print(f"Missile time to fake target: {t_missile_to_target:.2f} s")
        print(f"Effective concealment duration: {total_effective_time:.3f} s")

        # Detailed analysis at key time points
        print(f"\n=== Detailed analysis at key time points ===")
        for t_check in [0, 1, 2, 3, 4, 5]:
            if t_check <= min(effective_duration, t_missile_to_target - t_blast1 - t_blast2 - t_blast3):
                t_missile_check = t_blast1 + t_blast2 + t_blast3 + t_check
                pos_missile = missile_position(t_missile_check)
                pos_smoke1 = smoke1_position(t_check)
                pos_smoke2 = smoke2_position(t_check)
                pos_smoke3 = smoke3_position(t_check)

                missile_to_target_direction = real_target - pos_missile
                missile_to_target_direction = missile_to_target_direction / np.linalg.norm(missile_to_target_direction)
                distance1 = distance_point_to_line(pos_smoke1, pos_missile, missile_to_target_direction)
                distance2 = distance_point_to_line(pos_smoke2, pos_missile, missile_to_target_direction)
                distance3 = distance_point_to_line(pos_smoke3, pos_missile, missile_to_target_direction)

                is_between1 = is_smoke1_between(pos_missile, pos_smoke1, real_target)
                is_between2 = is_smoke2_between(pos_missile, pos_smoke2, real_target)
                is_between3 = is_smoke3_between(pos_missile, pos_smoke3, real_target)
                direct_distance1 = np.linalg.norm(pos_missile - pos_smoke1)
                direct_distance2 = np.linalg.norm(pos_missile - pos_smoke2)
                direct_distance3 = np.linalg.norm(pos_missile - pos_smoke3)

                missile_to_smoke1 = pos_smoke1 - pos_missile
                missile_to_smoke2 = pos_smoke2 - pos_missile
                missile_to_smoke3 = pos_smoke3 - pos_missile
                missile_to_target = real_target - pos_missile
                dot_product1 = np.dot(missile_to_smoke1, missile_to_target)
                dot_product2 = np.dot(missile_to_smoke2, missile_to_target)
                dot_product3 = np.dot(missile_to_smoke3, missile_to_target)
                angle_deg1 = np.degrees(np.arccos(dot_product1 / (np.linalg.norm(missile_to_smoke1) * np.linalg.norm(missile_to_target))))
                angle_deg2 = np.degrees(np.arccos(dot_product2 / (np.linalg.norm(missile_to_smoke2) * np.linalg.norm(missile_to_target))))
                angle_deg3 = np.degrees(np.arccos(dot_product3 / (np.linalg.norm(missile_to_smoke3) * np.linalg.norm(missile_to_target))))

                print(f"Explosion after {t_check}s:")
                print(f"  Missile position: {pos_missile}")
                print(f"  Cloud1 position: {pos_smoke1}")
                print(f"  Cloud2 position: {pos_smoke2}")
                print(f"  Cloud3 position: {pos_smoke3}")
                print(f"  Distance to LOS1: {distance1:.2f} m")
                print(f"  Distance to LOS2: {distance2:.2f} m")
                print(f"  Distance to LOS3: {distance3:.2f} m")
                print(f"  Direct distance1: {direct_distance1:.2f} m")
                print(f"  Direct distance2: {direct_distance2:.2f} m")
                print(f"  Direct distance3: {direct_distance3:.2f} m")
                print(f"  Smoke between missile and target: {'Yes' if (is_between1 or is_between2 or is_between3) else 'No'}")
                print(f"  Missile-Cloud1-Target angle: {angle_deg1:.1f}°")
                print(f"  Missile-Cloud2-Target angle: {angle_deg2:.1f}°")
                print(f"  Missile-Cloud3-Target angle: {angle_deg3:.1f}°")
                print(f"  Inside cloud1: {'Yes' if direct_distance1 <= effective_radius else 'No'}")
                print(f"  Inside cloud2: {'Yes' if direct_distance2 <= effective_radius else 'No'}")
                print(f"  Inside cloud3: {'Yes' if direct_distance3 <= effective_radius else 'No'}")
                print(f"  Effective concealment: {'Yes' if ((distance1 <= effective_radius and is_between1) or direct_distance1 <= effective_radius) or ((distance2 <= effective_radius and is_between2) or direct_distance2 <= effective_radius) or ((distance3 <= effective_radius and is_between3) or direct_distance3 <= effective_radius) else 'No'}")

    if visualize:
        # Visualization
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectories
        missile_times = np.linspace(0, t_missile_to_target, 100)
        missile_traj = np.array([missile_position(t) for t in missile_times])
        ax.plot(missile_traj[:, 0], missile_traj[:, 1], missile_traj[:, 2], 'r-', label='Missile trajectory', linewidth=2)

        # Plot smoke descent trajectories
        smoke1_times = np.linspace(0, min(effective_duration, t_missile_to_target - t_blast1), 50)
        smoke1_traj = np.array([smoke1_position(t) for t in smoke1_times])
        ax.plot(smoke1_traj[:, 0], smoke1_traj[:, 1], smoke1_traj[:, 2], 'b-', label='Smoke descent trajectory 1', linewidth=2)

        smoke2_times = np.linspace(0, min(effective_duration, t_missile_to_target - t_blast2), 50)
        smoke2_traj = np.array([smoke2_position(t) for t in smoke2_times])
        ax.plot(smoke2_traj[:, 0], smoke2_traj[:, 1], smoke2_traj[:, 2], 'g-', label='Smoke descent trajectory 2', linewidth=2)

        smoke3_times = np.linspace(0, min(effective_duration, t_missile_to_target - t_blast3), 50)
        smoke3_traj = np.array([smoke3_position(t) for t in smoke3_times])
        ax.plot(smoke3_traj[:, 0], smoke3_traj[:, 1], smoke3_traj[:, 2], 'm-', label='Smoke descent trajectory 3', linewidth=2)

        # Mark key points
        ax.scatter(*fake_target, color='red', s=200, label='Fake target', marker='x')
        ax.scatter(*real_target, color='blue', s=200, label='Real target', marker='o')
        ax.scatter(*M1_start, color='orange', s=100, label='Missile start')
        ax.scatter(*blast_position1, color='green', s=100, label='Blast point 1')
        ax.scatter(*blast_position2, color='purple', s=100, label='Blast point 2')
        ax.scatter(*blast_position3, color='cyan', s=100, label='Blast point 3')

        # Set view and labels
        ax.view_init(elev=20, azim=45)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Smoke interference analysis for missile M1\nEffective concealment time: {total_effective_time:.3f}s')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    return total_effective_time

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
            (0, 3600),   # i1 range
            (80, 120),   # j1 range
            (0, 15),     # q1 range
            (0, 15),     # s1 range
            (0, 3600),   # i2 range
            (80, 120),   # j2 range
            (0, 15),     # q2 range
            (0, 15),     # s2 range
            (0, 3600),   # i3 range
            (80, 120),   # j3 range
            (0, 15),     # q3 range
            (0, 15)      # s3 range
        ]
    
    def create_individual(self):
        # Create an individual (parameter set)
        individual = []
        for param_range in self.param_ranges:
            individual.append(random.randint(param_range[0], param_range[1] - 1))
        return individual
    
    def create_population(self):
        # Create initial population
        return [self.create_individual() for _ in range(self.pop_size)]
    
    def evaluate(self, individual):
        # Evaluate individual fitness
        i1, j1, q1, s1, i2, j2, q2, s2, i3, j3, q3, s3 = individual
        result = calculate_smoke_obscuration(
            drone_direction1=[np.cos(np.radians(i1/10)), np.sin(np.radians(i1/10)), 0],
            drone_speed1=j1,
            drop_time1=q1,
            blast_delay1=s1,
            drone_direction2=[np.cos(np.radians(i2/10)), np.sin(np.radians(i2/10)), 0],
            drone_speed2=j2,
            drop_time2=q2,
            blast_delay2=s2,
            drone_direction3=[np.cos(np.radians(i3/10)), np.sin(np.radians(i3/10)), 0],
            drone_speed3=j3,
            drop_time3=q3,
            blast_delay3=s3,
            visualize=False,
            verbose=True
        )
        return result  # return fitness value
    
    def select(self, population, fitnesses):
        # Tournament selection
        selected = []
        for _ in range(self.pop_size):
            # randomly select 3 individuals for the tournament
            candidates = random.sample(list(zip(population, fitnesses)), 3)
            # choose the individual with highest fitness
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
            # Evaluate all individuals
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
    # Create GA instance and run
    ga = GeneticAlgorithm(pop_size=100, generations=50)
    solutions = ga.run()

    if solutions:
        print(f"\nFound {len(solutions)} candidate solutions (deduplicated).")

        # Sort by fitness (higher is better) and select top N solutions
        solutions_sorted = sorted(solutions, key=lambda x: x[1], reverse=True)
        top_n = min(5, len(solutions_sorted))
        print(f"\nTop {top_n} solutions:")
        for rank, (sol, fit) in enumerate(solutions_sorted[:top_n], 1):
            print(f"  Rank {rank}: fitness={fit:.3f}")

        # Print best solution details
        best_sol, best_fit = solutions_sorted[0]
        print("\nBest solution details:")
        print(f"Solution : i1={best_sol[0]}, j1={best_sol[1]}, q1={best_sol[2]}, s1={best_sol[3]}, i2={best_sol[4]}, j2={best_sol[5]}, q2={best_sol[6]}, s2={best_sol[7]}, i3={best_sol[8]}, j3={best_sol[9]}, q3={best_sol[10]}, s3={best_sol[11]}, result={best_fit}")
    else:
        print("\nNo valid solutions found")

if __name__ == "__main__":
    main()
