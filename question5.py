import numpy as np
from scipy.optimize import linear_sum_assignment

def calculate_smoke_obscuration(drone_direction, drone_speed, drop_time, blast_delay, 
                               visualize=False, verbose=False):
    """
    Calculate the effective concealment time of smoke screen interference for missiles.
    
    Parameters:
    drone_direction: drone direction vectors (array of 5 x 3)
    drone_speed: drone speeds (array 5 x 3 or 5)
    drop_time: drop times for each drone's smoke charges (5 x 3)
    blast_delay: delay from drop to detonation for each charge (5 x 3)
    visualize: whether to visualize results
    verbose: whether to print detailed information

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

    # Missile initial positions
    M_start = np.array([[20000, 0, 2000],
                        [19000, 600, 2100],
                        [18000, -600, 1900]])

    # Drone initial positions
    FY_start = np.array([[17800, 0, 1800],
                        [12000, 1400, 1400],
                        [6000, -3000, 700],
                        [11000, 2000, 1800],
                        [13000, -2000, 1300]])
    
    # Normalize drone direction vectors
    for i in range(5):
        drone_direction[i] = drone_direction[i] / np.linalg.norm(drone_direction[i])

    # Drone velocity vectors
    v_drone_vector = drone_speed * drone_direction

    if verbose:
        print(f"Drone1 velocity vector: {v_drone_vector[0]}")
        print(f"Drone2 velocity vector: {v_drone_vector[1]}")
        print(f"Drone3 velocity vector: {v_drone_vector[2]}")
        print(f"Drone4 velocity vector: {v_drone_vector[3]}")
        print(f"Drone5 velocity vector: {v_drone_vector[4]}")

    # Time parameters
    t_drop = drop_time
    t_blast_delay = blast_delay
    t_blast = t_drop + t_blast_delay

    # Compute drop and blast positions for each drone and each smoke charge
    drop_position = np.zeros((5, 3, 3))
    blast_position = np.zeros((5, 3, 3))
    for i in range(5):
        for j in range(3):
            drop_position[i, j, ] = FY_start[i] + v_drone_vector[i] * t_drop[i, j]
            vertical_drop = 0.5 * g * t_blast_delay[i, j]**2
            blast_position[i, j, ] = [drop_position[i, j, 0] + v_drone_vector[i, 0] * t_blast_delay[i, j], 
                                    drop_position[i, j, 1] + v_drone_vector[i, 1] * t_blast_delay[i, j], 
                                    drop_position[i, j, 2] - vertical_drop]

    if verbose:
        print(f"Drone1 smoke1 drop position: {drop_position[0, 0, ]}")
        print(f"Drone1 smoke1 blast position: {blast_position[0, 0, ]}")
        print(f"Drone1 smoke2 drop position: {drop_position[0, 1, ]}")
        print(f"Drone1 smoke2 blast position: {blast_position[0, 1, ]}")
        print(f"Drone1 smoke3 drop position: {drop_position[0, 2, ]}")
        print(f"Drone1 smoke3 blast position: {blast_position[0, 2, ]}")
        print(f"Drone2 smoke1 drop position: {drop_position[1, 0, ]}")
        print(f"Drone2 smoke1 blast position: {blast_position[1, 0, ]}")
        print(f"Drone2 smoke2 drop position: {drop_position[1, 1, ]}")
        print(f"Drone2 smoke2 blast position: {blast_position[1, 1, ]}")
        print(f"Drone2 smoke3 drop position: {drop_position[1, 2, ]}")
        print(f"Drone2 smoke3 blast position: {blast_position[1, 2, ]}")
        print(f"Drone3 smoke1 drop position: {drop_position[2, 0, ]}")
        print(f"Drone3 smoke1 blast position: {blast_position[2, 0, ]}")
        print(f"Drone3 smoke2 drop position: {drop_position[2, 1, ]}")
        print(f"Drone3 smoke2 blast position: {blast_position[2, 1, ]}")
        print(f"Drone3 smoke3 drop position: {drop_position[2, 2, ]}")
        print(f"Drone3 smoke3 blast position: {blast_position[2, 2, ]}")
        print(f"Drone4 smoke1 drop position: {drop_position[3, 0, ]}")
        print(f"Drone4 smoke1 blast position: {blast_position[3, 0, ]}")
        print(f"Drone4 smoke2 drop position: {drop_position[3, 1, ]}")
        print(f"Drone4 smoke2 blast position: {blast_position[3, 1, ]}")
        print(f"Drone4 smoke3 drop position: {drop_position[3, 2, ]}")
        print(f"Drone4 smoke3 blast position: {blast_position[3, 2, ]}")
        print(f"Drone5 smoke1 drop position: {drop_position[4, 0, ]}")
        print(f"Drone5 smoke1 blast position: {blast_position[4, 0, ]}")
        print(f"Drone5 smoke2 drop position: {drop_position[4, 1, ]}")
        print(f"Drone5 smoke2 blast position: {blast_position[4, 1, ]}")
        print(f"Drone5 smoke3 drop position: {drop_position[4, 2, ]}")
        print(f"Drone5 smoke3 blast position: {blast_position[4, 2, ]}")
    

    # Missile flight directions (towards fake target)
    missile_direction = np.zeros((3, 3))
    v_missile_vector = np.zeros((3, 3))
    for i in range(3):
        missile_direction[i] = fake_target - M_start[i]
        missile_direction[i] = missile_direction[i] / np.linalg.norm(missile_direction[i, ])
    v_missile_vector = v_missile * missile_direction
    
    if verbose:
        print(f"Missile1 velocity vector: {v_missile_vector[0]}")
        print(f"Missile1 start position: {M_start[0]}")
        print(f"Missile2 velocity vector: {v_missile_vector[1]}")
        print(f"Missile2 start position: {M_start[1]}")
        print(f"Missile3 velocity vector: {v_missile_vector[2]}")
        print(f"Missile3 start position: {M_start[2]}")

    # Time for missile to reach fake target
    def time_to_target(num, position, velocity, target):
        t_x = (target[0] - position[num, 0]) / velocity[num, 0]
        return t_x

    t_missile_to_target = np.array([time_to_target(0, M_start, v_missile_vector, fake_target), 
                                    time_to_target(1, M_start, v_missile_vector, fake_target), 
                                    time_to_target(2, M_start, v_missile_vector, fake_target)])
    
    if verbose:
        print(f"Missile1 time to fake target: {t_missile_to_target[0]:.2f} s")
        print(f"Missile2 time to fake target: {t_missile_to_target[1]:.2f} s")
        print(f"Missile3 time to fake target: {t_missile_to_target[2]:.2f} s")

    # Missile position at time t
    def missile_position(num, t):
        return M_start[num] + v_missile_vector[num] * t

    # Smoke cloud position at time t (since blast)
    def smoke_position(Dnum, Snum, t_smoke):
        return np.array([blast_position[Dnum, Snum, 0], blast_position[Dnum, Snum, 1], blast_position[Dnum, Snum, 2] - v_smoke_sink * t_smoke])#[Dnum, Snum]

    # Angle condition: determine if smoke is between missile and real target
    def is_smoke_between(Mnum, Dnum, Snum, missile_pos, smoke_pos, target_pos):
        missile_to_smoke = smoke_pos[Dnum, Snum] - missile_pos[Mnum, Dnum, Snum]
        missile_to_target = target_pos - missile_pos[Mnum, Dnum, Snum]
        
        dot_product = np.dot(missile_to_smoke, missile_to_target)
        cos_angle = dot_product / (np.linalg.norm(missile_to_smoke) * np.linalg.norm(missile_to_target))
        
        return cos_angle > 0

    # Distance from point to line
    def distance_point_to_line(point, line_point, line_direction):
        ap = point - line_point
        projection = np.dot(ap, line_direction) / np.linalg.norm(line_direction)
        foot_point = line_point + projection * line_direction
        return np.linalg.norm(point - foot_point)

    # Determine if missile passes through smoke cloud
    def is_missile_through_smoke(Mnum, Dnum, Snum, missile_pos, smoke_pos, missile_prev_pos, smoke_prev_pos):
        radius = effective_radius
        
        missile_move = missile_pos[Mnum, Dnum, Snum] - missile_prev_pos[Mnum, Dnum, Snum]
        missile_move_length = np.linalg.norm(missile_move)
        
        if missile_move_length == 0:
            return False
        
        smoke_move = smoke_pos[Dnum, Snum] - smoke_prev_pos[Dnum, Snum]
        relative_move = missile_move - smoke_move
        relative_move_length = np.linalg.norm(relative_move)
        
        if relative_move_length == 0:
            return np.linalg.norm(missile_pos[Mnum, Dnum, Snum] - smoke_pos[Dnum, Snum]) <= radius
        
        relative_direction = relative_move / relative_move_length
        smoke_to_missile = missile_prev_pos[Mnum, Dnum, Snum] - smoke_prev_pos[Dnum, Snum]
        projection = np.dot(smoke_to_missile, relative_direction)
        perpendicular_dist = np.linalg.norm(smoke_to_missile - projection * relative_direction)
        
        if perpendicular_dist <= radius and 0 <= projection <= relative_move_length:
            return True
        
        if np.linalg.norm(missile_prev_pos[Mnum, Dnum, Snum] - smoke_prev_pos[Dnum, Snum]) <= radius:
            return True
        if np.linalg.norm(missile_pos[Mnum, Dnum, Snum] - smoke_pos[Dnum, Snum]) <= radius:
            return True
        
        return False

    # Compute effective concealment time
    start_time = 0.0
    End_time = np.arange(3 * 5 * 3).reshape(3, 5, 3)
    for i in range(3):
        for j in range(5):
            for k in range(3):
                tx = t_missile_to_target[i] - t_blast[j, k]
                if tx <= 0:
                    tx = 0
                End_time[i, j, k] = min(effective_duration, tx)
    for i in range(3):
        for j in range(5):
            for k in range(3):
                if End_time[i, j, k] <= 0:
                    End_time[i, j, k] = 0
    end_time = np.max(End_time)
    
    time_step = 0.01
    total_effective_time = 0.0
    current_time = start_time
    
    # Store previous positions for crossing detection
    prev_missile_pos = np.zeros((3, 5, 3, 3))
    prev_smoke_pos = np.zeros((5, 3, 3))
    pos_missile = np.zeros((3, 5, 3, 3))
    pos_smoke = np.zeros((5, 3, 3))

    for i in range(3):
        for j in range(5): 
            for k in range(3):
                prev_missile_pos[i, j, k] = missile_position(i, t_blast[j, k])
                prev_smoke_pos[j, k, ] = smoke_position(j, k, 0)
                
                while current_time <= end_time:
                    t_missile = t_blast[j, k] + current_time
                    if t_missile > t_missile_to_target[i]:
                        break
                        
                    pos_missile[i, j, k, ] = missile_position(i, t_missile)
                    pos_smoke[j, k] = smoke_position(j, k, current_time)
                    
                    # Compute distance
                    missile_to_target_direction = real_target - pos_missile[i, j, k, ]
                    missile_to_target_direction = missile_to_target_direction / np.linalg.norm(missile_to_target_direction)
                    distance = distance_point_to_line(pos_smoke[j, k], pos_missile[i, j, k], missile_to_target_direction)
                    
                    # Check angle condition
                    is_between = is_smoke_between(i, j, k, pos_missile, pos_smoke, real_target)
                    
                    # Check if missile passes through smoke cloud
                    is_through = is_missile_through_smoke(i, j, k, pos_missile, pos_smoke, prev_missile_pos, prev_smoke_pos)
                    
                    # Check if missile is inside the cloud
                    in_smoke = np.linalg.norm(pos_missile[i, j, k] - pos_smoke[j, k]) <= effective_radius
                    
                    if (distance <= effective_radius and is_between) or is_through or in_smoke:
                        total_effective_time += time_step
                    
                    # Update previous positions
                    prev_missile_pos[i, j, k, ] = pos_missile[i, j, k, ]
                    prev_smoke_pos[j, k, ] = pos_smoke[j, k, ]
                    
                    current_time += time_step
                
    return total_effective_time

# Example call
effective_time = calculate_smoke_obscuration(
        drone_direction = np.array([[-0.5, -0.5, 0], 
                                    [-0.5, -0.5, 0], 
                                    [-0.5, -0.5, 0], 
                                    [-0.5, -0.5, 0], 
                                    [-0.5, -0.5, 0]]),
        drone_speed = np.array([[100, 100, 100],
                                [100, 100, 100],
                                [100, 100, 100],
                                [100, 100, 100],
                                [100, 100, 100]]),
        drop_time = np.array([[10, 15, 20],
                              [10, 15, 20],
                              [10, 15, 20],
                              [10, 15, 20],
                              [10, 15, 20]]),
        blast_delay = np.array([[6, 6, 6],
                                [6, 6, 6],
                                [6, 6, 6],
                                [6, 6, 6],
                                [6, 6, 6]]),
        visualize = True,
        verbose = False
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
            (0, 63),    # i1 range
            (0, 63),    # i2 range
            (0, 63),    # i3 range
            (0, 63),    # i4 range
            (0, 63),    # i5 range
            (80, 120),  # j1 range
            (80, 120),  # j2 range
            (80, 120),  # j3 range
            (80, 120),  # j4 range
            (80, 120),  # j5 range
            (0, 30),    # q11 range
            (0, 15),    # s11 range
            (0, 30),    # q12 range
            (0, 15),    # s12 range
            (0, 30),    # q13 range
            (0, 15),    # s13 range
            (0, 30),    # q21 range
            (0, 15),    # s21 range
            (0, 30),    # q22 range
            (0, 15),    # s22 range
            (0, 30),    # q23 range
            (0, 15),    # s23 range
            (0, 30),    # q31 range
            (0, 15),    # s31 range
            (0, 30),    # q32 range
            (0, 15),    # s32 range
            (0, 30),    # q33 range
            (0, 15),    # s33 range
            (0, 30),    # q41 range
            (0, 15),    # s41 range
            (0, 30),    # q42 range
            (0, 15),    # s42 range
            (0, 30),    # q43 range
            (0, 15),    # s43 range
            (0, 30),    # q51 range
            (0, 15),    # s51 range
            (0, 30),    # q52 range
            (0, 15),    # s52 range
            (0, 30),    # q53 range
            (0, 15)     # s53 range
        ]
    
    def create_individual(self):
        # Create an individual (a set of parameters)
        individual = []
        for param_range in self.param_ranges:
            individual.append(random.randint(param_range[0], param_range[1] - 1))
        return individual
    
    def create_population(self):
        # Create initial population
        return [self.create_individual() for _ in range(self.pop_size)]
    
    def evaluate(self, individual):
        # Evaluate fitness of an individual
        i1, i2, i3, i4, i5, j1, j2, j3, j4, j5, q11, s11, q12, s12, q13, s13, q21, s21, q22, s22, q23, s23, q31, s31, q32, s32, q33, s33, q41, s41, q42, s42, q43, s43, q51, s51, q52, s52, q53, s53 = individual
        result = calculate_smoke_obscuration(
            drone_direction=np.array([[np.cos(i1/10), np.sin(i1/10), 0], 
                                      [np.cos(i2/10), np.sin(i2/10), 0], 
                                      [np.cos(i3/10), np.sin(i3/10), 0], 
                                      [np.cos(i4/10), np.sin(i4/10), 0], 
                                      [np.cos(i5/10), np.sin(i5/10), 0]]),
            drone_speed=np.array([[j1, j1, j1],
                                  [j2, j2, j2],
                                  [j3, j3, j3],
                                  [j4, j4, j4],
                                  [j5, j5, j5]]),
            drop_time=np.array([[q11, q12, q13], 
                                [q21, q22, q23], 
                                [q31, q32, q33], 
                                [q41, q42, q43], 
                                [q51, q52, q53]]),
            blast_delay=np.array([[s11, s12, s13], 
                                  [s21, s22, s23], 
                                  [s31, s32, s33], 
                                  [s41, s42, s43], 
                                  [s51, s52, s53]]),
            visualize=False,
            verbose=False
        )
        return result  # return fitness value
    
    def select(self, population, fitnesses):
        # Tournament selection
        selected = []
        for _ in range(self.pop_size):
            # randomly select 3 individuals for competition
            candidates = random.sample(list(zip(population, fitnesses)), 3)
            # select the one with highest fitness
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
            # Evaluate all individuals in the population
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
    # Output and select best solutions
    if solutions:
        print(f"\nFound {len(solutions)} candidate solutions (deduplicated).")

        # Sort by fitness (higher is better) and select top N solutions
        solutions_sorted = sorted(solutions, key=lambda x: x[1], reverse=True)
        # solutions_sorted = heapq.nlargest(top_n, solutions, key=lambda x: x[1])
        top_n = min(5, len(solutions_sorted))
        print(f"\nTop {top_n} solutions:")
        for rank, (sol, fit) in enumerate(solutions_sorted[:top_n], 1):
            print(f"  Rank {rank}: fitness={fit:.3f}")
        
        # Print best solution details
        best_sol, best_fit = solutions_sorted[0]
        print("\nBest solution details:")
        print(f"i1={best_sol[0]},    j1={best_sol[5]},    q11={best_sol[10]},  s11={best_sol[11]},  q12={best_sol[12]},  s12={best_sol[13]},  q13={best_sol[14]},  s13={best_sol[15]}")
        print(f"i2={best_sol[1]},    j2={best_sol[6]},    q21={best_sol[16]},  s21={best_sol[17]},  q22={best_sol[18]},  s22={best_sol[19]},  q23={best_sol[20]},  s23={best_sol[21]}")
        print(f"i3={best_sol[2]},    j3={best_sol[7]},    q31={best_sol[22]},  s31={best_sol[23]},  q32={best_sol[24]},  s32={best_sol[25]},  q33={best_sol[26]},  s33={best_sol[27]}")
        print(f"i4={best_sol[3]},    j4={best_sol[8]},    q41={best_sol[28]},  s41={best_sol[29]},  q42={best_sol[30]},  s42={best_sol[31]},  q43={best_sol[32]},  s43={best_sol[33]}")
        print(f"i5={best_sol[4]},    j5={best_sol[9]},    q51={best_sol[34]},  s51={best_sol[35]},  q52={best_sol[36]},  s52={best_sol[37]},  q53={best_sol[38]},  s53={best_sol[39]}")
        print(f"fitness={best_fit}\n")
    else:
        print("\nNo valid solutions found")

if __name__ == "__main__":
    main()


