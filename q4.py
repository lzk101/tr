import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_smoke_obscuration(drone_direction1, drone_speed1, drop_time1, blast_delay1, drone_direction2, drone_speed2, drop_time2, blast_delay2, drone_direction3, drone_speed3, drop_time3, blast_delay3, 
                                visualize=False, verbose=False):
    """
    计算烟幕对导弹M1的有效遮蔽时间
    
    参数：
    drone_direction: 无人机飞行方向向量 (3D向量)
    drone_speed: 无人机速度 (m/s)
    drop_time: 任务分配后投放烟幕的时间 (s)
    blast_delay: 从投放至爆炸的延迟 (s)
    visualize: 是否可视化结果
    verbose: 是否输出详细信息
    
    返回：
    effective_time: 有效遮蔽时间 (s)
    """
    
    # 定义常量
    g = 9.8
    v_missile = 300
    v_smoke_sink = 3
    effective_radius = 10
    effective_duration = 20

    # 目标位置
    fake_target = np.array([0, 0, 0])
    real_target = np.array([0, 200, 0])

    # 导弹M1初始位置
    M1_start = np.array([20000, 0, 2000])

    # 无人机FY1初始位置
    FY1_start = np.array([17800,0,1800])
    FY2_start = np.array([12000,1400,1400])
    FY3_start = np.array([6000,-3000,700])
    
    # 归一化无人机方向向量
    drone_direction1 = np.array(drone_direction1)
    if np.linalg.norm(drone_direction1) > 0:
        drone_direction1 = drone_direction1 / np.linalg.norm(drone_direction1)
    drone_direction2 = np.array(drone_direction2)  
    if np.linalg.norm(drone_direction2) > 0:
        drone_direction2 = drone_direction2 / np.linalg.norm(drone_direction2)
    drone_direction3 = np.array(drone_direction3)  
    if np.linalg.norm(drone_direction3) > 0:
        drone_direction3 = drone_direction3 / np.linalg.norm(drone_direction3)
    
    # 无人机速度向量
    v_drone_vector1 = drone_speed1 * drone_direction1
    v_drone_vector2 = drone_speed2 * drone_direction2
    v_drone_vector3 = drone_speed3 * drone_direction3
    
    if verbose:
        print(f"无人机1速度向量: {v_drone_vector1}")
        print(f"无人机2速度向量: {v_drone_vector2}")
        print(f"无人机3速度向量: {v_drone_vector3}")

    # 时间参数
    t_drop1 = drop_time1
    t_blast_delay1 = blast_delay1
    t_blast1 = t_drop1 + t_blast_delay1
    t_drop2 = drop_time2
    t_blast_delay2 = blast_delay2
    t_blast2 = t_drop2 + t_blast_delay2
    t_drop3 = drop_time3
    t_blast_delay3 = blast_delay3
    t_blast3 = t_drop3 + t_blast_delay3

    # 计算位置
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
        print(f"投放位置1: {drop_position1}")
        print(f"爆炸位置1: {blast_position1}")
        print(f"投放位置2: {drop_position2}")
        print(f"爆炸位置2: {blast_position2}")
        print(f"投放位置3: {drop_position3}")
        print(f"爆炸位置3: {blast_position3}")

    # 导弹飞行方向（朝向假目标）
    missile_direction = fake_target - M1_start
    missile_direction = missile_direction / np.linalg.norm(missile_direction)
    v_missile_vector = v_missile * missile_direction
    
    if verbose:
        print(f"导弹速度向量: {v_missile_vector}")
        print(f"导弹初始位置: {M1_start}")

    # 计算导弹到达假目标的时间
    def time_to_target(position, velocity, target):
        t_x = (target[0] - position[0]) / velocity[0]
        return t_x

    t_missile_to_target = time_to_target(M1_start, v_missile_vector, fake_target)
    
    if verbose:
        print(f"导弹到达假目标的时间: {t_missile_to_target:.2f} s")

    # 计算导弹在时间t的位置
    def missile_position(t):
        return M1_start + v_missile_vector * t

    # 计算烟云在时间t的位置（自爆炸以来）
    def smoke1_position(t_smoke):
        return np.array([blast_position1[0], blast_position1[1], blast_position1[2] - v_smoke_sink * t_smoke])
    def smoke2_position(t_smoke):
        return np.array([blast_position2[0], blast_position2[1], blast_position2[2] - v_smoke_sink * t_smoke])
    def smoke3_position(t_smoke):
        return np.array([blast_position3[0], blast_position3[1], blast_position3[2] - v_smoke_sink * t_smoke])

    # 角度条件：确定烟幕是否在导弹和真实目标之间
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

    # 计算点到线的距离
    def distance_point_to_line(point, line_point, line_direction):
        ap = point - line_point
        projection = np.dot(ap, line_direction) / np.linalg.norm(line_direction)
        foot_point = line_point + projection * line_direction
        return np.linalg.norm(point - foot_point)

    # 确定导弹是否穿过烟云
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

    # 计算有效遮蔽时间
    start_time = 0
    end_time = min(effective_duration, t_missile_to_target - t_blast1 - t_blast2 - t_blast3)
    
    if end_time <= 0:
        return 0.0
    
    time_step = 0.01
    total_effective_time = 0.0
    current_time = start_time
    
    # 存储前一位置用于穿越检测
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
        
        # Check angle condition
        is_between = is_smoke1_between(pos_missile, pos_smoke1, real_target) + is_smoke2_between(pos_missile, pos_smoke2, real_target) + is_smoke3_between(pos_missile, pos_smoke3, real_target)
        
        # Check if missile passes through smoke cloud
        is_through = is_missile_through_smoke1(pos_missile, pos_smoke1, prev_missile_pos, prev_smoke1_pos) + is_missile_through_smoke2(pos_missile, pos_smoke2, prev_missile_pos, prev_smoke2_pos) + is_missile_through_smoke3(pos_missile, pos_smoke3, prev_missile_pos, prev_smoke3_pos)
        
        # Check if missile is inside the cloud
        in_smoke1 = np.linalg.norm(pos_missile - pos_smoke1) <= effective_radius
        in_smoke2 = np.linalg.norm(pos_missile - pos_smoke2) <= effective_radius
        in_smoke3 = np.linalg.norm(pos_missile - pos_smoke3) <= effective_radius
        in_smoke = in_smoke1 + in_smoke2 + in_smoke3
        
        if (distance1 <= effective_radius and is_between) or (distance2 <= effective_radius and is_between) or (distance3 <= effective_radius and is_between) or is_through or in_smoke:
            total_effective_time += time_step
        
        # 更新前一位置
        prev_missile_pos = pos_missile
        prev_smoke1_pos = pos_smoke1
        prev_smoke2_pos = pos_smoke2
        prev_smoke3_pos = pos_smoke3
        
        current_time += time_step
    
    if verbose:
        print(f"\n=== 计算结果 ===")
        print(f"烟幕弹1投放时间: {t_drop1:.1f} s")
        print(f"烟幕弹1爆炸时间: {t_blast1:.1f} s")
        print(f"1爆炸位置: ({blast_position1[0]:.1f}, {blast_position1[1]:.1f}, {blast_position1[2]:.1f})")
        print(f"烟幕弹2投放时间: {t_drop2:.1f} s")
        print(f"烟幕弹2爆炸时间: {t_blast2:.1f} s")
        print(f"2爆炸位置: ({blast_position2[0]:.1f}, {blast_position2[1]:.1f}, {blast_position2[2]:.1f})")
        print(f"烟幕弹3投放时间: {t_drop3:.1f} s")
        print(f"烟幕弹3爆炸时间: {t_blast3:.1f} s")
        print(f"3爆炸位置: ({blast_position3[0]:.1f}, {blast_position3[1]:.1f}, {blast_position3[2]:.1f})")
        print(f"导弹到达假目标的时间: {t_missile_to_target:.2f} s")
        print(f"有效遮蔽持续时间: {total_effective_time:.3f} 秒")

        # 关键时间点的详细分析
        print(f"\n=== 关键时间点的详细分析 ===")
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
                
                print(f"{t_check}s 后爆炸:")
                print(f"  导弹位置: {pos_missile}")
                print(f"  云1位置: {pos_smoke1}")
                print(f"  云2位置: {pos_smoke2}")
                print(f"  云3位置: {pos_smoke3}")
                print(f"  到视线1距离: {distance1:.2f}m")
                print(f"  到视线2距离: {distance2:.2f}m")
                print(f"  到视线3距离: {distance3:.2f}m")
                print(f"  直接距离1: {direct_distance1:.2f}m")
                print(f"  直接距离2: {direct_distance2:.2f}m")
                print(f"  直接距离3: {direct_distance3:.2f}m")
                print(f"  烟幕在导弹和目标之间: {'是' if is_between else '否'}")
                print(f"  导弹-烟幕1-目标角度: {angle_deg1:.1f}°")
                print(f"  导弹-烟幕2-目标角度: {angle_deg2:.1f}°")
                print(f"  导弹-烟幕3-目标角度: {angle_deg3:.1f}°")
                print(f"  在云1内: {'是' if direct_distance1 <= effective_radius else '否'}")
                print(f"  在云2内: {'是' if direct_distance2 <= effective_radius else '否'}")
                print(f"  在云3内: {'是' if direct_distance3 <= effective_radius else '否'}")
                print(f"  有效遮蔽: {'是' if ((distance1 <= effective_radius and is_between1) or direct_distance1 <= effective_radius) or ((distance2 <= effective_radius and is_between2) or direct_distance2 <= effective_radius) or ((distance3 <= effective_radius and is_between3) or direct_distance3 <= effective_radius) else '否'}")

    if visualize:
        # 可视化
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制轨迹
        missile_times = np.linspace(0, t_missile_to_target, 100)
        missile_traj = np.array([missile_position(t) for t in missile_times])
        ax.plot(missile_traj[:, 0], missile_traj[:, 1], missile_traj[:, 2], 'r-', label='导弹轨迹', linewidth=2)

        # 绘制烟幕1下降轨迹
        smoke1_times = np.linspace(0, min(effective_duration, t_missile_to_target - t_blast1), 50)
        smoke1_traj = np.array([smoke1_position(t) for t in smoke1_times])
        ax.plot(smoke1_traj[:, 0], smoke1_traj[:, 1], smoke1_traj[:, 2], 'b-', label='烟幕下降轨迹', linewidth=2)
        # 绘制烟幕2下降轨迹
        smoke2_times = np.linspace(0, min(effective_duration, t_missile_to_target - t_blast2), 50)
        smoke2_traj = np.array([smoke2_position(t) for t in smoke2_times])
        ax.plot(smoke2_traj[:, 0], smoke2_traj[:, 1], smoke2_traj[:, 2], 'g-', label='烟幕下降轨迹', linewidth=2)
        # 绘制烟幕3下降轨迹 
        smoke3_times = np.linspace(0, min(effective_duration, t_missile_to_target - t_blast3), 50)
        smoke3_traj = np.array([smoke3_position(t) for t in smoke3_times])
        ax.plot(smoke3_traj[:, 0], smoke3_traj[:, 1], smoke3_traj[:, 2], 'm-', label='烟幕下降轨迹', linewidth=2)

        # 标记关键点
        ax.scatter(*fake_target, color='red', s=200, label='假目标', marker='x')
        ax.scatter(*real_target, color='blue', s=200, label='真实目标', marker='o')
        ax.scatter(*M1_start, color='orange', s=100, label='导弹起点')
        ax.scatter(*blast_position1, color='green', s=100, label='爆炸点')
        ax.scatter(*blast_position2, color='purple', s=100, label='爆炸点')
        ax.scatter(*blast_position3, color='cyan', s=100, label='爆炸点')

        # 设置视图和标签
        ax.view_init(elev=20, azim=45)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'导弹M1的烟幕干扰分析\n有效遮蔽时间: {total_effective_time:.3f}s')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    return total_effective_time


# effective_time = calculate_smoke_obscuration(
#         drone_direction=[-0.5, -0.5, 0],  # towards fake target
#         drone_speed=110,
#         drop_time=20,
#         blast_delay=7,
#         visualize=True,
#         verbose=False
#     )
import numpy as np
import tqdm
import random


# 自定义遗传算法实现
class GeneticAlgorithm:
    def __init__(self, pop_size=100, crossover_rate=0.7, mutation_rate=0.2, generations=50):
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        
        # 参数范围
        self.param_ranges = [
            (0, 3600),   # i1 范围
            (80, 120),  # j1 范围
            (0, 15),    # q1 范围
            (0, 15),    # s1 范围
            (0, 3600),   # i2 范围
            (80, 120),  # j2 范围
            (0, 15),    # q2 范围
            (0, 15),    # s2 范围
            (0, 3600),   # i3 范围
            (80, 120),  # j3 范围
            (0, 15),    # q3 范围
            (0, 15)     # s3 范围
        ]
    
    def create_individual(self):
        # 创建个体（参数集）
        individual = []
        for param_range in self.param_ranges:
            individual.append(random.randint(param_range[0], param_range[1] - 1))
        return individual
    
    def create_population(self):
        # 创建初始种群
        return [self.create_individual() for _ in range(self.pop_size)]
    
    def evaluate(self, individual):
        # 评估个体适应度
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
        return result  # 返回适应度值
    
    def select(self, population, fitnesses):
        # 锦标赛选择
        selected = []
        for _ in range(self.pop_size):
            # 随机选择3个个体进行竞争
            candidates = random.sample(list(zip(population, fitnesses)), 3)
            # 选择适应度最高的个体
            winner = max(candidates, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected
    
    def crossover(self, parent1, parent2):
        # 单点交叉
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1, parent2
    
    def mutate(self, individual):
        # 均匀变异
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                low, high = self.param_ranges[i]
                mutated[i] = random.randint(low, high - 1)
        return mutated
    
    def run(self):
        # 创建初始种群
        population = self.create_population()
        
        # 存储所有有效解
        solutions = []
        
        print("开始遗传算法优化...")
        for gen in tqdm.tqdm(range(self.generations)):
            # 评估种群中所有个体
            fitnesses = [self.evaluate(ind) for ind in population]
            
            # 记录有效解
            for i, fit in enumerate(fitnesses):
                if fit != 0:
                    solutions.append((population[i], fit))
            
            # 选择
            selected = self.select(population, fitnesses)
            
            # 交叉和变异
            next_population = []
            for i in range(0, self.pop_size, 2):
                parent1 = selected[i]
                parent2 = selected[i+1] if i+1 < self.pop_size else selected[0]
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                next_population.extend([child1, child2])
            
            # 确保种群大小保持不变
            population = next_population[:self.pop_size]
        
        # 移除重复解
        unique_solutions = []
        seen = set()
        for sol, fit in solutions:
            key = tuple(sol)
            if key not in seen:
                seen.add(key)
                unique_solutions.append((sol, fit))
        
        return unique_solutions

def main():
    # 创建遗传算法实例并运行
    ga = GeneticAlgorithm(pop_size=100, generations=50)
    solutions = ga.run()
    
    # 输出结果
    if solutions:
        print(f"\n找到 {len(solutions)} 个有效解:")
        for i, (sol, fit) in enumerate(solutions, 1):
            print(f"解 {i}: i1={sol[0]}, j1={sol[1]}, q1={sol[2]}, s1={sol[3]}, i2={sol[4]}, j2={sol[5]}, q2={sol[6]}, s2={sol[7]},i3={sol[8]}, j3={sol[9]}, q3={sol[10]}, s3={sol[11]},结果={fit}")
    else:
        print("\n未找到有效解")

if __name__ == "__main__":
    main()

