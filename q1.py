import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import os
from datetime import datetime
from matplotlib.lines import Line2D

# 配置字体（优先使用 DejaVu Sans 以支持广泛字符）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = "战场图片"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 生成文件名时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 创建包含两个子图的图形
fig = plt.figure(figsize=(16, 8))

# 第一个子图：全景视图
ax1 = fig.add_subplot(121, projection='3d')

# 定义坐标数据
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

# 绘制圆柱体
def draw_cylinder(ax, center, radius, height, color, alpha=0.6):
    circle_bottom = Circle((center[0], center[1]), radius, color=color, alpha=alpha)
    ax.add_patch(circle_bottom)
    art3d.pathpatch_2d_to_3d(circle_bottom, z=center[2], zdir="z")
    
    circle_top = Circle((center[0], center[1]), radius, color=color, alpha=alpha)
    ax.add_patch(circle_top)
    art3d.pathpatch_2d_to_3d(circle_top, z=center[2] + height, zdir="z")
    
    return circle_bottom

# 绘制诱饵目标（红色圆柱）——以高度区分
fake_cylinder = draw_cylinder(ax1, fake_target, 10, 15, 'red', 0.9)
ax1.text(fake_target[0], fake_target[1], fake_target[2] + 20, '诱饵目标', 
    color='red', fontsize=14, ha='center', fontweight='bold', 
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# 绘制真实目标（蓝色圆柱）——以高度区分
real_cylinder = draw_cylinder(ax1, real_target, 10, 20, 'blue', 0.9)
ax1.text(real_target[0], real_target[1], real_target[2] + 25, '真实目标', 
    color='blue', fontsize=14, ha='center', fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# 绘制导弹（红色三角）
for name, pos in missiles.items():
    ax1.scatter(pos[0], pos[1], pos[2], c='red', marker='^', s=300,
               edgecolors='black', linewidth=2)
    # 调整标签位置，使其靠近标记
    label_offset = 500
    ax1.text(pos[0] + label_offset, pos[1] + label_offset, pos[2] + label_offset, name, 
            color='red', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

# 绘制无人机（绿色正方形）
for name, pos in drones.items():
    ax1.scatter(pos[0], pos[1], pos[2], c='green', marker='s', s=200,
               edgecolors='black', linewidth=1.5)
    # 调整标签位置，使其靠近标记
    label_offset = 500
    ax1.text(pos[0] + label_offset, pos[1] + label_offset, pos[2] + label_offset, name, 
            color='green', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

# 绘制导弹飞行方向箭头
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

# 调整 ax1 视角并添加简短说明
ax1.view_init(elev=20, azim=-60)
ax1.text2D(0.02, 0.95, 
          "情景:\n"
          "• 红色圆柱: 诱饵目标\n"
          "• 蓝色圆柱: 真实目标\n"
          "• 红色三角: 导弹\n"
          "• 绿色正方: 无人机\n"
          "• 橙色箭头: 飞行方向", 
          transform=ax1.transAxes, fontsize=10, verticalalignment='top',
          bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", 
                   alpha=0.8, edgecolor='gold'))

# 第二个子图：目标区域放大
ax2 = fig.add_subplot(122, projection='3d')

# 绘制目标区域放大
draw_cylinder(ax2, fake_target, 10, 15, 'red', 0.9)
draw_cylinder(ax2, real_target, 10, 20, 'blue', 0.9)

ax2.text(fake_target[0], fake_target[1], fake_target[2] + 18, '诱饵目标', 
         color='red', fontsize=14, ha='center', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
ax2.text(real_target[0], real_target[1], real_target[2] + 23, '真实目标', 
         color='blue', fontsize=14, ha='center', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Set unequal axis ranges for the close-up view
ax2.set_xlim([-50, 250])
ax2.set_ylim([-50, 250])
ax2.set_zlim([-10, 60])

ax2.set_xlabel('X 坐标 (m)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Y 坐标 (m)', fontsize=12, fontweight='bold')
ax2.set_zlabel('Z 坐标 (m)', fontsize=12, fontweight='bold')
ax2.set_title('目标区域放大\n（诱饵与真实目标高度差）', fontsize=14, fontweight='bold')

ax2.grid(True, alpha=0.4)
ax2.view_init(elev=35, azim=-45)

# Add distance annotation
ax2.plot([0, 0], [0, 200], [15, 20], 'k--', alpha=0.6)
ax2.text(10, 100, 18, '200米', fontsize=11, fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

plt.tight_layout()

# 保存图片
save_path_png = os.path.join(output_dir, f"战场三维地图_{timestamp}.png")
save_path_pdf = os.path.join(output_dir, f"战场三维地图_{timestamp}.pdf")

plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')

print(f"图片已保存到目录: {output_dir}")
print(f"PNG 文件: 战场三维地图_{timestamp}.png")
print(f"PDF 文件: 战场三维地图_{timestamp}.pdf")

# 显示图像（在无头后端可能为非交互式）
plt.show()

 # 保存信息
print(f"\n文件已保存！")
print(f"生成的图片位于文件夹 '{output_dir}'")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义常量
g = 9.8
v_missile = 300
v_smoke_sink = 3
effective_radius = 10
effective_duration = 20

# 目标位置
fake_target = np.array([0, 0, 0])
real_target = np.array([0, 200, 0])

# 导弹 M1 起始位置
M1_start = np.array([20000, 0, 2000])

# 无人机 FY1 起始位置
FY1_start = np.array([17800, 0, 1800])

# 无人机速度
v_drone = 120
direction_to_target = fake_target[:2] - FY1_start[:2]
direction_to_target = direction_to_target / np.linalg.norm(direction_to_target)
v_drone_vector = np.array([v_drone * direction_to_target[0], 
                          v_drone * direction_to_target[1], 0])

print(f"无人机速度向量: {v_drone_vector}")

# 时间参数
t_drop = 1.5
t_blast_delay = 3.6
t_blast = t_drop + t_blast_delay

# Compute positions
drop_position = FY1_start + v_drone_vector * t_drop
vertical_drop = 0.5 * g * t_blast_delay**2
blast_position = np.array([drop_position[0]+v_drone_vector[0]*t_blast_delay, drop_position[1], drop_position[2] - vertical_drop])

print(f"投放位置: {drop_position}")
print(f"起爆位置: {blast_position}")

# 导弹飞行方向（指向诱饵目标）
missile_direction = fake_target - M1_start
missile_direction = missile_direction / np.linalg.norm(missile_direction)
v_missile_vector = v_missile * missile_direction
print(missile_direction, v_missile_vector)
print(f"导弹速度向量: {v_missile_vector}")
print(f"导弹起始位置: {M1_start}")

# Calculate time for missile to reach decoy target
def time_to_target(position, velocity, target):
    t_x = (target[0] - position[0]) / velocity[0]
    return t_x

t_missile_to_target = time_to_target(M1_start, v_missile_vector, fake_target)
print(f"导弹到诱饵目标的时间: {t_missile_to_target:.2f} s")

# Compute missile position at time t
def missile_position(t):
    return M1_start + v_missile_vector * t

# Compute smoke cloud position at time t (since blast)
def smoke_position(t_smoke):
    return np.array([blast_position[0], blast_position[1], blast_position[2] - v_smoke_sink * t_smoke])

# Angle condition: determine if smoke is between missile and real target
def is_smoke_between(missile_pos, smoke_pos, target_pos):
    """
    判断烟幕是否位于导弹和目标之间。
    如果导弹-烟幕-目标夹角为锐角，则烟幕在两者之间。
    如果夹角为钝角，则烟幕在目标后方（无效）。
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
    判断导弹轨迹线段是否与烟幕球体相交。
    使用相对运动（导弹减去烟幕）简化相交测试。
    """
    # 烟幕云半径
    radius = effective_radius

    # 导弹移动向量
    missile_move = missile_pos - missile_prev_pos
    missile_move_length = np.linalg.norm(missile_move)

    if missile_move_length == 0:
        return False

    # 烟幕移动向量
    smoke_move = smoke_pos - smoke_prev_pos

    # 使用相对运动简化计算
    relative_move = missile_move - smoke_move
    relative_move_length = np.linalg.norm(relative_move)

    if relative_move_length == 0:
        # 相对静止：检查当前距离
        return np.linalg.norm(missile_pos - smoke_pos) <= radius

    # 计算导弹相对烟幕的运动方向
    relative_direction = relative_move / relative_move_length

    # 上一时刻烟幕到导弹的向量
    smoke_to_missile = missile_prev_pos - smoke_prev_pos

    # 计算投影长度
    projection = np.dot(smoke_to_missile, relative_direction)

    # 计算到烟幕的垂直距离
    perpendicular_dist = np.linalg.norm(smoke_to_missile - projection * relative_direction)

    # 如果垂直距离小于等于半径且投影在运动范围内，则可能相交
    if perpendicular_dist <= radius and 0 <= projection <= relative_move_length:
        return True

    # 检查起点或终点是否在云团内
    if np.linalg.norm(missile_prev_pos - smoke_prev_pos) <= radius:
        return True
    if np.linalg.norm(missile_pos - smoke_pos) <= radius:
        return True

    return False

# 计算有效遮蔽时间
def calculate_effective_time():
    start_time = 0
    end_time = min(effective_duration, t_missile_to_target - t_blast)
    
    if end_time <= 0:
        return 0.0
    
    time_step = 0.01  # 稍微增大步长以提升性能
    total_effective_time = 0.0
    current_time = start_time
    
    # 存储前一时刻的位置用于穿越检测
    prev_missile_pos = missile_position(t_blast)
    prev_smoke_pos = smoke_position(0)
    
    while current_time <= end_time:
        t_missile = t_blast + current_time
        if t_missile > t_missile_to_target:
            break
            
        pos_missile = missile_position(t_missile)
        pos_smoke = smoke_position(current_time)
        


        # 计算距离
        missile_to_target_direction = real_target - pos_missile
        missile_to_target_direction = missile_to_target_direction / np.linalg.norm(missile_to_target_direction)
        distance = distance_point_to_line(pos_smoke, pos_missile, missile_to_target_direction)
        # 检查夹角条件：烟幕必须在导弹和目标之间
        is_between = is_smoke_between(pos_missile, pos_smoke, real_target)

        # 检查导弹是否穿越烟幕云团
        is_through = is_missile_through_smoke(pos_missile, pos_smoke, prev_missile_pos, prev_smoke_pos)

        # 以下三种情况计入有效遮蔽时间：
        # 1. 到视线距离 <= 有效半径 且 烟幕位于导弹与目标之间（标准遮蔽）
        # 2. 导弹穿越烟幕云团（穿越情况）
        # 3. 导弹直接位于烟幕云团内部
        in_smoke = np.linalg.norm(pos_missile - pos_smoke) <= effective_radius

        if (distance <= effective_radius and is_between) or is_through or in_smoke:
            total_effective_time += time_step

        # 更新前一时刻的位置
        prev_missile_pos = pos_missile
        prev_smoke_pos = pos_smoke
        
        current_time += time_step
    
    return total_effective_time

 # 计算并输出结果
effective_time = calculate_effective_time()
print(f"\n=== 计算结果 ===")
print(f"烟幕投放时间: {t_drop:.1f} s")
print(f"烟幕起爆时间: {t_blast:.1f} s")
print(f"起爆位置: ({blast_position[0]:.1f}, {blast_position[1]:.1f}, {blast_position[2]:.1f})")
print(f"导弹到诱饵目标时间: {t_missile_to_target:.2f} s")
print(f"有效遮蔽持续时间: {effective_time:.3f} s")

 # 关键时间点详细分析
print(f"\n=== 关键时间点详细分析 ===")
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
        
        print(f"{t_check}s 起爆后:")
        print(f"  导弹位置: {pos_missile}")
        print(f"  烟幕位置: {pos_smoke}")
        print(f"  到视线距离: {distance:.2f} m")
        print(f"  直接距离: {direct_distance:.2f} m")
        print(f"  烟幕在导弹和目标之间: {'是' if is_between else '否'}")
        print(f"  导弹-烟幕-目标角度: {angle_deg:.1f}°")
        print(f"  是否在云团内: {'是' if direct_distance <= effective_radius else '否'}")
        print(f"  有效遮蔽: {'是' if (distance <= effective_radius and is_between) or direct_distance <= effective_radius else '否'}")

 # 可视化
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

 # 绘制轨迹
missile_times = np.linspace(0, t_missile_to_target, 100)
missile_traj = np.array([missile_position(t) for t in missile_times])
ax.plot(missile_traj[:, 0], missile_traj[:, 1], missile_traj[:, 2], 'r-', label='导弹轨迹', linewidth=2)

 # 绘制烟幕下降轨迹
smoke_times = np.linspace(0, min(effective_duration, t_missile_to_target - t_blast), 50)
smoke_traj = np.array([smoke_position(t) for t in smoke_times])
ax.plot(smoke_traj[:, 0], smoke_traj[:, 1], smoke_traj[:, 2], 'b-', label='烟幕下降轨迹', linewidth=2)

 # 标记关键点
ax.scatter(*fake_target, color='red', s=200, label='诱饵目标', marker='x')
ax.scatter(*real_target, color='blue', s=200, label='真实目标', marker='o')
ax.scatter(*M1_start, color='orange', s=100, label='导弹起点')
ax.scatter(*blast_position, color='green', s=100, label='起爆点')

 # 在关键时刻绘制烟幕云团
for t in [0, 10, 20]:
    if t < len(smoke_times):
        smoke_pos = smoke_position(t)
        # 绘制球体表示烟幕云团
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = smoke_pos[0] + effective_radius * np.outer(np.cos(u), np.sin(v))
        y = smoke_pos[1] + effective_radius * np.outer(np.sin(u), np.sin(v))
        z = smoke_pos[2] + effective_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='cyan', alpha=0.2)

 # 设置视角和标签
ax.view_init(elev=20, azim=45)
ax.set_xlabel('X 轴 (米)')
ax.set_ylabel('Y 轴 (米)')
ax.set_zlabel('Z 轴 (米)')
ax.set_title('导弹 M1 的烟幕干扰分析（含穿越检测）')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()
