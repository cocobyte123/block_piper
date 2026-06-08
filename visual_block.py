import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon

class BlockVisualizer:
    """一个专门用于可视化积木搭建过程的类，支持3D和正面双视图。"""
    def __init__(self, instances, prototypes, architecture, build_origin, gripper_offset=None):
        """【核心修改】新增 gripper_offset 参数。"""
        self.instances = instances
        self.prototypes = prototypes
        self.architecture = architecture
        self.build_origin = build_origin
        self.gripper_offset = gripper_offset  # 存储夹爪偏移
        self.colors = plt.cm.YlOrRd(np.linspace(0.3, 1, len(self.prototypes)))
        self.type_to_color = {t: c for t, c in zip(self.prototypes.keys(), self.colors)}

    def _draw_block(self, ax_3d, ax_2d, block_id, color, alpha, zorder):
        """【核心修复】回归经典绘图方式，一次性绘制填充和边框。"""
        instance = self.instances[block_id]
        center = instance["place_pos"] - self.build_origin
        size = self.prototypes[instance["type"]]["size"]
        angle_rad = instance.get("place_angle_rad", 0.0)
        
        dx, dy, dz = size / 2.0
        
        # --- 3D 绘图 ---
        vertices_3d = np.array([
            [-dx, -dy, -dz], [ dx, -dy, -dz], [ dx,  dy, -dz], [-dx,  dy, -dz],
            [-dx, -dy,  dz], [ dx, -dy,  dz], [ dx,  dy,  dz], [-dx,  dy,  dz]
        ])
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0], [np.sin(angle_rad), np.cos(angle_rad), 0], [0, 0, 1]])
        rotated_vertices_3d = vertices_3d @ rotation_matrix.T + center
        
        faces_3d = [
            [rotated_vertices_3d[0], rotated_vertices_3d[1], rotated_vertices_3d[2], rotated_vertices_3d[3]], [rotated_vertices_3d[4], rotated_vertices_3d[5], rotated_vertices_3d[6], rotated_vertices_3d[7]],
            [rotated_vertices_3d[0], rotated_vertices_3d[1], rotated_vertices_3d[5], rotated_vertices_3d[4]], [rotated_vertices_3d[2], rotated_vertices_3d[3], rotated_vertices_3d[7], rotated_vertices_3d[6]],
            [rotated_vertices_3d[1], rotated_vertices_3d[2], rotated_vertices_3d[6], rotated_vertices_3d[5]], [rotated_vertices_3d[4], rotated_vertices_3d[7], rotated_vertices_3d[3], rotated_vertices_3d[0]]
        ]
        # 【核心修复】在一次调用中同时定义填充和边框，彻底解决颜色覆盖问题
        ax_3d.add_collection3d(Poly3DCollection(
            faces_3d, 
            facecolors=color, 
            linewidths=1, 
            edgecolors='k', 
            alpha=alpha, 
            zorder=zorder
        ))

        # --- 2D 正面视图绘图 (只看X-Z平面) ---
        front_face_vertices_2d = np.array([
            [center[0] - dx, center[2] - dz], [center[0] + dx, center[2] - dz],
            [center[0] + dx, center[2] + dz], [center[0] - dx, center[2] + dz]
        ])
        patch = Polygon(front_face_vertices_2d, facecolor=color, edgecolor='k', linewidth=1, alpha=alpha, zorder=zorder)
        ax_2d.add_patch(patch)

    def _create_figure(self, main_title):
        """创建包含3D和2D子图的画布。"""
        fig = plt.figure(figsize=(16, 8))
        ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
        ax_2d = fig.add_subplot(1, 2, 2)
        fig.suptitle(main_title, fontsize=16)
        return fig, ax_3d, ax_2d

    def visualize_blueprint(self, save_path):
        """【修改】可视化最终蓝图，保持纯净，不显示任何点位。"""
        print(f"[Visualizer] 正在生成纯净蓝图: {save_path}")
        fig, ax_3d, ax_2d = self._create_figure("Final Build Blueprint")

        all_block_ids = [item for sublist in self.architecture.values() for item in sublist]
        for i, block_id in enumerate(all_block_ids):
            instance = self.instances[block_id]
            color = self.type_to_color[instance["type"]]
            self._draw_block(ax_3d, ax_2d, block_id, color=color, alpha=0.9, zorder=i)

        self._set_ax_limits_and_labels(ax_3d, ax_2d)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def visualize_placement_strategy_for_layer(self, layer_index, save_path):
        """【核心修改】只绘制一个最终的预放置目标点，并保留撤离箭头。"""
        layer_key = f"layer_{layer_index}"
        if layer_key not in self.architecture: return
            
        print(f"[Visualizer] 正在生成第 {layer_index} 层策略图: {save_path}")
        fig, ax_3d, ax_2d = self._create_figure(f"Layer {layer_index} Placement Strategy")

        target_layer_blocks = self.architecture[layer_key]
        blocks_to_draw = [b for i in range(1, layer_index + 1) for b in self.architecture.get(f"layer_{i}", [])]

        for i, block_id in enumerate(blocks_to_draw):
            is_target = block_id in target_layer_blocks
            alpha = 0.9 if is_target else 0.15
            zorder_offset = len(blocks_to_draw) if is_target else 0
            color = self.type_to_color[self.instances[block_id]["type"]] if is_target else 'gray'
            self._draw_block(ax_3d, ax_2d, block_id, color=color, alpha=alpha, zorder=i + zorder_offset)
        
        for block_id in target_layer_blocks:
            instance = self.instances[block_id]
            
            # --- 【核心修改】计算并绘制唯一的预放置目标点 ---
            if self.gripper_offset is not None and "place_pos" in instance:
                # 1. 获取积木的放置中心点
                block_place_center = instance["place_pos"] - self.build_origin
                
                # 2. 计算预放置高度的偏移
                pre_place_height_offset = np.array([0, 0, instance["pre_place_height"]])
                
                # 3. 计算最终的预放置目标点 (积木中心 + 预留高度 + 夹爪偏移)
                final_pre_place_target = block_place_center + pre_place_height_offset + self.gripper_offset
                
                # 4. 在3D和2D视图中绘制这个点
                z_order_target_point = len(blocks_to_draw) * 4 # 确保在最顶层
                for ax in [ax_3d, ax_2d]:
                    coords = (final_pre_place_target[0], final_pre_place_target[1], final_pre_place_target[2]) if ax == ax_3d else (final_pre_place_target[0], final_pre_place_target[2])
                    ax.scatter(*coords, c='red', s=120, marker='*', edgecolors='black', zorder=z_order_target_point, label="Pre-place Target")

            # --- 保留撤离方向的箭头 ---
            place_pos_rel = instance["place_pos"] - self.build_origin
            lift_dir, lift_dist = instance["lift_direction"], instance["lift_distance"]
            lift_offset = np.array([0,0,lift_dist]) if lift_dir=="up" else np.array([-lift_dist,0,0]) if lift_dir=="left" else np.array([lift_dist,0,0])
            release_pos_rel = place_pos_rel + np.array([0,0,0.03])

            z_order_arrow = len(blocks_to_draw) * 3
            # 3D 箭头
            ax_3d.quiver(release_pos_rel[0], release_pos_rel[1], release_pos_rel[2], lift_offset[0], lift_offset[1], lift_offset[2], color='blue', length=lift_dist, normalize=True, zorder=z_order_arrow)
            # 2D 箭头
            ax_2d.arrow(release_pos_rel[0], release_pos_rel[2], lift_offset[0], lift_offset[2], head_width=0.005, head_length=0.01, fc='blue', ec='blue', zorder=z_order_arrow)

        # 添加图例以避免混淆
        handles, labels = ax_3d.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            fig.legend(by_label.values(), by_label.keys(), loc='upper right')

        self._set_ax_limits_and_labels(ax_3d, ax_2d)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    def _set_ax_limits_and_labels(self, ax_3d, ax_2d):
        """【核心修改】分离2D和3D的缩放逻辑，并提供手动调整参数。"""
        
        # --- 【手动调整区域】 ---
        # 你可以在这里调整 padding 来控制视图的缩放。
        # padding 越大，视图中的物体看起来越小（留白越多）。
        # padding 越小，视图中的物体看起来越大（留白越少）。
        padding_3d = 0.5  # 3D视图的留白比例，建议值 0.1 ~ 0.5
        padding_2d = 0.1  # 2D视图的留白大小(米)，建议值 0.05 ~ 0.2
        # --- 【手动调整区域结束】 ---

        all_coords = [i["place_pos"] - self.build_origin for i in self.instances.values() if "place_pos" in i]
        if not all_coords: return
        
        all_coords = np.array(all_coords)
        min_coords = all_coords.min(axis=0)
        max_coords = all_coords.max(axis=0)
        mid_coords = (min_coords + max_coords) / 2.0
        ranges = max_coords - min_coords

        # --- 3D 视图缩放逻辑 ---
        # 找到最大的轴范围，并以此为基准创建一个立方体，再增加padding
        max_range_3d = ranges.max() * (1 + padding_3d)
        ax_3d.set_xlim(mid_coords[0] - max_range_3d / 2, mid_coords[0] + max_range_3d / 2)
        ax_3d.set_ylim(mid_coords[1] - max_range_3d / 2, mid_coords[1] + max_range_3d / 2)
        ax_3d.set_zlim(mid_coords[2] - max_range_3d / 2, mid_coords[2] + max_range_3d / 2)
        ax_3d.set_xlabel("X"); ax_3d.set_ylabel("Y"); ax_3d.set_zlabel("Z")
        ax_3d.set_title("3D Perspective View")
        
        # --- 2D 视图缩放逻辑 ---
        # 分别计算X和Z的范围，并各自增加padding
        ax_2d.set_xlim(min_coords[0] - padding_2d, max_coords[0] + padding_2d)
        ax_2d.set_ylim(min_coords[2] - padding_2d, max_coords[2] + padding_2d)
        ax_2d.set_xlabel("X"); ax_2d.set_ylabel("Z")
        ax_2d.set_title("2D Front View (Height Check)")
        ax_2d.set_aspect('equal', adjustable='box')
        ax_2d.grid(True, linestyle='--', alpha=0.6)