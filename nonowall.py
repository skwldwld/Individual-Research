import open3d as o3d
import numpy as np
import os


def remove_box_region(points_np, min_bound, max_bound):
    """
    주어진 박스 경계 안에 포함되는 점들의 인덱스를 반환.
    """
    inside_mask = np.all((points_np >= min_bound) & (points_np <= max_bound), axis=1)
    return np.where(inside_mask)[0]


def remove_walls_by_box(
    pcd_path, output_dir,
    boundary_thickness=5.0,
    voxel_size=0.03
):
    print(f"Loading point cloud from: {pcd_path}")
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return

    if not pcd.has_points():
        print("Loaded point cloud has no points.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    points_np = np.asarray(pcd.points)
    min_x, max_x = np.min(points_np[:, 0]), np.max(points_np[:, 0])
    min_y, max_y = np.min(points_np[:, 1]), np.max(points_np[:, 1])
    min_z, max_z = np.min(points_np[:, 2]), np.max(points_np[:, 2])

    box_candidates = {
        'x+': remove_box_region(points_np, [max_x - boundary_thickness, min_y, min_z], [max_x, max_y, max_z]),
        'x-': remove_box_region(points_np, [min_x, min_y, min_z], [min_x + boundary_thickness, max_y, max_z]),
        'z+': remove_box_region(points_np, [min_x, min_y, max_z - boundary_thickness], [max_x, max_y, max_z]),
        'z-': remove_box_region(points_np, [min_x, min_y, min_z], [max_x, max_y, min_z + boundary_thickness]),
    }

    wall_counts = {k: len(v) for k, v in box_candidates.items()}
    top2 = sorted(wall_counts, key=wall_counts.get, reverse=True)[:2]
    print(f"Top 2 wall directions (most dense): {top2}")

    wall_points_indices = np.unique(np.concatenate([box_candidates[k] for k in top2]))
    non_wall_indices = np.setdiff1d(np.arange(len(points_np)), wall_points_indices)

    non_wall_pcd = pcd.select_by_index(non_wall_indices)
    removed_pcd = pcd.select_by_index(wall_points_indices)
    non_wall_pcd.paint_uniform_color([0.2, 0.8, 0.2])
    removed_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    non_wall_pcd = non_wall_pcd.voxel_down_sample(voxel_size=voxel_size)
    output_file_path = os.path.join(output_dir, "nonowall.pcd")

    try:
        o3d.io.write_point_cloud(output_file_path, non_wall_pcd)
        print(f"✅ Saved: {output_file_path} ({len(non_wall_pcd.points)} points)")
    except Exception as e:
        print(f"❌ Error saving cleaned point cloud: {e}")

    o3d.visualization.draw_geometries(
        [non_wall_pcd, removed_pcd],
        window_name="Box Filtered (Green: Cleaned, Red: Removed)"
    )


def main():
    input_file = "input/3BP_CS_model_Cloud.pcd"
    output_directory = "output"
    remove_walls_by_box(
        pcd_path=input_file,
        output_dir=output_directory,
        boundary_thickness=9.0,  # 박스 두께
        voxel_size=0.03
    )


if __name__ == "__main__":
    main()
