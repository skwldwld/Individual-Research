import open3d as o3d
import numpy as np
import os

def remove_z_wall(points_np, min_z, max_z, boundary_thickness):
    # z축 경계 영역에 있는 점 인덱스 반환
    wall_minus = np.where((points_np[:, 2] >= min_z) & (points_np[:, 2] <= min_z + boundary_thickness))[0]
    wall_plus  = np.where((points_np[:, 2] >= max_z - boundary_thickness) & (points_np[:, 2] <= max_z))[0]
    return np.concatenate([wall_minus, wall_plus])

def remove_walls_z(
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
    min_z, max_z = np.min(points_np[:, 2]), np.max(points_np[:, 2])

    wall_indices = remove_z_wall(points_np, min_z, max_z, boundary_thickness)
    non_wall_indices = np.setdiff1d(np.arange(len(points_np)), wall_indices)

    non_wall_pcd = pcd.select_by_index(non_wall_indices)
    removed_pcd = pcd.select_by_index(wall_indices)
    non_wall_pcd.paint_uniform_color([0.2, 0.8, 0.2])
    removed_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    non_wall_pcd = non_wall_pcd.voxel_down_sample(voxel_size=voxel_size)
    output_file_path = os.path.join(output_dir, "nonowall.pcd")

    try:
        o3d.io.write_point_cloud(output_file_path, non_wall_pcd)
        print(f"[SUCCESS] Saved: {output_file_path} ({len(non_wall_pcd.points)} points)")
    except Exception as e:
        print(f"[ERROR] Error saving cleaned point cloud: {e}")

    o3d.visualization.draw_geometries(
        [non_wall_pcd, removed_pcd],
        window_name="Z Wall Filtered (Green: Cleaned, Red: Removed)"
    )

def main():
    input_file = "../input/3BP_CS_model_Cloud.pcd"
    output_directory = "../output"
    remove_walls_z(
        pcd_path=input_file,
        output_dir=output_directory,
        boundary_thickness=9.0,  # 벽 두께(m)
        voxel_size=0.03
    )

if __name__ == "__main__":
    main()
