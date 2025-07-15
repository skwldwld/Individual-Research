import open3d as o3d
import numpy as np
import os

def remove_walls(
    pcd_path, output_dir,
    voxel_size=0.03,
    normal_radius_factor=3,
    normal_max_nn=80,
    wall_angle_threshold_deg=25, # 벽 수직성 임계값 더 증가
    boundary_thickness=5.0, # ⭐⭐⭐이 값을 크게 증가⭐⭐⭐
    floor_ceiling_tolerance=0.0 # 바닥/천장 제거 비활성화
):
    """
    RANSAC 없이 포인트 클라우드에서 벽면을 추정하고 제거합니다. (Y축이 높이 축인 경우)
    """
    print(f"Loading point cloud from: {pcd_path}")
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        print("Please ensure the file path is correct and it's a valid .pcd or .ply file.")
        return

    if not pcd.has_points():
        print("Loaded point cloud has no points. Exiting.")
        return

    print(f"Loaded point cloud with {len(pcd.points)} points.")
    original_pcd = pcd
    original_total_points = len(original_pcd.points)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print("\n--- Starting Wall Estimation and Removal (No RANSAC - Y-axis as height) ---")
    print(f"  Voxel Size for Normals: {voxel_size}")
    print(f"  Normal Search Radius: {voxel_size * normal_radius_factor:.3f}")
    print(f"  Normal Max Neighbors: {normal_max_nn}")
    print(f"  Wall Angle Threshold (Y-axis): {wall_angle_threshold_deg:.1f} degrees")
    print(f"  Boundary Thickness for Outer Walls: {boundary_thickness:.3f} meters")
    print(f"  Floor/Ceiling Tolerance: {floor_ceiling_tolerance:.3f} meters (Disabled if 0)")

    pcd_for_normals = original_pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Downsampled to {len(pcd_for_normals.points)} points for normal calculation.")

    search_radius = voxel_size * normal_radius_factor
    pcd_for_normals.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=normal_max_nn))
    # Y축 방향으로 법선 정렬
    pcd_for_normals.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 1., 0.])) 

    points_np = np.asarray(original_pcd.points)
    normals_np_downsampled = np.asarray(pcd_for_normals.normals)

    print("Mapping normals to original point cloud...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_for_normals)
    original_normals_mapped = np.zeros_like(points_np)
    mappable_indices = []
    for i, p in enumerate(points_np):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(p, 1) 
        if idx and np.linalg.norm(normals_np_downsampled[idx[0]]) > 0:
            original_normals_mapped[i] = normals_np_downsampled[idx[0]]
            mappable_indices.append(i)

    min_x, max_x = np.min(points_np[:, 0]), np.max(points_np[:, 0])
    min_y, max_y = np.min(points_np[:, 1]), np.max(points_np[:, 1])
    min_z, max_z = np.min(points_np[:, 2]), np.max(points_np[:, 2])
    print(f"  Model Bounding Box: X:[{min_x:.2f}, {max_x:.2f}], Y:[{min_y:.2f}, {max_y:.2f}], Z:[{min_z:.2f}, {max_z:.2f}]")

    wall_points_indices = []
    count_angle_ok = 0
    count_alignment_ok = 0
    count_boundary_ok = 0
    count_all_ok = 0

    for i in mappable_indices:
        point = points_np[i]
        normal = original_normals_mapped[i]
        normal = normal / np.linalg.norm(normal)
        angle_with_y = np.degrees(np.arccos(np.abs(np.dot(normal, np.array([0, 1, 0])))))
        is_angle_ok = abs(angle_with_y - 90) <= wall_angle_threshold_deg
        if is_angle_ok:
            count_angle_ok += 1
        normal_alignment_threshold = np.cos(np.radians(90 - wall_angle_threshold_deg)) 
        is_alignment_ok = (np.abs(normal[0]) > normal_alignment_threshold) or \
                          (np.abs(normal[2]) > normal_alignment_threshold) # ⭐Z축으로 변경⭐
        if is_alignment_ok:
            count_alignment_ok += 1
        is_boundary_ok = ((abs(point[0] - min_x) < boundary_thickness) or \
                          (abs(point[0] - max_x) < boundary_thickness) or \
                          (abs(point[2] - min_z) < boundary_thickness) or \
                          (abs(point[2] - max_z) < boundary_thickness)) # ⭐Z축으로 변경⭐
        if is_boundary_ok:
            count_boundary_ok += 1
        if is_angle_ok and is_alignment_ok and is_boundary_ok:
            wall_points_indices.append(i)
            count_all_ok += 1

    print(f"\n--- Debugging Counts ---")
    print(f"  Points with correct angle to Y-axis (is_angle_ok): {count_angle_ok}")
    print(f"  Points with normal aligned to X/Z axis (is_alignment_ok): {count_alignment_ok}")
    print(f"  Points within boundary thickness (is_boundary_ok): {count_boundary_ok}")
    print(f"  Points satisfying ALL conditions (count_all_ok): {count_all_ok}")
    print(f"Identified {len(wall_points_indices)} potential wall points.")

    all_removed_indices = np.unique(np.array(wall_points_indices))
    non_wall_indices = np.setdiff1d(np.arange(original_total_points), all_removed_indices)
    non_wall_pcd = original_pcd.select_by_index(non_wall_indices)
    non_wall_pcd.paint_uniform_color([0.2, 0.8, 0.2])

    # ⭐⭐⭐ 다운샘플링 추가 ⭐⭐⭐
    down_voxel_size = 0.03  # 다운샘플링 voxel 크기(필요시 조절)
    non_wall_pcd = non_wall_pcd.voxel_down_sample(voxel_size=down_voxel_size)
    print(f"Downsampled cleaned point cloud: {len(non_wall_pcd.points)} points")

    removed_pcd = original_pcd.select_by_index(all_removed_indices)
    removed_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    output_file_path = os.path.join(output_dir, "nonowall.pcd")
    try:
        o3d.io.write_point_cloud(output_file_path, non_wall_pcd)
        print(f"✅ Cleaned point cloud saved: {output_file_path} ({len(non_wall_pcd.points)} points)")
    except Exception as e:
        print(f"❌ Error saving cleaned point cloud: {e}")

    print(f"Total points removed: {len(all_removed_indices)}")
    print(f"Remaining points: {len(non_wall_pcd.points)}")

    print("\nVisualizing Final Result (Green: Cleaned, Red: Removed)")
    o3d.visualization.draw_geometries([non_wall_pcd, removed_pcd],
                                      window_name="Cleaned vs Removed (No RANSAC - Y-axis as height)")

def main():
    input_file = "input/3BP_CS_model_Cloud.pcd"
    output_directory = "output" # 출력 폴더명 변경
    remove_walls(
        pcd_path=input_file,
        output_dir=output_directory,
        voxel_size=0.03,
        normal_radius_factor=3,
        normal_max_nn=80,
        wall_angle_threshold_deg=25, # 벽면 수직성 허용 오차 (20 -> 25)
        boundary_thickness=7.5, # ⭐이 값을 크게 증가시켰습니다 (0.4 -> 5.0). 건물 크기에 맞춰 조절⭐
        floor_ceiling_tolerance=0.0 # 바닥/천장 제거 비활성화
    )

if __name__ == "__main__":
    main()