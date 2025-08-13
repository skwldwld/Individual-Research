import open3d as o3d
import numpy as np
import os

# -----------------------------
# Utils
# -----------------------------
def find_ceiling_indices(points_np, axis_idx, thickness):
    """
    높이축(axis_idx)의 최댓값 부근(천장 밴드) 인덱스 반환.
    퍼센타일을 100으로 두어 최상단 밴드를 안전하게 잡음.
    """
    if points_np.size == 0:
        return np.array([], dtype=int)
    max_val = np.percentile(points_np[:, axis_idx], 100)
    mask = (points_np[:, axis_idx] >= max_val - thickness) & (points_np[:, axis_idx] <= max_val + 1e-6)
    return np.where(mask)[0].astype(int)

def find_wall_indices_z(points_np, thickness):
    """
    Z축의 최소/최대 경계 밴드(벽) 인덱스 반환.
    """
    if points_np.size == 0:
        return np.array([], dtype=int)
    z = points_np[:, 2]
    min_z, max_z = np.min(z), np.max(z)
    mask_min = (z >= min_z) & (z <= min_z + thickness)
    mask_max = (z >= max_z - thickness) & (z <= max_z)
    return np.where(mask_min | mask_max)[0].astype(int)

def axis_to_index(height_axis):
    if isinstance(height_axis, str):
        return {'x':0, 'y':1, 'z':2}.get(height_axis.lower(), 2)
    return int(height_axis)

# -----------------------------
# Main
# -----------------------------
def remove_walls_and_ceiling(
    pcd_path,
    output_dir,
    wall_thickness=5.0,
    ceiling_thickness=5.0,
    voxel_size=0.03,
    height_axis='z'  # 천장 판단에 쓸 높이축: 'z' 또는 'y' (또는 0/1/2)
):
    print(f"[INFO] Loading point cloud: {pcd_path}")
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
    except Exception as e:
        print(f"[ERROR] read_point_cloud failed: {e}")
        return

    if not pcd.has_points():
        print("[ERROR] Loaded point cloud has no points.")
        return

    os.makedirs(output_dir, exist_ok=True)

    pts = np.asarray(pcd.points)
    h_idx = axis_to_index(height_axis)

    # 1) 인덱스 계산
    wall_idx = find_wall_indices_z(pts, wall_thickness)
    ceil_idx = find_ceiling_indices(pts, h_idx, ceiling_thickness)

    # 2) 분리
    removed_wall_pcd = pcd.select_by_index(wall_idx) if len(wall_idx) > 0 else o3d.geometry.PointCloud()
    removed_ceiling_pcd = pcd.select_by_index(ceil_idx) if len(ceil_idx) > 0 else o3d.geometry.PointCloud()

    remove_union = np.union1d(wall_idx, ceil_idx)
    keep_idx = np.setdiff1d(np.arange(len(pts)), remove_union)
    kept_pcd = pcd.select_by_index(keep_idx) if len(keep_idx) > 0 else o3d.geometry.PointCloud()

    # 3) 색 / 다운샘플
    if len(kept_pcd.points) > 0:
        kept_pcd.paint_uniform_color([0.2, 0.8, 0.2])  # 초록
        if voxel_size and voxel_size > 0:
            kept_pcd = kept_pcd.voxel_down_sample(voxel_size=float(voxel_size))
    if len(removed_wall_pcd.points) > 0:
        removed_wall_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 빨강
    if len(removed_ceiling_pcd.points) > 0:
        removed_ceiling_pcd.paint_uniform_color([0.0, 0.4, 1.0])  # 파랑 (저장 X, 시각화만)

    # 4) 저장: kept + 벽만 저장, 천장은 저장 안 함
    out_kept = os.path.join(output_dir, "nonowall.pcd")           # 최종 보존본
    out_removed_walls = os.path.join(output_dir, "removed_walls.pcd")  # 제거된 벽

    try:
        if len(kept_pcd.points) > 0:
            o3d.io.write_point_cloud(out_kept, kept_pcd)
            print(f"[SUCCESS] Saved kept: {out_kept} ({len(kept_pcd.points)} pts)")
        else:
            print("[WARN] No points left after removal; kept cloud is empty. (not saved)")

        if len(removed_wall_pcd.points) > 0:
            o3d.io.write_point_cloud(out_removed_walls, removed_wall_pcd)
            print(f"[SUCCESS] Saved removed walls: {out_removed_walls} ({len(removed_wall_pcd.points)} pts)")
        else:
            print("[INFO] No wall points detected within thickness; removed_walls not saved.")
    except Exception as e:
        print(f"[ERROR] Error saving point clouds: {e}")

    # 5) 시각화: 확인용 (천장은 파랑으로만 표시, 파일 저장 없음)
    geoms = [g for g in [kept_pcd, removed_wall_pcd, removed_ceiling_pcd] if g.has_points()]
    if geoms:
        o3d.visualization.draw_geometries(
            geoms,
            window_name="Removed (Green: Kept, Red: Walls Saved, Blue: Ceiling Deleted)"
        )
    else:
        print("[INFO] Nothing to visualize.")

def main():
    # input_file = "../input/3BP_CS_model_Cloud.pcd"
    input_file = "../input/6pp_testbed.pcd"
    output_directory = "../output"
    remove_walls_and_ceiling(
        pcd_path=input_file,
        output_dir=output_directory,
        wall_thickness=9.0,      # 벽 두께(m)
        ceiling_thickness=9.0,   # 천장 두께(m)
        voxel_size=0.03,
        height_axis='y'          # 데이터가 Y-up이면 'y', Z-up이면 'z'
    )

if __name__ == "__main__":
    main()
