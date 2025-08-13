import open3d as o3d
import numpy as np
import argparse
import os
from typing import Tuple

# -----------------------------
# Utils
# -----------------------------
def axis_to_index(height_axis):
    if isinstance(height_axis, str):
        return {'x': 0, 'y': 1, 'z': 2}.get(height_axis.lower(), 2)
    return int(height_axis)

def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    if voxel is None or voxel <= 0:
        return pcd
    return pcd.voxel_down_sample(float(voxel))

def estimate_normals_inplace(pcd: o3d.geometry.PointCloud, radius: float, max_nn: int = 60):
    if len(pcd.points) == 0:
        return
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.normalize_normals()

def find_ceiling_indices(points_np: np.ndarray, axis_idx: int, thickness: float) -> np.ndarray:
    """
    높이축 최댓값 부근(천장 밴드) 인덱스 반환.
    """
    if points_np.size == 0:
        return np.array([], dtype=int)
    h = points_np[:, axis_idx]
    max_val = np.max(h)  # 굴곡 바닥이어도 천장은 '최대' 부근
    mask = (h >= max_val - thickness) & (h <= max_val + 1e-6)
    return np.where(mask)[0].astype(int)

def bounds_xy(points_np: np.ndarray) -> Tuple[float, float, float, float]:
    """
    X, Z(또는 X, Y) 경계 반환. (x_min, x_max, z_min, z_max)
    """
    x = points_np[:, 0]
    z = points_np[:, 2]
    return float(np.min(x)), float(np.max(x)), float(np.min(z)), float(np.max(z))

def find_wall_indices_by_verticality_and_band(
    points_np: np.ndarray,
    normals_np: np.ndarray,
    up_idx: int,
    band_thickness: float,
    verticality_deg: float = 20.0,
) -> np.ndarray:
    """
    '벽' = (1) 수직면: 법선의 up 성분이 작음 + (2) 외곽 밴드(x/z min/max 근처).
    굴곡진 바닥/벽 형태에서도 직사각형 가정 없이 동작 가능.
    """
    if points_np.size == 0:
        return np.array([], dtype=int)

    # 1) 수직성 필터: |n_up| < sin(theta)
    # up 성분이 0이면 완전 수직면. 허용 각도를 키우면 더 느슨해짐.
    n_up = np.abs(normals_np[:, up_idx])
    s = np.sin(np.deg2rad(verticality_deg))
    vertical_mask = (n_up <= s)

    # 2) 외곽 밴드(x/z min/max 근처) 필터
    x = points_np[:, 0]
    z = points_np[:, 2]
    x_min, x_max, z_min, z_max = bounds_xy(points_np)
    near_x_band = (x <= x_min + band_thickness) | (x >= x_max - band_thickness)
    near_z_band = (z <= z_min + band_thickness) | (z >= z_max - band_thickness)
    band_mask = near_x_band | near_z_band

    indices = np.where(vertical_mask & band_mask)[0].astype(int)
    return indices

def paint_if_any(pcd: o3d.geometry.PointCloud, rgb):
    if len(pcd.points) > 0:
        pcd.paint_uniform_color(rgb)

# -----------------------------
# Main
# -----------------------------
def remove_walls_and_ceiling(
    pcd_path: str,
    output_dir: str,
    wall_thickness: float = 0.15,        # m 단위: 외곽 밴드 폭
    ceiling_thickness: float = 0.20,     # m 단위: 최상단 밴드 두께
    voxel_size: float = 0.03,            # m 단위: 선행 다운샘플
    height_axis: str = 'y',              # 'y' 또는 'z' (또는 0/1/2)
    wall_verticality_deg: float = 20.0,  # 벽 수직성 허용 각도(작을수록 수직에 가까운 면만 벽으로 인식)
    visualize: bool = True,
):
    print(f"[INFO] Loading point cloud: {pcd_path}")
    try:
        pcd_raw = o3d.io.read_point_cloud(pcd_path)
    except Exception as e:
        print(f"[ERROR] read_point_cloud failed: {e}")
        return

    if not pcd_raw.has_points():
        print("[ERROR] Loaded point cloud has no points.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 0) 선행 다운샘플 (포인트 수가 많을수록 먼저 줄이는 게 핵심)
    if voxel_size and voxel_size > 0:
        print(f"[INFO] Pre-downsample with voxel={voxel_size}")
        pcd = voxel_downsample(pcd_raw, voxel_size)
    else:
        pcd = pcd_raw

    pts = np.asarray(pcd.points)
    if pts.size == 0:
        print("[ERROR] Empty after downsample.")
        return

    up_idx = axis_to_index(height_axis)

    # 1) 노멀 추정 (벽 수직성 판정용)
    #    반경은 복셀의 2.5~4배 권장. 데이터가 조밀하면 늘리고, 성기면 줄이기.
    normal_radius = max(voxel_size * 3.0, 0.05) if voxel_size else 0.1
    estimate_normals_inplace(pcd, radius=normal_radius, max_nn=60)
    normals = np.asarray(pcd.normals)
    if normals.shape[0] != pts.shape[0]:
        print("[WARN] Normal estimation failed or mismatched; fallback to boundary-only walls.")
        normals = np.zeros_like(pts)
        normals[:, up_idx] = 0.0  # 수직으로 가정

    # 2) 벽/천장 인덱스 계산
    wall_idx = find_wall_indices_by_verticality_and_band(
        pts, normals, up_idx=up_idx, band_thickness=wall_thickness, verticality_deg=wall_verticality_deg
    )
    ceil_idx = find_ceiling_indices(pts, up_idx, thickness=ceiling_thickness)

    # 3) 분리
    removed_wall_pcd = pcd.select_by_index(wall_idx) if len(wall_idx) > 0 else o3d.geometry.PointCloud()
    removed_ceiling_pcd = pcd.select_by_index(ceil_idx) if len(ceil_idx) > 0 else o3d.geometry.PointCloud()

    remove_union = np.union1d(wall_idx, ceil_idx)
    keep_idx = np.setdiff1d(np.arange(len(pts)), remove_union)
    kept_pcd = pcd.select_by_index(keep_idx) if len(keep_idx) > 0 else o3d.geometry.PointCloud()

    # 4) 색상 표시
    paint_if_any(kept_pcd, [0.2, 0.8, 0.2])        # 초록
    paint_if_any(removed_wall_pcd, [1.0, 0.0, 0.0]) # 빨강
    paint_if_any(removed_ceiling_pcd, [0.0, 0.4, 1.0])  # 파랑 (저장 X)

    # 5) 저장 (천장은 저장하지 않음)
    out_kept = os.path.join(output_dir, "nonowall.pcd")
    out_removed_walls = os.path.join(output_dir, "removed_walls.pcd")
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
            print("[INFO] No wall points detected; removed_walls not saved.")
    except Exception as e:
        print(f"[ERROR] Error saving point clouds: {e}")

    # 6) 시각화
    if visualize:
        geoms = [g for g in [kept_pcd, removed_wall_pcd, removed_ceiling_pcd] if g.has_points()]
        if geoms:
            o3d.visualization.draw_geometries(
                geoms,
                window_name="Removed (Green: Kept, Red: Walls Saved, Blue: Ceiling Deleted)"
            )
        else:
            print("[INFO] Nothing to visualize.")

def parse_args():
    p = argparse.ArgumentParser(description="Remove walls (vertical + boundary band) and ceiling from a PCD/PLY.")
    p.add_argument("--input", "-i", type=str, required=True, help="Input PCD/PLY path")
    p.add_argument("--out", "-o", type=str, default="../output", help="Output directory")
    p.add_argument("--voxel", type=float, default=0.03, help="Pre-downsample voxel size (m)")
    p.add_argument("--height-axis", type=str, default="y", help="Up axis: x|y|z or 0|1|2 (default: y)")
    p.add_argument("--wall-thickness", type=float, default=0.15, help="Boundary band thickness for walls (m)")
    p.add_argument("--wall-verticality-deg", type=float, default=20.0, help="Verticality tolerance for walls (deg)")
    p.add_argument("--ceiling-thickness", type=float, required=True, help="Ceiling band thickness near top (m)")
    p.add_argument("--no-vis", action="store_true", help="Disable Open3D viewer")
    return p.parse_args()

def main():
    args = parse_args()
    # height_axis 정규화
    try:
        height_axis = int(args.height_axis)
    except:
        height_axis = args.height_axis

    remove_walls_and_ceiling(
        pcd_path=args.input,
        output_dir=args.out,
        wall_thickness=args.wall_thickness,
        ceiling_thickness=args.ceiling_thickness,
        voxel_size=args.voxel,
        height_axis=height_axis,
        wall_verticality_deg=args.wall_verticality_deg,
        visualize=(not args.no_vis),
    )

if __name__ == "__main__":
    main()
