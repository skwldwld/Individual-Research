# remove_rect.py  (네모형 전제 버전: 내용 유지 + '벽 축/개수' 옵션 추가)
import open3d as o3d
import numpy as np
import os
import argparse

def find_ceiling(points_np, axis_idx, thickness):
    """
    높이축(axis_idx)의 최댓값 부근(천장 밴드) 인덱스 반환.
    퍼센타일을 100으로 두어 최상단 밴드를 안전하게 잡음.
    """
    if points_np.size == 0:
        return np.array([], dtype=int)
    max_val = np.percentile(points_np[:, axis_idx], 100)
    mask = (points_np[:, axis_idx] >= max_val - thickness) & (points_np[:, axis_idx] <= max_val + 1e-6)
    return np.where(mask)[0].astype(int)

def _axis_to_index(axis):
    if isinstance(axis, str):
        return {'x':0, 'y':1, 'z':2}.get(axis.lower(), 2)
    return int(axis)

def find_wall(points_np, thickness, wall_axis='z', sides='both'):
    """
    직사각형 가정: 선택한 '벽 축'(x/y/z)의 최소/최대 경계 밴드를 벽으로 간주.
    sides: 'both' | 'min' | 'max'
    """
    if points_np.size == 0:
        return np.array([], dtype=int)
    aidx = _axis_to_index(wall_axis)
    vals = points_np[:, aidx]
    vmin, vmax = np.min(vals), np.max(vals)

    use_min = (sides in ('both', 'min'))
    use_max = (sides in ('both', 'max'))

    mask_min = (vals >= vmin) & (vals <= vmin + thickness) if use_min else np.zeros_like(vals, dtype=bool)
    mask_max = (vals >= vmax - thickness) & (vals <= vmax)     if use_max else np.zeros_like(vals, dtype=bool)

    return np.where(mask_min | mask_max)[0].astype(int)

def axis_to_index(height_axis):
    if isinstance(height_axis, str):
        return {'x':0, 'y':1, 'z':2}.get(height_axis.lower(), 2)
    return int(height_axis)

def remove(
    pcd_path,
    output_dir,
    wall_thickness=5.0,
    ceiling_thickness=5.0,
    voxel_size=0.03,
    height_axis='z',   # 천장 판단 축
    wall_axis='z',     # 벽 경계 축 (신규)
    wall_sides='both'  # 'both' | 'min' | 'max' (신규)
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
    wall_idx = find_wall(pts, wall_thickness, wall_axis=wall_axis, sides=wall_sides)
    ceil_idx = find_ceiling(pts, h_idx, ceiling_thickness)

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

# ==== CLI 래퍼 ====
def _parse_args():
    p = argparse.ArgumentParser(description="Rect-room wall/ceiling removal (내용 유지, 옵션만 확장).")
    p.add_argument("--input", required=True, help="입력 PCD/PLY/OBJ 경로")
    p.add_argument("--out", required=True, help="출력 디렉토리")
    p.add_argument("--wall-thickness", type=float, default=5.0, help="경계 벽 두께(m)")
    p.add_argument("--ceiling-thickness", type=float, default=5.0, help="천장 두께(m)")
    p.add_argument("--voxel", type=float, default=0.03, help="다운샘플 복셀 크기(m)")
    p.add_argument("--height-axis", choices=["x","y","z"], default="z", help="천장 판단 축")
    # 신규
    p.add_argument("--wall-axis", choices=["x","y","z"], default="z", help="벽 경계 축")
    p.add_argument("--wall-sides", choices=["both","min","max"], default="both", help="제거할 벽: 양쪽/최소쪽/최대쪽")
    return p.parse_args()

def _main():
    args = _parse_args()
    print(f"[REMOVE-RECT] input={os.path.abspath(args.input)}")
    print(f"[REMOVE-RECT] out={os.path.abspath(args.out)}")
    print(f"[REMOVE-RECT] wall={args.wall_thickness}, ceil={args.ceiling_thickness}, voxel={args.voxel}, up={args.height_axis}, wall_axis={args.wall_axis}, wall_sides={args.wall_sides}")
    remove(
        pcd_path=args.input,
        output_dir=args.out,
        wall_thickness=args.wall_thickness,
        ceiling_thickness=args.ceiling_thickness,
        voxel_size=args.voxel,
        height_axis=args.height_axis,
        wall_axis=args.wall_axis,
        wall_sides=args.wall_sides
    )

if __name__ == "__main__":
    _main()
