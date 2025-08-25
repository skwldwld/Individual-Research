# remove.py  (강화본: 일반형 벽 제거 + 노멀 게이트 + 프리 다운샘플)
import os
import argparse
import numpy as np
import open3d as o3d
import cv2
import math

# -----------------------
# 유틸
# -----------------------
def axis_to_index(height_axis):
    if isinstance(height_axis, str):
        return {'x': 0, 'y': 1, 'z': 2}.get(height_axis.lower(), 2)
    return int(height_axis)

def find_ceiling(points_np, axis_idx, thickness):
    """상부 천장 띠 제거(높이축 기준 최상단 근처)."""
    if points_np.size == 0:
        return np.array([], dtype=int)
    y = points_np[:, axis_idx]
    ymax = np.percentile(y, 100)
    mask = (y >= ymax - thickness) & (y <= ymax + 1e-6)
    return np.where(mask)[0].astype(int)

def _grid_from_points_xz(points_np, grid_cell):
    """점들을 x–z 평면에 래스터화하기 위한 그리드 좌표와 메타데이터를 만든다."""
    x = points_np[:, 0]; z = points_np[:, 2]
    xmin, zmin = np.min(x), np.min(z)
    ix = np.floor((x - xmin) / grid_cell).astype(np.int32)
    iz = np.floor((z - zmin) / grid_cell).astype(np.int32)
    W = int(max(1, ix.max() + 1)); H = int(max(1, iz.max() + 1))
    return ix, iz, W, H, xmin, zmin

def find_walls_general(points_np, band_m=0.15, grid_cell=0.05,
                       morph_k=3, min_area_px=20, method="erosion"):
    """
    x–z 평면 기준 외곽 '띠'를 벽으로 간주.
    method:
      - "dist"   : 거리변환 기반
      - "erosion": 침식으로 내부 코어를 만들고 (mask - core)를 띠로 사용(권장)
    반환: 벽에 해당하는 포인트 인덱스(np.int64)
    """
    if points_np.size == 0:
        return np.array([], dtype=np.int64)

    ix, iz, W, H, xmin, zmin = _grid_from_points_xz(points_np, grid_cell)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[iz, ix] = 255

    if morph_k and morph_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            keep[labels == i] = 255
    mask = keep
    if mask.max() == 0:
        return np.array([], dtype=np.int64)

    band_px = max(1, int(round(band_m / max(1e-6, grid_cell))))

    if method == "erosion":
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * band_px + 1, 2 * band_px + 1))
        core = cv2.erode(mask, k)
        wall_cells = (mask > 0) & (core == 0)
    else:
        dist = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=3)
        wall_cells = (mask > 0) & (dist <= band_px)

    linear = iz.astype(np.int64) * W + ix.astype(np.int64)
    wall_linear = np.where(wall_cells.reshape(-1))[0]
    wall_linear_set = set(wall_linear.tolist())
    sel_idx = np.where([(li in wall_linear_set) for li in linear])[0].astype(np.int64)
    return sel_idx

def vertical_indices_by_normals(points_np, up_axis='z', angle_deg=20.0, k_nn=30):
    """
    Up축과의 각도가 90°±angle_deg 인 점들(=수직 벽)을 골라 인덱스를 반환.
    """
    if points_np.size == 0:
        return np.array([], dtype=np.int64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max(10, int(k_nn))))
    n = np.asarray(pcd.normals, dtype=np.float32)

    up = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[str(up_axis).lower()]
    up = np.asarray(up, dtype=np.float32)
    s = math.sin(math.radians(max(1e-3, float(angle_deg))))
    mask = (np.abs((n * up).sum(axis=1)) <= s)
    return np.where(mask)[0].astype(np.int64)

def _random_downsample_pcd(pcd: o3d.geometry.PointCloud, target_n: int) -> o3d.geometry.PointCloud:
    n = np.asarray(pcd.points).shape[0]
    if n <= target_n:
        return pcd
    idx = np.random.choice(n, size=target_n, replace=False)
    return pcd.select_by_index(idx.tolist())

# -----------------------
# 메인 로직
# -----------------------
def remove_cloud(pcd_path, output_dir,
                 wall_band=0.15, grid_cell=0.05, morph_k=3, min_area_px=20,
                 band_method="erosion",
                 only_vertical=False, vertical_angle_deg=20.0, vertical_k_nn=30,
                 ceiling_thickness=0.10, height_axis='z',
                 pre_voxel_size=0.0, target_points=0, every_k=0,
                 post_voxel=0.03, visualize=True):
    print(f"[INFO] Loading point cloud: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_points():
        raise RuntimeError("Loaded point cloud has no points.")
    pcd.remove_non_finite_points()
    print(f"[STATS] loaded points = {len(pcd.points)}")

    # 프리 다운샘플
    if pre_voxel_size and pre_voxel_size > 0:
        pcd = pcd.voxel_down_sample(float(pre_voxel_size))
        print(f"[STATS] after pre-voxel({pre_voxel_size}) = {len(pcd.points)}")
    if target_points and target_points > 0:
        before = len(pcd.points)
        pcd = _random_downsample_pcd(pcd, int(target_points))
        print(f"[STATS] after target-points({target_points}) {before}->{len(pcd.points)}")
    if every_k and every_k > 1:
        before = len(pcd.points)
        pcd = pcd.uniform_down_sample(int(every_k))
        print(f"[STATS] after every-k({every_k}) {before}->{len(pcd.points)}")

    pts = np.asarray(pcd.points)

    # 벽 인덱스
    wall_idx = find_walls_general(
        pts, band_m=float(wall_band), grid_cell=float(grid_cell),
        morph_k=int(morph_k), min_area_px=int(min_area_px),
        method=str(band_method)
    )

    # 수직 노멀 게이트
    if only_vertical and wall_idx.size > 0:
        v_idx = vertical_indices_by_normals(pts, up_axis=height_axis,
                                            angle_deg=float(vertical_angle_deg),
                                            k_nn=int(vertical_k_nn))
        wall_idx = np.intersect1d(wall_idx, v_idx, assume_unique=False)
    print(f"[STATS] wall indices = {len(wall_idx)}")

    # 천장 인덱스
    h_idx = axis_to_index(height_axis)
    ceil_idx = find_ceiling(pts, h_idx, float(ceiling_thickness))
    print(f"[STATS] ceiling indices = {len(ceil_idx)}")

    # 분리
    removed_wall_pcd   = pcd.select_by_index(wall_idx) if wall_idx.size > 0 else o3d.geometry.PointCloud()
    removed_ceiling_pc = pcd.select_by_index(ceil_idx) if ceil_idx.size > 0 else o3d.geometry.PointCloud()
    remove_union = np.union1d(wall_idx, ceil_idx)
    keep_idx = np.setdiff1d(np.arange(len(pts), dtype=np.int64), remove_union)
    kept_pcd = pcd.select_by_index(keep_idx) if keep_idx.size > 0 else o3d.geometry.PointCloud()

    # 후처리 복셀
    if kept_pcd.has_points() and post_voxel and post_voxel > 0:
        kept_pcd = kept_pcd.voxel_down_sample(float(post_voxel))
        print(f"[STATS] kept after voxel({post_voxel}) = {len(kept_pcd.points)}")

    # 색상
    if kept_pcd.has_points():           kept_pcd.paint_uniform_color([0.2, 0.8, 0.2])
    if removed_wall_pcd.has_points():   removed_wall_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    if removed_ceiling_pc.has_points(): removed_ceiling_pc.paint_uniform_color([0.0, 0.4, 1.0])

    # 저장
    os.makedirs(output_dir, exist_ok=True)
    out_kept = os.path.join(output_dir, "nonowall.pcd")
    out_removed_walls = os.path.join(output_dir, "removed_walls.pcd")

    if kept_pcd.has_points():
        o3d.io.write_point_cloud(out_kept, kept_pcd, write_ascii=False, compressed=True)
        print(f"[SUCCESS] Saved kept: {out_kept} ({len(kept_pcd.points)} pts)")
    else:
        print("[WARN] No points left after removal; kept cloud is empty. (not saved)")

    if removed_wall_pcd.has_points():
        o3d.io.write_point_cloud(out_removed_walls, removed_wall_pcd, write_ascii=False, compressed=True)
        print(f"[SUCCESS] Saved removed walls: {out_removed_walls} ({len(removed_wall_pcd.points)} pts)")
    else:
        print("[INFO] No wall points detected within band; removed_walls not saved.")

    # 시각화
    if visualize:
        geoms = [g for g in [kept_pcd, removed_wall_pcd, removed_ceiling_pc] if g.has_points()]
        if geoms:
            o3d.visualization.draw_geometries(
                geoms,
                window_name="Removed (Green: Kept, Red: Walls Saved, Blue: Ceiling Deleted)"
            )
        else:
            print("[INFO] Nothing to visualize.")

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="General wall/ceiling removal with shape-agnostic band + optional normal gating.")
    p.add_argument("--input", required=True, help="입력 PCD/PLY/OBJ 경로")
    p.add_argument("--out", required=True, help="출력 디렉토리")

    # 벽(일반형)
    p.add_argument("--wall-band", type=float, default=0.15, help="경계에서 제거할 벽 두께(m)")
    p.add_argument("--grid-cell", type=float, default=0.05, help="래스터 셀 크기(m)")
    p.add_argument("--morph-k", type=int, default=3, help="모폴로지 커널(픽셀)")
    p.add_argument("--min-area-px", type=int, default=20, help="유지할 최소 컴포넌트 면적(픽셀)")
    p.add_argument("--band-method", choices=["erosion", "dist"], default="erosion", help="벽 띠 계산 방식")

    # 수직 노멀 게이트
    p.add_argument("--only-vertical", action="store_true", help="수직 벽 성분만 제거(노멀 기반)")
    p.add_argument("--vertical-angle-deg", type=float, default=20.0, help="Up축과 90°±각 허용치")
    p.add_argument("--vertical-k-nn", type=int, default=30, help="노멀 추정 KNN")

    # 천장/축
    p.add_argument("--ceiling-thickness", type=float, default=0.10, help="천장 두께(m)")
    p.add_argument("--height-axis", choices=["x", "y", "z"], default="z")

    # 속도/품질
    p.add_argument("--pre-voxel-size", type=float, default=0.0, help="로드 직후 복셀 다운샘플")
    p.add_argument("--target-points", type=int, default=0, help="최대 남길 포인트 수(랜덤 샘플)")
    p.add_argument("--every-k", type=int, default=0, help="균등 다운샘플 간격(K)")
    p.add_argument("--post-voxel", type=float, default=0.03, help="남긴 포인트 복셀 다운샘플")

    # 표시
    p.add_argument("--no-vis", action="store_true", help="시각화 끔")

    # 하위호환(기존 --wall-thickness를 --wall-band로 매핑)
    p.add_argument("--wall-thickness", type=float, default=None, help="(deprecated) 직사각 전제 두께 → wall-band로 매핑")
    return p.parse_args()

def main():
    args = parse_args()
    wall_band = args.wall_band if args.wall_thickness is None else float(args.wall_thickness)

    print(f"[REMOVE] input={os.path.abspath(args.input)}")
    print(f"[REMOVE] out={os.path.abspath(args.out)}")
    print(f"[REMOVE] wall_band={wall_band} m, grid_cell={args.grid_cell} m, morph_k={args.morph_k}, min_area_px={args.min_area_px}, band_method={args.band_method}")
    print(f"[REMOVE] only_vertical={args.only_vertical}, vertical_angle={args.vertical_angle_deg} deg, vertical_knn={args.vertical_k_nn}")
    print(f"[REMOVE] ceil_thick={args.ceiling_thickness} m, up={args.height_axis}")
    print(f"[REMOVE] pre_voxel={args.pre_voxel_size}, target_points={args.target_points}, every_k={args.every_k}, post_voxel={args.post_voxel}")

    remove_cloud(
        pcd_path=args.input,
        output_dir=args.out,
        wall_band=wall_band,
        grid_cell=args.grid_cell,
        morph_k=args.morph_k,
        min_area_px=args.min_area_px,
        band_method=args.band_method,
        only_vertical=args.only_vertical,
        vertical_angle_deg=args.vertical_angle_deg,
        vertical_k_nn=args.vertical_k_nn,
        ceiling_thickness=args.ceiling_thickness,
        height_axis=args.height_axis,
        pre_voxel_size=args.pre_voxel_size,
        target_points=args.target_points,
        every_k=args.every_k,
        post_voxel=args.post_voxel,
        visualize=(not args.no_vis),
    )

if __name__ == "__main__":
    main()
