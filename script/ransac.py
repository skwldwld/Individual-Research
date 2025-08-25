# ransac.py
import open3d as o3d
import numpy as np
import os

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _angle_deg_to_axis(n, axis):
    n = _unit(np.asarray(n, float))
    axis = _unit(np.asarray(axis, float))
    c = float(np.clip(abs(np.dot(n, axis)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def _split_by_plane_offset(original, plane, offset_m):
    """평면(법선 n, 절편 d) 기준 부호 있는 거리로 컷한다."""
    a, b, c, d = map(float, plane[:4])
    n = np.array([a, b, c], dtype=float)
    n = _unit(n)
    if np.linalg.norm(n) == 0:
        idx_all = np.arange(len(original.points))
        return idx_all, np.array([], dtype=np.int64)
    pts = np.asarray(original.points)
    signed = pts @ n + d  # n·p + d, n은 단위벡터
    lower_idx = np.where(signed <= offset_m)[0]
    upper_idx = np.setdiff1d(np.arange(len(pts)), lower_idx)
    return lower_idx, upper_idx

def ransac_main(
    input_path,
    output_dir="../output/ransac",
    max_planes=40,
    distance_threshold=0.02,
    ransac_n=3,
    num_iterations=1000,
    min_inliers_ratio=0.01,      # '현재 남은 점' 비율
    vertical_offset=0.2,         # 평면 법선 방향 거리로 해석
    up_axis='auto',              # 'y' | 'z' | 'auto'
    angle_tol_deg=30.0,
    visualize=True
):
    """
    RANSAC으로 평면을 찾아 바닥 후보(Up축과 이루는 각도 작음)를 고르고,
    평면 법선 기준 거리로 확장 컷(lower)/상부(upper)를 분리한다.
    """
    os.makedirs(output_dir, exist_ok=True)

    pcd = o3d.io.read_point_cloud(input_path)
    if not pcd.has_points():
        print(f"[ERROR] No points: {input_path}")
        return False

    total = len(pcd.points)
    original = pcd
    current_pcd = original
    current_idx = np.arange(total)

    # Up 벡터 후보 구성
    up_candidates = []
    ua = str(up_axis).lower()
    if ua == 'y':
        up_candidates = [np.array([0,1,0], float)]
    elif ua == 'z':
        up_candidates = [np.array([0,0,1], float)]
    else:
        up_candidates = [np.array([0,1,0], float), np.array([0,0,1], float)]

    planes = []  # list of dict(model, inliers(orig idx), normal, angle_to_up)

    planes_found = 0
    while planes_found < max_planes and len(current_pcd.points) >= ransac_n:
        plane, inliers = current_pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
        inliers = np.asarray(inliers, dtype=np.int64)

        # 현재 남아있는 점 기준으로 인라이어 하한
        cur_total = len(current_pcd.points)
        if inliers.size == 0 or inliers.size < max(1, int(cur_total * min_inliers_ratio)):
            break

        # 원본 인덱스
        orig_inliers = current_idx[inliers]
        n = np.asarray(plane[:3], float)
        if np.linalg.norm(n) == 0:
            # 비정상 모델이면 제거만 하고 진행
            remaining = np.setdiff1d(current_idx, orig_inliers)
            current_pcd = original.select_by_index(remaining)
            current_idx = remaining
            planes_found += 1
            continue

        n = _unit(n)
        # 후보 up들과의 각도 중 최소
        ang = min(_angle_deg_to_axis(n, up) for up in up_candidates)

        planes.append({
            "model": plane,
            "inliers": orig_inliers,
            "normal": n,
            "angle": ang
        })

        # 다음 반복을 위해 제거
        remaining = np.setdiff1d(current_idx, orig_inliers)
        current_pcd = original.select_by_index(remaining)
        current_idx = remaining
        planes_found += 1

    if not planes:
        print("[WARN] No planes detected.")
        return False

    # 바닥 후보: 각도 기준 통과하는 것들
    floor_sets = [p for p in planes if p["angle"] <= angle_tol_deg]
    if not floor_sets:
        # 없으면 가장 수평에 가까운 것 하나 채택
        floor_sets = [min(planes, key=lambda p: p["angle"])]

    # 인라이어 수가 가장 큰 바닥 후보를 대표로 사용
    floor_best = max(floor_sets, key=lambda p: len(p["inliers"]))
    floor_idx = np.unique(floor_best["inliers"])
    floor_cloud = original.select_by_index(floor_idx) if floor_idx.size > 0 else o3d.geometry.PointCloud()

    # 평면 법선 방향 오프셋 컷
    if floor_idx.size > 0 and vertical_offset > 0:
        lower_idx, upper_idx = _split_by_plane_offset(original, floor_best["model"], vertical_offset)
    else:
        lower_idx = floor_idx
        upper_idx = np.setdiff1d(np.arange(total), lower_idx)

    # 오버컷 방지: 상부가 0이면 컷 완화(상위 95퍼센타일)
    if upper_idx.size == 0 and vertical_offset > 0:
        # 평면 부호거리 95퍼센타일까지만 자름
        a,b,c,d = map(float, floor_best["model"][:4])
        n = _unit(np.array([a,b,c], float))
        signed = np.asarray(original.points) @ n + d
        cut95 = float(np.quantile(signed, 0.95))
        lower_idx = np.where(signed <= cut95)[0]
        upper_idx = np.setdiff1d(np.arange(total), lower_idx)
        print(f"[INFO] over-cut detected. relax cut to 95th percentile; kept={upper_idx.size}")

    lower_cloud = original.select_by_index(lower_idx)
    upper_cloud = original.select_by_index(upper_idx)

    # 저장
    out_floor = os.path.join(output_dir, "floor_plane.pcd")
    out_expanded = os.path.join(output_dir, "expanded_floor_plane.pcd")
    out_topview = os.path.join(output_dir, "topview.pcd")

    def _safe_write(path, cloud):
        if not cloud.has_points():
            print(f"[SKIP] empty -> {path}")
            return False
        return o3d.io.write_point_cloud(path, cloud)

    _safe_write(out_floor, floor_cloud)
    _safe_write(out_expanded, lower_cloud)
    _safe_write(out_topview, upper_cloud)

    print(f"[DONE] planes={len(planes)}, "
          f"floor={len(floor_cloud.points)}, cut(lower)={len(lower_cloud.points)}, kept(upper/topview)={len(upper_cloud.points)}")
    print(f"[OUT] floor={out_floor}\n[OUT] expanded={out_expanded}\n[OUT] topview={out_topview}")

    if visualize:
        if len(lower_cloud.points) > 0:
            lower_cloud.paint_uniform_color([0.0, 1.0, 0.0])   # 제거(초록)
        if len(upper_cloud.points) > 0:
            upper_cloud.paint_uniform_color([1.0, 0.0, 0.0])   # 유지(빨강)
        o3d.visualization.draw_geometries(
            [g for g in [lower_cloud, upper_cloud] if g.has_points()],
            window_name="Final Cut: Lower(removed, green) vs Kept(upper, red)"
        )
    return True

def main():
    # 실내 기본값. 필요하면 CLI 파서 붙여라.
    ransac_main(
        "../output/nonowall.pcd",
        distance_threshold=0.03,
        ransac_n=3,
        num_iterations=2000,
        min_inliers_ratio=0.002,
        vertical_offset=0.10,
        up_axis='auto',
        angle_tol_deg=25.0,
        visualize=True
    )

if __name__ == "__main__":
    main()
