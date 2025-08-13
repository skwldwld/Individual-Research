# ransac.py
import open3d as o3d
import numpy as np
import os
 
def ransac_main(
    input_path: str,
    output_dir: str = "../output/ransac",
    max_planes: int = 40,
    distance_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    min_inliers_ratio: float = 0.01,
    vertical_offset: float = 0.2,   # 바닥 윗부분 컷 높이(미터)
    visualize: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    pcd = o3d.io.read_point_cloud(input_path)
    if not pcd.has_points():
        print(f"[ERROR] No points: {input_path}")
        return False

    total = len(pcd.points)
    original = pcd
    current_pcd = original
    current_idx = np.arange(total)

    floor_indices = []
    planes_found = 0

    # 다중 평면 탐색
    while planes_found < max_planes and len(current_pcd.points) > 0:
        plane, inliers = current_pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
        if len(inliers) == 0 or len(inliers) < total * min_inliers_ratio:
            break

        orig_inliers = current_idx[inliers]
        n = np.asarray(plane[:3], dtype=float)
        nrm = np.linalg.norm(n)
        if nrm == 0:
            # 그냥 제거하고 진행
            remaining = np.setdiff1d(current_idx, orig_inliers)
            current_pcd = original.select_by_index(remaining)
            current_idx = remaining
            planes_found += 1
            continue

        n /= nrm
        angle_deg = np.degrees(np.arccos(np.abs(np.dot(n, [0.0, 1.0, 0.0]))))
        if angle_deg <= 30.0:
            floor_indices.extend(orig_inliers)

        remaining = np.setdiff1d(current_idx, orig_inliers)
        current_pcd = original.select_by_index(remaining)
        current_idx = remaining
        planes_found += 1

    # 바닥·상부 분리
    floor_idx = np.unique(np.asarray(floor_indices, dtype=np.int64))
    floor_cloud = original.select_by_index(floor_idx) if floor_idx.size > 0 else o3d.geometry.PointCloud()

    # 바닥 아랫부분 컷: y <= (floor_y_max + offset)
    if len(floor_cloud.points) > 0 and vertical_offset > 0:
        all_pts = np.asarray(original.points)
        y_max = np.max(np.asarray(floor_cloud.points)[:, 1])
        y_cut = y_max + float(vertical_offset)
        lower_mask = all_pts[:, 1] <= y_cut
        lower_idx = np.where(lower_mask)[0]
    else:
        lower_idx = floor_idx

    lower_cloud = original.select_by_index(lower_idx)  # 제거(확장바닥) 세트
    upper_idx = np.setdiff1d(np.arange(total), lower_idx)
    upper_cloud = original.select_by_index(upper_idx)  # 유지(상부)

    # 저장 (outline이 그대로 쓴다)
    out_floor = os.path.join(output_dir, "floor_plane.pcd")
    out_expanded = os.path.join(output_dir, "expanded_floor_plane.pcd")
    out_topview = os.path.join(output_dir, "topview.pcd")

    o3d.io.write_point_cloud(out_floor, floor_cloud)
    o3d.io.write_point_cloud(out_expanded, lower_cloud)
    o3d.io.write_point_cloud(out_topview, upper_cloud)

    print(f"[DONE] planes={planes_found}, floor={len(floor_cloud.points)}, "
          f"cut(lower)={len(lower_cloud.points)}, kept(upper/topview)={len(upper_cloud.points)}")
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

if __name__ == "__main__":
    # remove.py가 만든 ../output/nonowall.pcd를 입력으로 사용
    ransac_main("../output/nonowall.pcd")
