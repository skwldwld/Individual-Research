import os
import cv2
import numpy as np
import open3d as o3d
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay
import pickle
import json
import math

# ------------------------------
# 유틸
# ------------------------------
def extract_corners_from_mask(mask, epsilon_ratio=0.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    all_points = []
    for contour in contours:
        if cv2.contourArea(contour) > 10:
            if epsilon_ratio > 0:
                epsilon = epsilon_ratio * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = approx[:, 0, :]
            else:
                points = contour[:, 0, :]
            all_points.append(points)
    return all_points

def load_binary_mask(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary

def extract_filled_pixels_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    draw = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) > 10:
            cv2.drawContours(draw, [c], -1, 255, thickness=cv2.FILLED)
    filled = np.argwhere(draw > 0)
    filled = np.flip(filled, axis=1)  # (x,y)
    return filled

def resample_contour(contour, num_points):
    if len(contour) < 2:
        return contour
    dists = np.sqrt(np.sum(np.diff(contour, axis=0, append=contour[:1])**2, axis=1))
    cumulative = np.cumsum(dists)
    cumulative = np.insert(cumulative, 0, 0)[:-1]
    total_length = cumulative[-1] + dists[-1]
    ts = np.linspace(0, total_length, num_points, endpoint=False)
    fx = interp1d(cumulative, contour[:, 0], kind='linear', fill_value="extrapolate")
    fy = interp1d(cumulative, contour[:, 1], kind='linear', fill_value="extrapolate")
    resampled_contour = np.stack([fx(ts), fy(ts)], axis=1)
    return resampled_contour

def load_pixel_to_points_mapping(pkl_path):
    try:
        with open(pkl_path, 'rb') as f:
            pixel_to_points = pickle.load(f)
        print(f"[SUCCESS] Loaded pixel-to-points mapping from {pkl_path}")
        print(f"   Number of pixel mappings: {len(pixel_to_points)}")
        return pixel_to_points
    except Exception as e:
        print(f"[ERROR] Error loading pixel-to-points mapping: {e}")
        return {}

# ------------------------------
# 좌표계 정합(핵심): topview 픽셀 (px,py) -> 월드 (x,z) 2D 유사변환 추정
# ------------------------------
def _parse_pixel_key(k):
    # 키가 (x,y) tuple/list 이거나 'x,y' 문자열일 수 있음
    if isinstance(k, (tuple, list)) and len(k) >= 2:
        return int(k[0]), int(k[1])
    if isinstance(k, str):
        s = k.strip().replace('(', '').replace(')', '').replace('[', '').replace(']', '')
        parts = s.split(',')
        if len(parts) >= 2:
            try:
                return int(float(parts[0])), int(float(parts[1]))
            except:
                return None
    return None

def build_correspondences_from_pkl(pkl_path, img_height, sample_every=20, max_points=20000):
    d = load_pixel_to_points_mapping(pkl_path)
    src = []  # pixel plane (px, img_h-1-py)
    dst = []  # world xz
    c = 0
    for i, (k, pts) in enumerate(d.items()):
        if sample_every > 1 and (i % sample_every != 0):
            continue
        parsed = _parse_pixel_key(k)
        if parsed is None:
            continue
        px, py = parsed
        if not isinstance(pts, (list, tuple)) or len(pts) == 0:
            continue
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        world_xz = np.mean(arr[:, [0, 2]], axis=0)
        src.append([float(px), float(img_height - 1 - py)])
        dst.append([float(world_xz[0]), float(world_xz[1])])
        c += 1
        if c >= max_points:
            break
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    print(f"[INFO] Built {len(src)} correspondence pairs from {os.path.basename(pkl_path)} (after sampling)")
    return src, dst

def estimate_affine_partial_2d(src, dst, ransac_thresh=0.5):
    # 유사변환(회전+등방성 스케일+이동) 추정. 반사 포함 가능
    if len(src) < 3 or len(dst) < 3:
        return None, None
    M, inliers = cv2.estimateAffinePartial2D(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=20000,
        confidence=0.999,
        refineIters=50
    )
    if M is None:
        print("[ERROR] Failed to estimate affine")
        return None, None
    A = M[:, :2]
    sx = np.linalg.norm(A[:, 0])
    sy = np.linalg.norm(A[:, 1])
    s = (sx + sy) / 2.0
    # 회전 각도(라디안)
    if s > 1e-8:
        R = A / s
        theta = math.atan2(R[1, 0], R[0, 0])
        deg = math.degrees(theta)
    else:
        deg = 0.0
    print(f"[INFO] Affine estimated. scale≈{s:.6f}, rot≈{deg:.3f} deg, trans=({M[0,2]:.4f},{M[1,2]:.4f}), inliers={int(inliers.sum()) if inliers is not None else 'NA'}")
    return M, inliers

def apply_affine_to_pixels(M, pts_px, img_height):
    # pts_px: (N,2) in pixel coords (px,py). 내부에서 y-flip 적용
    if pts_px.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    px = pts_px[:, 0:1].astype(np.float32)
    py = pts_px[:, 1:2].astype(np.float32)
    yflip = (img_height - 1) - py
    ones = np.ones_like(px)
    P = np.concatenate([px, yflip, ones], axis=1).T  # (3,N)
    W = (M @ P).T  # (N,2)
    return W.astype(np.float32)

# ------------------------------
# 메쉬 유틸
# ------------------------------
def create_constrained_flat_mesh(points_2d, y_level, contour_2d):
    if len(points_2d) < 3:
        return o3d.geometry.TriangleMesh()
    tri = Delaunay(points_2d)
    triangles = []
    for simplex in tri.simplices:
        tri_pts = points_2d[simplex]
        centroid = np.mean(tri_pts, axis=0)
        if cv2.pointPolygonTest(contour_2d.astype(np.float32), tuple(centroid), False) >= 0:
            triangles.append(simplex)
    if len(triangles) == 0:
        return o3d.geometry.TriangleMesh()
    triangles = np.array(triangles)
    vertices = np.column_stack([points_2d[:, 0], np.full(len(points_2d), y_level), points_2d[:, 1]])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh

# ------------------------------
# 높이 추정 유틸 (PKL에서 y 사용)
# ------------------------------
def _collect_y_from_pkl_in_contour(pkl_path, contour_px, img_height, min_pts=50):
    d = load_pixel_to_points_mapping(pkl_path)
    ys = []
    contour_np = np.asarray(contour_px, dtype=np.float32)
    for k, pts in d.items():
        parsed = _parse_pixel_key(k)
        if parsed is None:
            continue
        px, py = parsed
        # 컨투어는 픽셀좌표계 기준. y-flip 없이 그대로.
        inside = cv2.pointPolygonTest(contour_np, (float(px), float(py)), False)
        if inside >= 0:
            arr = np.asarray(pts, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                ys.extend(arr[:, 1].tolist())
    ys = np.asarray(ys, dtype=np.float32)
    if ys.size < min_pts:
        return None  # 충분치 않음
    return ys

def estimate_height_from_pkl(pkl_path, floor_y_level, contour_px=None, img_height=None,
                             percentile=98.0, min_height=1.0, max_height=50.0):
    """컨투어 내부 픽셀에 해당하는 PKL y들로 높이를 추정. 부족하면 전체 PKL 기반으로 fallback.
    height = clamp(percentile(y) - floor_y_level).
    """
    ys = None
    if contour_px is not None:
        ys = _collect_y_from_pkl_in_contour(pkl_path, contour_px, img_height)
    if ys is None or ys.size == 0:
        # fallback: 전체 PKL
        d = load_pixel_to_points_mapping(pkl_path)
        y_all = []
        for pts in d.values():
            arr = np.asarray(pts, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                y_all.extend(arr[:, 1].tolist())
        ys = np.asarray(y_all, dtype=np.float32)
    if ys.size == 0:
        print("[WARNING] No Y samples found for height estimation. Use min_height.")
        return float(min_height)
    top = float(np.percentile(ys, percentile))
    h = float(max(min(top - float(floor_y_level), max_height), min_height))
    return h

# ------------------------------
# Poisson from PCD (floor)
# ------------------------------
def poisson_mesh_from_pcd(
    input_pcd_path,
    output_mesh_path,
    normal_mode="z",
    depth=9,
    density_quantile=0.01,
    normal_radius=0.02,
    normal_max_nn=10,
    tangent_k=10
):
    print(f"[INFO] [1/5] Loading: {input_pcd_path}")
    pcd = o3d.io.read_point_cloud(input_pcd_path)
    if not pcd.has_points():
        print("[ERROR] No points found in the input file.")
        return None
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    print(f"Number of points after downsampling: {len(pcd.points)}")
    print(f"[INFO] [2/5] Estimating normals... (radius={normal_radius}, max_nn={normal_max_nn})")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=normal_radius, max_nn=normal_max_nn))
    if normal_mode == "y":
        normals = np.tile(np.array([0., 1., 0.]), (np.asarray(pcd.points).shape[0], 1))
        pcd.normals = o3d.utility.Vector3dVector(normals)
    elif normal_mode == "z":
        normals = np.tile(np.array([0., 0., 1.]), (np.asarray(pcd.points).shape[0], 1))
        pcd.normals = o3d.utility.Vector3dVector(normals)
    else:
        pcd.orient_normals_consistent_tangent_plane(k=tangent_k)
    print(f"[INFO] [3/5] Running Poisson reconstruction... (depth={depth})")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    print(f"[INFO] [4/5] Removing low-density vertices...")
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, density_quantile)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"[INFO] [5/5] Saving mesh: {output_mesh_path}")
    os.makedirs(os.path.dirname(output_mesh_path), exist_ok=True)
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    print(f"[SUCCESS] Mesh saved: {output_mesh_path} ({len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles)")
    return mesh

# ------------------------------
# 메인
# ------------------------------
def main():
    z_fighting_offset = 0.001

    # 카테고리 설정: height_strategy 를 지정 가능
    # - 'fixed': 고정 높이 사용 (height)
    # - 'pkl_percentile': PKL 기반 퍼센타일로 컨투어별 높이 추정 (percentile/min_height/max_height)
    categories = [
        {
            "name": "wall",
            "mask_path": "../output/morph/topview/wall/morph_smoothed.png",
            "pkl_path": "../output/outline/topview/wall/pixel_to_points.pkl",
            "color": [0.9, 0.2, 0.2],
            "height_strategy": "pkl_percentile",
            "percentile": 98.0,
            "min_height": 2.0,
            "max_height": 50.0,
        },
        {
            "name": "material",
            "mask_path": "../output/morph/topview/material/morph_smoothed.png",
            "pkl_path": "../output/outline/topview/material/pixel_to_points.pkl",
            "color": [0.2, 0.8, 0.2],
            "height_strategy": "pkl_percentile",
            "percentile": 95.0,
            "min_height": 0.5,
            "max_height": 30.0,
        },
    ]

    # floor
    input_pcd_floor = "../output/pcd/floor/floor_plane.pcd"
    output_mesh_floor = "../output/mesh/final_floor.ply"

    # 바닥 Poisson 메쉬
    mesh_floor = poisson_mesh_from_pcd(
        input_pcd_floor,
        output_mesh_floor,
        normal_mode="y",
        depth=11,
        density_quantile=0.01,
        normal_radius=0.03,
        normal_max_nn=15,
        tangent_k=30
    )
    floor_y_level = 0.0
    if mesh_floor is not None:
        floor_y_level = mesh_floor.get_min_bound()[1]
        floor_bounds = mesh_floor.get_axis_aligned_bounding_box()
        print(f"Detected floor Y-level: {floor_y_level:.4f}")
        print(f"Floor Mesh X range: [{floor_bounds.min_bound[0]:.4f}, {floor_bounds.max_bound[0]:.4f}]")
        print(f"Floor Mesh Z range: [{floor_bounds.min_bound[2]:.4f}, {floor_bounds.max_bound[2]:.4f}]")
    else:
        print("[WARNING] Floor mesh failed; base at Y=0.")

    # 각 카테고리별로 독립 처리 + 자체 유사변환 추정 (픽셀->월드 xz)
    category_meshes = []

    for cat in categories:
        name = cat["name"]
        print(f"\n[====] Processing category: {name} [====]")

        mask = load_binary_mask(cat["mask_path"])
        if mask is None:
            print(f"[WARNING] Mask not found for {name}: {cat['mask_path']}. Skipped.")
            continue
        img_height_top = mask.shape[0]

        # --- 1) 이 카테고리의 PKL만으로 픽셀->월드 2D affine 추정 ---
        src_px, dst_w = build_correspondences_from_pkl(cat["pkl_path"], img_height_top, sample_every=10, max_points=30000)
        M, inliers = estimate_affine_partial_2d(src_px, dst_w, ransac_thresh=0.5)
        if M is None:
            print(f"[ERROR] Affine estimation failed for {name}. Skipped.")
            continue

        # --- 2) 카테고리별 윤곽/채움 픽셀 ---
        contour_px_list = extract_corners_from_mask(mask, epsilon_ratio=0.0)
        if not contour_px_list:
            print(f"[WARNING] No contours for {name}. Skipped.")
            continue
        filled_pixels = extract_filled_pixels_from_mask(mask)

        # --- 3) world 좌표 변환 (affine 적용) ---
        all_contours_world = []
        for i, contour_px in enumerate(contour_px_list):
            if len(contour_px) == 0:
                continue
            contour_px_arr = np.asarray(contour_px, dtype=np.float32)
            contour_world = apply_affine_to_pixels(M, contour_px_arr, img_height_top)
            all_contours_world.append(contour_world)
            print(f"  {name} contour {i+1}: {len(contour_px)} points")

        if len(filled_pixels) == 0:
            print(f"[WARNING] No filled pixels for {name}. Skipped.")
            continue
        filled_px_arr = np.asarray(filled_pixels, dtype=np.float32)
        filled_points_world = apply_affine_to_pixels(M, filled_px_arr, img_height_top)
        points_2d = filled_points_world  # (x,z)

        print(f"[INFO] Creating {name} meshes using Delaunay + contour mask ...")
        cat_mesh_accum = o3d.geometry.TriangleMesh()

        # 컨투어별 가변 높이로 생성
        for i, (contour_px, contour_world) in enumerate(zip(contour_px_list, all_contours_world), start=1):
            # --- 높이 결정 ---
            if cat.get("height_strategy", "fixed") == "fixed":
                fixed_h = float(cat.get("height", 10.0))
                height = fixed_h
            else:  # 'pkl_percentile'
                height = estimate_height_from_pkl(
                    cat["pkl_path"],
                    floor_y_level,
                    contour_px=contour_px,
                    img_height=img_height_top,
                    percentile=float(cat.get("percentile", 98.0)),
                    min_height=float(cat.get("min_height", 1.0)),
                    max_height=float(cat.get("max_height", 50.0)),
                )
            print(f"    -> contour {i} height = {height:.3f}")

            # --- base/top mesh ---
            try:
                base_mesh = create_constrained_flat_mesh(points_2d, floor_y_level + z_fighting_offset, contour_world)
                top_mesh  = create_constrained_flat_mesh(points_2d, floor_y_level + height + z_fighting_offset, contour_world)
            except Exception as e:
                print(f"  [ERROR] {name} contour {i}: {e}")
                continue

            # --- side extrude (개별 높이) ---
            n = len(contour_world)
            if n >= 3:
                extruded_top_outline_3d = np.column_stack([
                    contour_world[:, 0],
                    np.full(n, floor_y_level + height + z_fighting_offset),
                    contour_world[:, 1]
                ])
                extruded_base_outline_3d = np.column_stack([
                    contour_world[:, 0],
                    np.full(n, floor_y_level + z_fighting_offset),
                    contour_world[:, 1]
                ])
                side_vertices = np.vstack([extruded_base_outline_3d, extruded_top_outline_3d])
                side_triangles = []
                for j in range(n):
                    side_triangles.append([j, (j + 1) % n, (j + 1) % n + n])
                    side_triangles.append([j, (j + 1) % n + n, j + n])
                side_mesh = o3d.geometry.TriangleMesh()
                side_mesh.vertices = o3d.utility.Vector3dVector(side_vertices)
                side_mesh.triangles = o3d.utility.Vector3iVector(side_triangles)
                side_mesh.compute_vertex_normals()
            else:
                side_mesh = o3d.geometry.TriangleMesh()
                print(f"  [WARNING] {name} contour {i}: Too few points ({n}) for sides")

            # --- 합치기 ---
            contour_mesh = base_mesh + top_mesh + side_mesh
            cat_mesh_accum += contour_mesh

        # 카테고리 색칠 및 저장
        if len(cat_mesh_accum.vertices) == 0:
            print(f"[WARNING] Empty mesh for {name}")
            continue
        cat_mesh_accum.paint_uniform_color(cat["color"])
        os.makedirs("../output/mesh", exist_ok=True)
        cat_out = f"../output/mesh/{name}_mesh.ply"
        o3d.io.write_triangle_mesh(cat_out, cat_mesh_accum)
        print(f"[SUCCESS] {name} mesh saved: {cat_out}")
        category_meshes.append(cat_mesh_accum)

    # 최종 병합: 카테고리 + 바닥
    merged_mesh = None
    if category_meshes:
        merged_mesh = category_meshes[0]
        for m in category_meshes[1:]:
            merged_mesh += m

    if merged_mesh is not None and mesh_floor is not None:
        mesh_floor.paint_uniform_color([0.1, 0.1, 0.7])
        print("[INFO] [Merging] Merging category meshes with floor...")
        final_mesh = merged_mesh + mesh_floor
        merged_output = "../output/mesh/merged_wall_material_floor.ply"
        o3d.io.write_triangle_mesh(merged_output, final_mesh)
        print(f"[SUCCESS] Merged mesh saved: {merged_output} ({len(final_mesh.vertices)} vertices, {len(final_mesh.triangles)} triangles)")
        o3d.visualization.draw_geometries([final_mesh], mesh_show_back_face=True)
    else:
        # 시각화 폴백
        if merged_mesh is not None:
            o3d.visualization.draw_geometries([merged_mesh], mesh_show_back_face=True)
        elif mesh_floor is not None:
            o3d.visualization.draw_geometries([mesh_floor], mesh_show_back_face=True)
        else:
            print("[ERROR] Nothing to visualize.")

if __name__ == "__main__":
    main()
