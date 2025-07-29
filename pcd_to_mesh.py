import cv2
import numpy as np
import open3d as o3d
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay
import pickle
import json

def extract_corners(img_path, epsilon_ratio=0.0):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError(f"No contour found in {img_path}")
    contour = max(contours, key=cv2.contourArea)
    if epsilon_ratio > 0:
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx[:, 0, :]
    else:
        points = contour[:, 0, :]
    return points

def extract_filled_pixels(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError(f"No contour found in {img_path}")
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    filled_pixels = np.argwhere(mask > 0)
    filled_pixels = np.flip(filled_pixels, axis=1) # (x, y)
    return filled_pixels

def pixel_to_world_top(px, py, min_x, min_z, scale, img_height):
    wx = px / scale + min_x
    wz = (img_height - 1 - py) / scale + min_z
    return wx, wz

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
        print(f"✅ Loaded pixel-to-points mapping from {pkl_path}")
        print(f"   Number of pixel mappings: {len(pixel_to_points)}")
        return pixel_to_points
    except Exception as e:
        print(f"❌ Error loading pixel-to-points mapping: {e}")
        return {}

def calculate_coordinate_mapping_from_pkl(floor_pkl_path, top_pkl_path):
    floor_pixel_to_points = load_pixel_to_points_mapping(floor_pkl_path)
    top_pixel_to_points = load_pixel_to_points_mapping(top_pkl_path)
    if not floor_pixel_to_points or not top_pixel_to_points:
        print("⚠️ Could not load pixel-to-points mappings")
        return 0.0, 0.0, {}
    floor_points_3d = []
    for pixel_coord, points_list in floor_pixel_to_points.items():
        for point in points_list:
            floor_points_3d.append(point)
    if not floor_points_3d:
        print("⚠️ No floor points found in pkl file")
        return 0.0, 0.0, {}
    floor_points_3d = np.array(floor_points_3d)
    floor_x_min, floor_x_max = np.min(floor_points_3d[:, 0]), np.max(floor_points_3d[:, 0])
    floor_z_min, floor_z_max = np.min(floor_points_3d[:, 2]), np.max(floor_points_3d[:, 2])
    floor_y_level = np.min(floor_points_3d[:, 1])
    top_points_3d = []
    for pixel_coord, points_list in top_pixel_to_points.items():
        for point in points_list:
            top_points_3d.append(point)
    if not top_points_3d:
        print("⚠️ No top points found in pkl file")
        return 0.0, 0.0, {}
    top_points_3d = np.array(top_points_3d)
    top_x_min, top_x_max = np.min(top_points_3d[:, 0]), np.max(top_points_3d[:, 0])
    top_z_min, top_z_max = np.min(top_points_3d[:, 2]), np.max(top_points_3d[:, 2])
    floor_center_x = (floor_x_min + floor_x_max) / 2
    floor_center_z = (floor_z_min + floor_z_max) / 2
    top_center_x = (top_x_min + top_x_max) / 2
    top_center_z = (top_z_min + top_z_max) / 2
    translation_x = 0.0
    translation_z = 0.0
    mapping_info = {
        'floor_bounds': {
            'x_min': floor_x_min, 'x_max': floor_x_max,
            'z_min': floor_z_min, 'z_max': floor_z_max,
            'y_level': floor_y_level
        },
        'top_bounds': {
            'x_min': top_x_min, 'x_max': top_x_max,
            'z_min': top_z_min, 'z_max': top_z_max
        },
        'floor_center': (floor_center_x, floor_center_z),
        'top_center': (top_center_x, top_center_z),
        'translation': (translation_x, translation_z),
        'num_floor_points': len(floor_points_3d),
        'num_top_points': len(top_points_3d)
    }
    print(f"📐 PKL-based Coordinate Mapping Info (Original Position Preserved):")
    print(f"   Floor bounds: X[{floor_x_min:.4f}, {floor_x_max:.4f}], Z[{floor_z_min:.4f}, {floor_z_max:.4f}]")
    print(f"   Top bounds: X[{top_x_min:.4f}, {top_x_max:.4f}], Z[{top_z_min:.4f}, {top_z_max:.4f}]")
    print(f"   Floor center: ({floor_center_x:.4f}, {floor_center_z:.4f})")
    print(f"   Top center: ({top_center_x:.4f}, {top_center_z:.4f})")
    print(f"   Translation: ({translation_x:.4f}, {translation_z:.4f}) - 원본 위치 유지")
    print(f"   Floor points: {len(floor_points_3d)}, Top points: {len(top_points_3d)}")
    return translation_x, translation_z, mapping_info

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
    print(f"📦 [1/5] Loading: {input_pcd_path}")
    pcd = o3d.io.read_point_cloud(input_pcd_path)
    if not pcd.has_points():
        print("❌ No points found in the input file.")
        return None
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    print(f"Number of points after downsampling: {len(pcd.points)}")
    print(f"🔄 [2/5] Estimating normals... (radius={normal_radius}, max_nn={normal_max_nn})")
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
    print(f"🧠 [3/5] Running Poisson reconstruction... (depth={depth})")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    print(f"🧹 [4/5] Removing low-density vertices...")
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, density_quantile)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"💾 [5/5] Saving mesh: {output_mesh_path}")
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    print(f"✅ Mesh saved: {output_mesh_path} ({len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles)")
    return mesh

# 핵심! Delaunay + 윤곽선 내부만 남기는 flat mesh 함수
def create_constrained_flat_mesh(points_2d, y_level, contour_2d):
    tri = Delaunay(points_2d)
    triangles = []
    for simplex in tri.simplices:
        tri_pts = points_2d[simplex]
        centroid = np.mean(tri_pts, axis=0)
        # contour_2d는 폐곡선, float32, shape (N,2)
        if cv2.pointPolygonTest(contour_2d.astype(np.float32), tuple(centroid), False) >= 0:
            triangles.append(simplex)
    triangles = np.array(triangles)
    vertices = np.column_stack([points_2d[:,0], np.full(len(points_2d), y_level), points_2d[:,1]])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh

def main():
    min_proj_x_top = -51.25127410888672
    min_proj_z_top = -33.814571380615234
    scale_factor_top = 5
    img_height_top = 163
    fixed_height = 10.0
    z_fighting_offset = 0.001
    top_img = "output/morph/top_view/morph_smoothed.png"

    try:
        top_corners_px_raw = extract_corners(top_img, epsilon_ratio=0.0)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        print("Please ensure 'output/morph/top_view/morph_smoothed.png' exists.")
        return

    try:
        filled_pixels = extract_filled_pixels(top_img)
    except Exception as e:
        print(f"Error extracting filled pixels: {e}")
        return

    input_pcd_floor = "output/pcd/final_result_floor_plane.pcd"
    output_mesh_floor = "output/mesh/final_result_floor_plane_poisson.ply"
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
    floor_x_min, floor_x_max, floor_z_min, floor_z_max = np.nan, np.nan, np.nan, np.nan
    if mesh_floor is not None:
        floor_y_level = mesh_floor.get_min_bound()[1]
        print(f"Detected floor Y-level from PCD: {floor_y_level:.4f}")
        floor_bounds = mesh_floor.get_axis_aligned_bounding_box()
        floor_x_min, floor_x_max = floor_bounds.min_bound[0], floor_bounds.max_bound[0]
        floor_z_min, floor_z_max = floor_bounds.min_bound[2], floor_bounds.max_bound[2]
        print(f"Floor Mesh X range: [{floor_x_min:.4f}, {floor_x_max:.4f}]")
        print(f"Floor Mesh Z range: [{floor_z_min:.4f}, {floor_z_max:.4f}]")
    else:
        print("⚠️ Warning: Floor mesh could not be generated. Extruded mesh base will be at Y=0.")

    # 좌표 mapping(pkl) -> translation
    floor_pkl_path = "output/outline/floor_plane/pixel_to_points.pkl"
    top_pkl_path = "output/outline/top_view/pixel_to_points.pkl"
    translation_x, translation_z, mapping_info = calculate_coordinate_mapping_from_pkl(
        floor_pkl_path, top_pkl_path
    )

    # ----- 윤곽선 world 좌표 변환 -----
    contour_px = extract_corners(top_img, epsilon_ratio=0.0)
    contour_world = np.array([
        pixel_to_world_top(px, py, min_proj_x_top, min_proj_z_top, scale_factor_top, img_height_top)
        for (px, py) in contour_px
    ])
    contour_world[:, 0] += translation_x
    contour_world[:, 1] += translation_z

    # 내부 채움 포인트들 world 좌표 변환 (x, z만 추출)
    filled_points_world = np.array([
        pixel_to_world_top(px, py, min_proj_x_top, min_proj_z_top, scale_factor_top, img_height_top)
        for (px, py) in filled_pixels
    ])
    filled_points_world[:, 0] += translation_x
    filled_points_world[:, 1] += translation_z
    points_2d = filled_points_world  # shape (N,2) : (x, z)

    print("🔧 Creating base and top meshes using Delaunay + contour mask ...")
    base_mesh = create_constrained_flat_mesh(points_2d, floor_y_level + z_fighting_offset, contour_world)
    top_mesh  = create_constrained_flat_mesh(points_2d, floor_y_level + fixed_height + z_fighting_offset, contour_world)

    # 벽(사이드) 메쉬 - 윤곽선 world 좌표로 extrude
    side_contour = contour_world  # 이미 world 좌표 변환 + translation 적용
    n = len(side_contour)
    extruded_top_outline_3d = np.column_stack([
        side_contour[:, 0],
        np.full(n, floor_y_level + fixed_height + z_fighting_offset),
        side_contour[:, 1]
    ])
    extruded_base_outline_3d = np.column_stack([
        side_contour[:, 0],
        np.full(n, floor_y_level + z_fighting_offset),
        side_contour[:, 1]
    ])
    side_vertices = np.vstack([extruded_base_outline_3d, extruded_top_outline_3d])
    side_triangles = []
    for i in range(n):
        side_triangles.append([i, (i + 1) % n, (i + 1) % n + n])
        side_triangles.append([i, (i + 1) % n + n, i + n])
    extruded_mesh_sides = o3d.geometry.TriangleMesh()
    extruded_mesh_sides.vertices = o3d.utility.Vector3dVector(side_vertices)
    extruded_mesh_sides.triangles = o3d.utility.Vector3iVector(side_triangles)
    extruded_mesh_sides.compute_vertex_normals()

    # 최종 합치기
    solid_extruded_mesh = base_mesh + top_mesh + extruded_mesh_sides
    solid_extruded_mesh.paint_uniform_color([0.1, 0.7, 0.1])
    output_extruded_mesh_filename = "output/pcd/solid_extruded_top_view_mesh.ply"
    o3d.io.write_triangle_mesh(output_extruded_mesh_filename, solid_extruded_mesh)
    print(f"✅ Solid extruded mesh saved: {output_extruded_mesh_filename}")

    # 디버그용 구 표시(윤곽선 꼭짓점)
    # sphere_meshes = []
    # sphere_radius = 1.0
    # sphere_color = [1, 0, 0]
    # for (x, z) in side_contour:
    #     v = np.array([x, floor_y_level + fixed_height + z_fighting_offset, z])
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    #     sphere.translate(v)
    #     sphere.paint_uniform_color(sphere_color)
    #     sphere_meshes.append(sphere)

    # 바닥 + extrude 합치기, 시각화
    if solid_extruded_mesh is not None and mesh_floor is not None:
        mesh_floor.paint_uniform_color([0.1, 0.1, 0.7])
        print("🔗 [Merging] Merging meshes...")
        merged_mesh = solid_extruded_mesh + mesh_floor
        merged_output = "output/mesh/merged_result_solid.ply"
        o3d.io.write_triangle_mesh(merged_output, merged_mesh)
        print(f"🎉 Merged mesh saved: {merged_output} ({len(merged_mesh.vertices)} vertices, {len(merged_mesh.triangles)} triangles)")
        print("👁️ Visualizing merged mesh + spheres...")
        o3d.visualization.draw_geometries([merged_mesh], mesh_show_back_face=True)
        # o3d.visualization.draw_geometries([merged_mesh] + sphere_meshes, mesh_show_back_face=True)
    else:
        print("⚠️ Warning: Could not generate extruded mesh or floor mesh. Skipping merge.")
        if solid_extruded_mesh is not None:
            print("👁️ Visualizing only extruded mesh + spheres...")
            o3d.visualization.draw_geometries([solid_extruded_mesh], mesh_show_back_face=True)
            # o3d.visualization.draw_geometries([solid_extruded_mesh] + sphere_meshes, mesh_show_back_face=True)
        elif mesh_floor is not None:
            print("👁️ Visualizing only floor mesh...")
            o3d.visualization.draw_geometries([mesh_floor], mesh_show_back_face=True)

if __name__ == "__main__":
    main()
