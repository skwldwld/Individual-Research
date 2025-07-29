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

def create_mesh_from_contour(contour_2d, y_level):
    """
    윤곽선을 기반으로 정확한 메쉬를 생성합니다. (현재 코드에서 직접 사용되지 않음, 참고용)
    """
    # 윤곽선을 3D로 변환
    contour_3d = np.column_stack([
        contour_2d[:, 0],
        np.full(len(contour_2d), y_level),
        contour_2d[:, 1]
    ])
    
    # 윤곽선의 중심점 계산
    center_2d = np.mean(contour_2d, axis=0)
    center_3d = np.array([center_2d[0], y_level, center_2d[1]])
    
    # 윤곽선 내부를 완전히 채우기 위한 삼각형 생성
    vertices = [center_3d]  # 중심점이 첫 번째 점
    triangles = []
    
    n = len(contour_3d)
    
    # 윤곽선의 각 점을 중심점과 연결하여 삼각형 생성
    for i in range(n):
        vertices.append(contour_3d[i])
        # 삼각형: 중심점, 현재 점, 다음 점
        triangles.append([0, i + 1, ((i + 1) % n) + 1])
    
    # 메쉬 생성
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    
    return mesh

def create_complete_filled_mesh(img_path, y_level, min_proj_x, min_proj_z, scale, img_height, translation_x=0.0, translation_z=0.0):
    """
    이미지에서 모든 윤곽선(외부 + 내부 구멍)을 추출하여 완전히 채워진 메쉬를 생성합니다.
    Args:
        translation_x, translation_z: 이 함수 내에서 추가로 적용할 변환값 (일반적으로 0으로 설정하여 외부에서 한 번만 변환하도록 함)
    """
    # 이미지 로드 및 이진화
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
    
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 모든 윤곽선 찾기 (외부 + 내부 구멍)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        raise ValueError(f"No contour found in {img_path}")
    
    # 가장 큰 외부 윤곽선 찾기
    external_contour = max(contours, key=cv2.contourArea)
    
    all_vertices = []
    all_triangles = []
    vertex_offset = 0
    
    # 외부 윤곽선 처리
    external_points_px = external_contour[:, 0, :]
    external_points_world = np.array([
        pixel_to_world_top(px, py, min_proj_x, min_proj_z, scale, img_height)
        for (px, py) in external_points_px
    ])
    
    # 외부에서 이미 translation이 적용되었으므로, 여기서는 추가 translation을 0으로 받거나, 필요하면 전달받은 값 적용
    external_points_world[:, 0] += translation_x
    external_points_world[:, 1] += translation_z
    
    # 외부 윤곽선을 3D로 변환
    external_3d = np.column_stack([
        external_points_world[:, 0],
        np.full(len(external_points_world), y_level),
        external_points_world[:, 1]
    ])
    
    # 외부 윤곽선의 중심점
    center_2d = np.mean(external_points_world, axis=0)
    center_3d = np.array([center_2d[0], y_level, center_2d[1]])
    
    all_vertices.append(center_3d)
    
    # 외부 윤곽선 점들 추가
    for point in external_3d:
        all_vertices.append(point)
    
    # 외부 윤곽선 삼각형 생성
    n_external = len(external_3d)
    for i in range(n_external):
        all_triangles.append([0, i + 1, ((i + 1) % n_external) + 1])
    
    vertex_offset = len(all_vertices)
    
    # 내부 구멍들 처리 (hierarchy 정보를 사용하여 외부 윤곽선 내부에 있는 것만 처리)
    for i, contour in enumerate(contours):
        # hierarchy[0][i][3] == 0 은 부모가 없다는 의미 (최외곽 윤곽선)
        # hierarchy[0][i][3] != -1 은 부모가 있다는 의미 (내부 윤곽선)
        # 즉, 내부 구멍은 외부 윤곽선의 자식으로 나타남
        if hierarchy[0][i][3] != -1 and cv2.contourArea(contour) > 10: # 부모가 있고, 너무 작지 않은 구멍
            hole_points_px = contour[:, 0, :]
            hole_points_world = np.array([
                pixel_to_world_top(px, py, min_proj_x, min_proj_z, scale, img_height)
                for (px, py) in hole_points_px
            ])
            
            hole_points_world[:, 0] += translation_x
            hole_points_world[:, 1] += translation_z
            
            # 구멍을 3D로 변환
            hole_3d = np.column_stack([
                hole_points_world[:, 0],
                np.full(len(hole_points_world), y_level),
                hole_points_world[:, 1]
            ])
            
            # 구멍의 중심점
            hole_center_2d = np.mean(hole_points_world, axis=0)
            hole_center_3d = np.array([hole_center_2d[0], y_level, hole_center_2d[1]])
            
            all_vertices.append(hole_center_3d)
            
            # 구멍 윤곽선 점들 추가
            for point in hole_3d:
                all_vertices.append(point)
            
            # 구멍 삼각형 생성 (일반적으로 구멍은 반시계 방향으로 그려지므로, 외부와 반대로 삼각형을 생성)
            n_hole = len(hole_3d)
            for j in range(n_hole):
                all_triangles.append([vertex_offset, vertex_offset + ((j + 1) % n_hole) + 1, vertex_offset + j + 1]) # 순서 변경
            
            vertex_offset = len(all_vertices)
    
    # 메쉬 생성
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
    mesh.compute_vertex_normals()
    
    return mesh

def calculate_coordinate_mapping(floor_mesh, top_contour_points, min_proj_x, min_proj_z, scale_factor, img_height):
    """
    바닥 메쉬와 바닥 위 메쉬의 좌표계를 정확하게 매핑합니다.
    """
    if floor_mesh is None or len(floor_mesh.vertices) == 0:
        print("⚠️ Floor mesh is empty, using default mapping")
        return 0.0, 0.0, {}
    
    # 바닥 메쉬의 경계 계산
    floor_bounds = floor_mesh.get_axis_aligned_bounding_box()
    floor_x_min, floor_x_max = floor_bounds.min_bound[0], floor_bounds.max_bound[0]
    floor_z_min, floor_z_max = floor_bounds.min_bound[2], floor_bounds.max_bound[2]
    floor_y_level = floor_bounds.min_bound[1]
    
    # 바닥 메쉬의 중심점
    floor_center_x = (floor_x_min + floor_x_max) / 2
    floor_center_z = (floor_z_min + floor_z_max) / 2
    
    # 바닥 위 윤곽선을 월드 좌표로 변환
    top_contour_world = np.array([
        pixel_to_world_top(px, py, min_proj_x, min_proj_z, scale_factor, img_height)
        for (px, py) in top_contour_points
    ])
    
    # 바닥 위 윤곽선의 중심점
    top_center_x = np.mean(top_contour_world[:, 0])
    top_center_z = np.mean(top_contour_world[:, 1])
    
    # 변환값 계산
    translation_x = floor_center_x - top_center_x
    translation_z = floor_center_z - top_center_z
    
    # 매핑 정보 수집
    mapping_info = {
        'floor_bounds': {
            'x_min': floor_x_min, 'x_max': floor_x_max,
            'z_min': floor_z_min, 'z_max': floor_z_max,
            'y_level': floor_y_level
        },
        'floor_center': (floor_center_x, floor_center_z),
        'top_center': (top_center_x, top_center_z),
        'translation': (translation_x, translation_z),
        'scale_factor': scale_factor,
        'projection_params': {
            'min_x': min_proj_x, 'min_z': min_proj_z,
            'img_height': img_height
        }
    }
    
    print(f"📐 Coordinate Mapping Info:")
    print(f"   Floor bounds: X[{floor_x_min:.4f}, {floor_x_max:.4f}], Z[{floor_z_min:.4f}, {floor_z_max:.4f}]")
    print(f"   Floor center: ({floor_center_x:.4f}, {floor_center_z:.4f})")
    print(f"   Top center: ({top_center_x:.4f}, {top_center_z:.4f})")
    print(f"   Translation: ({translation_x:.4f}, {translation_z:.4f})")
    
    return translation_x, translation_z, mapping_info

def load_pixel_to_points_mapping(pkl_path):
    """
    pkl 파일에서 픽셀-점 매핑 정보를 로드합니다.
    """
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
    """
    pkl 파일들을 사용해서 바닥과 바닥 위 메쉬의 좌표계를 정확하게 매핑합니다.
    원본 위치를 유지하도록 수정.
    """
    # pkl 파일들 로드
    floor_pixel_to_points = load_pixel_to_points_mapping(floor_pkl_path)
    top_pixel_to_points = load_pixel_to_points_mapping(top_pkl_path)
    
    if not floor_pixel_to_points or not top_pixel_to_points:
        print("⚠️ Could not load pixel-to-points mappings")
        return 0.0, 0.0, {}
    
    # 바닥 점들의 3D 좌표 추출
    floor_points_3d = []
    for pixel_coord, points_list in floor_pixel_to_points.items():
        for point in points_list:
            floor_points_3d.append(point)
    
    if not floor_points_3d:
        print("⚠️ No floor points found in pkl file")
        return 0.0, 0.0, {}
    
    floor_points_3d = np.array(floor_points_3d)
    
    # 바닥 점들의 경계 계산
    floor_x_min, floor_x_max = np.min(floor_points_3d[:, 0]), np.max(floor_points_3d[:, 0])
    floor_z_min, floor_z_max = np.min(floor_points_3d[:, 2]), np.max(floor_points_3d[:, 2])
    floor_y_level = np.min(floor_points_3d[:, 1])
    
    # 바닥 위 점들의 3D 좌표 추출
    top_points_3d = []
    for pixel_coord, points_list in top_pixel_to_points.items():
        for point in points_list:
            top_points_3d.append(point)
    
    if not top_points_3d:
        print("⚠️ No top points found in pkl file")
        return 0.0, 0.0, {}
    
    top_points_3d = np.array(top_points_3d)
    
    # 바닥 위 점들의 경계 계산
    top_x_min, top_x_max = np.min(top_points_3d[:, 0]), np.max(top_points_3d[:, 0])
    top_z_min, top_z_max = np.min(top_points_3d[:, 2]), np.max(top_points_3d[:, 2])
    
    # 원본 위치를 유지하기 위해 바닥의 실제 위치를 기준으로 설정
    # 바닥 위 물체들이 바닥의 실제 위치에 맞춰지도록 함
    floor_center_x = (floor_x_min + floor_x_max) / 2
    floor_center_z = (floor_z_min + floor_z_max) / 2
    
    # 바닥 위 물체들의 중심점
    top_center_x = (top_x_min + top_x_max) / 2
    top_center_z = (top_z_min + top_z_max) / 2
    
    # 바닥 위 물체들을 바닥의 실제 위치에 맞춤
    # 즉, 바닥 위 물체들이 바닥 위에 그대로 위치하도록 함
    translation_x = 0.0  # X 방향 이동 없음 (원본 위치 유지)
    translation_z = 0.0  # Z 방향 이동 없음 (원본 위치 유지)
    
    # 매핑 정보 수집
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

def main():
    min_proj_x_top = -51.25127410888672
    min_proj_z_top = -33.814571380615234
    scale_factor_top = 5
    img_height_top = 163
    fixed_height = 10.0
    z_fighting_offset = 0.001 # Z-fighting 방지를 위한 미세 오프셋
    top_img = "output/morph/top_view/morph_smoothed.png"

    # 기존 contour
    try:
        top_corners_px_raw = extract_corners(top_img, epsilon_ratio=0.0)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        print("Please ensure 'output/morph/top_view/morph_smoothed.png' exists.")
        return

    # 윤곽선 내부 픽셀까지 다 추출
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

    # 기존 contour world 좌표 변환 (for side)
    top_corners_world_initial = np.array([
        pixel_to_world_top(px, py, min_proj_x_top, min_proj_z_top, scale_factor_top, img_height_top)
        for (px, py) in top_corners_px_raw
    ])
    
    # pkl 파일을 사용한 정확한 좌표계 매핑 계산
    floor_pkl_path = "output/outline/floor_plane/pixel_to_points.pkl"
    top_pkl_path = "output/outline/top_view/pixel_to_points.pkl"
    
    translation_x, translation_z, mapping_info = calculate_coordinate_mapping_from_pkl(
        floor_pkl_path, top_pkl_path
    )
    
    # 매핑 정보 저장
    if mapping_info:
        mapping_file = "output/coordinate_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(mapping_info, f, indent=2)
        print(f"💾 Coordinate mapping saved to: {mapping_file}")
    
    # 기존 방식 (fallback)
    if not mapping_info:
        print("⚠️ Using fallback coordinate mapping")
        top_contour_centroid_x = np.mean(top_corners_world_initial[:, 0])
        top_contour_centroid_z = np.mean(top_corners_world_initial[:, 1])
        floor_centroid_x = (floor_x_min + floor_x_max) / 2 if not np.isnan(floor_x_min) else top_contour_centroid_x
        floor_centroid_z = (floor_z_min + floor_z_max) / 2 if not np.isnan(floor_z_min) else top_contour_centroid_z
        translation_x = floor_centroid_x - top_contour_centroid_x
        translation_z = floor_centroid_z - top_contour_centroid_z

    # --- Solid extrude용 내부 픽셀의 월드 좌표 변환 및 translation 적용 ----
    # 이 부분에서 translation_x, translation_z를 한 번만 적용합니다.
    filled_points_world = np.array([
        pixel_to_world_top(px, py, min_proj_x_top, min_proj_z_top, scale_factor_top, img_height_top)
        for (px, py) in filled_pixels
    ])
    filled_points_world[:, 0] += translation_x
    filled_points_world[:, 1] += translation_z

    # --- side는 기존 contour로만 만든다 ---
    num_resample_points = max(200, len(top_corners_px_raw))
    top_corners_world_aligned = top_corners_world_initial.copy()
    top_corners_world_aligned[:, 0] += translation_x
    top_corners_world_aligned[:, 1] += translation_z
    top_resampled = resample_contour(top_corners_world_aligned, num_resample_points)

    # Alpha 파라미터 계산
    distances = np.linalg.norm(np.diff(top_resampled, axis=0, append=top_resampled[:1]), axis=1)
    avg_dist = np.mean(distances)
    alpha_value = avg_dist * 0.05  # 더 작은 alpha 값으로 더 조밀한 메쉬 생성
    jitter_strength = avg_dist * 0.005 # 더 작은 jitter로 안정성 향상

    # --- 내부를 가득 채운 바닥(base)/윗면(top) 점 생성 ----
    # 점들의 밀도를 높이기 위해 더 많은 점 생성
    num_filled_points = len(filled_points_world)
    print(f"Number of filled pixels: {num_filled_points}")
    
    # 점들이 너무 적으면 보간으로 추가 점 생성
    if num_filled_points < 1000:
        # 윤곽선을 따라 더 많은 점 생성
        contour_points = top_resampled
        num_contour_points = len(contour_points)
        additional_points = []
        
        for i in range(num_contour_points):
            p1 = contour_points[i]
            p2 = contour_points[(i + 1) % num_contour_points]
            
            # 두 점 사이에 5개의 추가 점 생성
            for t in np.linspace(0, 1, 6)[1:-1]:  # 0과 1 제외
                interpolated_point = p1 * (1 - t) + p2 * t
                # 내부로 약간 이동
                center = np.mean(contour_points, axis=0)
                direction = center - interpolated_point
                direction = direction / np.linalg.norm(direction)
                inward_point = interpolated_point + direction * (avg_dist * 0.1)
                additional_points.append(inward_point)
        
        if additional_points:
            additional_points = np.array(additional_points)
            filled_points_world = np.vstack([filled_points_world, additional_points])
            print(f"Added {len(additional_points)} additional points. Total: {len(filled_points_world)}")

    base_points_3d = np.column_stack([
        filled_points_world[:, 0],
        np.full(len(filled_points_world), floor_y_level + z_fighting_offset),
        filled_points_world[:, 1]
    ])
    top_points_3d = np.column_stack([
        filled_points_world[:, 0],
        np.full(len(filled_points_world), floor_y_level + fixed_height + z_fighting_offset),
        filled_points_world[:, 1]
    ])

    # --- 바닥/천장 메쉬 생성 (Alpha shape 제거, 윤곽선 기반만 사용) ---
    print("🔧 Creating base and top meshes using contour-based method...")
    
    # 바닥 메쉬 생성
    try:
        base_mesh = create_complete_filled_mesh(
            top_img, floor_y_level + z_fighting_offset, 
            min_proj_x_top, min_proj_z_top, scale_factor_top, img_height_top,
            translation_x, translation_z  # translation을 함수 내부에서 적용
        )
        base_mesh.compute_vertex_normals()
        print(f"Solid base mesh: {len(base_mesh.vertices)} vertices, {len(base_mesh.triangles)} triangles")
    except Exception as e:
        print(f"❌ Error creating base mesh: {e}")
        base_mesh = o3d.geometry.TriangleMesh()

    # 천장 메쉬 생성
    try:
        top_mesh = create_complete_filled_mesh(
            top_img, floor_y_level + fixed_height + z_fighting_offset, 
            min_proj_x_top, min_proj_z_top, scale_factor_top, img_height_top,
            translation_x, translation_z  # translation을 함수 내부에서 적용
        )
        top_mesh.compute_vertex_normals()
        print(f"Solid top mesh: {len(top_mesh.vertices)} vertices, {len(top_mesh.triangles)} triangles")
    except Exception as e:
        print(f"❌ Error creating top mesh: {e}")
        top_mesh = o3d.geometry.TriangleMesh()

    # --- Side face (벽 메쉬) ---
    # 측면 메쉬도 천장/바닥과 동일한 좌표계 사용
    print("🔧 Creating side mesh with consistent coordinate system...")
    
    # 측면 메쉬를 위한 윤곽선 추출 (천장과 동일한 방식)
    try:
        # 벽은 윤곽선을 따라 extrude하므로 별도 처리
        # 하지만 좌표계는 천장/바닥과 동일하게 유지
        side_contour = extract_corners(top_img, epsilon_ratio=0.0)
        
        # create_complete_filled_mesh 함수와 동일한 방식으로 좌표 변환
        side_contour_world = np.array([
            pixel_to_world_top(px, py, min_proj_x_top, min_proj_z_top, scale_factor_top, img_height_top)
            for (px, py) in side_contour
        ])
        # translation 적용 (천장/바닥과 동일)
        side_contour_world[:, 0] += translation_x
        side_contour_world[:, 1] += translation_z
        
        # 측면 메쉬 생성 (윤곽선을 따라 extrude)
        n = len(side_contour_world)
        extruded_top_outline_3d = np.column_stack([
            side_contour_world[:, 0],
            np.full(len(side_contour_world), floor_y_level + fixed_height + z_fighting_offset),
            side_contour_world[:, 1]
        ])
        extruded_base_outline_3d = np.column_stack([
            side_contour_world[:, 0],
            np.full(len(side_contour_world), floor_y_level + z_fighting_offset),
            side_contour_world[:, 1]
        ])
        
        # 측면 메쉬 생성 (윤곽선을 따라)
        side_vertices = np.vstack([extruded_base_outline_3d, extruded_top_outline_3d])
        side_triangles = []
        for i in range(n):
            side_triangles.append([i, (i + 1) % n, (i + 1) % n + n])
            side_triangles.append([i, (i + 1) % n + n, i + n])
        
        extruded_mesh_sides = o3d.geometry.TriangleMesh()
        extruded_mesh_sides.vertices = o3d.utility.Vector3dVector(side_vertices)
        extruded_mesh_sides.triangles = o3d.utility.Vector3iVector(side_triangles)
        extruded_mesh_sides.compute_vertex_normals()
        print(f"Side mesh: {len(extruded_mesh_sides.vertices)} vertices, {len(extruded_mesh_sides.triangles)} triangles")
        
    except Exception as e:
        print(f"❌ Error creating side mesh: {e}")
        # 빈 메쉬 생성
        extruded_mesh_sides = o3d.geometry.TriangleMesh()

    # --- 최종 solid extrude mesh 합치기 ---
    solid_extruded_mesh = base_mesh + top_mesh + extruded_mesh_sides
    solid_extruded_mesh.paint_uniform_color([0.1, 0.7, 0.1])

    output_extruded_mesh_filename = "output/pcd/solid_extruded_top_view_mesh.ply"
    o3d.io.write_triangle_mesh(output_extruded_mesh_filename, solid_extruded_mesh)
    print(f"✅ Solid extruded mesh saved: {output_extruded_mesh_filename}")

    # --- Sphere debug용 (optional) ---
    sphere_meshes = []
    sphere_radius = 1.0 # 스피어 크기를 좀 더 현실적으로 조정
    sphere_color = [1, 0, 0]
    top_corners_world_for_spheres = np.array([
        pixel_to_world_top(px, py, min_proj_x_top, min_proj_z_top, scale_factor_top, img_height_top)
        for (px, py) in top_corners_px_raw
    ])
    top_corners_world_for_spheres[:, 0] += translation_x
    top_corners_world_for_spheres[:, 1] += translation_z
    for (x, z) in top_corners_world_for_spheres:
        v = np.array([x, floor_y_level + fixed_height + z_fighting_offset, z]) 
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(v)
        sphere.paint_uniform_color(sphere_color)
        sphere_meshes.append(sphere)
    # ----------------------------------

    # --- 바닥 + extrude 합치기, 시각화 ---
    if solid_extruded_mesh is not None and mesh_floor is not None:
        mesh_floor.paint_uniform_color([0.1, 0.1, 0.7])
        print("🔗 [Merging] Merging meshes...")
        merged_mesh = solid_extruded_mesh + mesh_floor
        merged_output = "output/mesh/merged_result_solid.ply"
        o3d.io.write_triangle_mesh(merged_output, merged_mesh)
        print(f"🎉 Merged mesh saved: {merged_output} ({len(merged_mesh.vertices)} vertices, {len(merged_mesh.triangles)} triangles)")
        print("👁️ Visualizing merged mesh + spheres...")
        o3d.visualization.draw_geometries([merged_mesh] + sphere_meshes, mesh_show_back_face=True)
    else:
        print("⚠️ Warning: Could not generate extruded mesh or floor mesh. Skipping merge.")
        if solid_extruded_mesh is not None:
            print("👁️ Visualizing only extruded mesh + spheres...")
            o3d.visualization.draw_geometries([solid_extruded_mesh] + sphere_meshes, mesh_show_back_face=True)
        elif mesh_floor is not None:
            print("👁️ Visualizing only floor mesh...")
            o3d.visualization.draw_geometries([mesh_floor], mesh_show_back_face=True)

if __name__ == "__main__":
    main()