import open3d as o3d
import numpy as np
import os

def load_and_ransac_multiple_planes(pcd_path, output_dir_base, max_planes=5,
                                    distance_threshold=0.02, ransac_n=3,
                                    num_iterations=1000, min_inliers_ratio=0.01,
                                    vertical_offset=0.0): # <-- vertical_offset 파라미터
    """
    3D Gaussian Splatting 모델 (PLY 또는 PCD 파일)을 불러와 RANSAC을 이용해 여러 평면을 검출하고,
    바닥 평면들을 하나의 파일로 합치고, 바닥을 제외한 부분도 별도 파일로 저장합니다.
    추가로 개별 바닥 평면들도 별도 파일로 저장합니다.

    Args:
        pcd_path (str): Gaussian Splatting 모델의 .pcd 또는 .ply 파일 경로
        output_dir_base (str): 결과 파일들을 저장할 기본 디렉토리 경로
        max_planes (int): 검출할 최대 평면의 수
        distance_threshold (float): 평면에 포함될 포인트와 모델 평면 간의 최대 거리.
        ransac_n (int): 평면 모델을 만들기 위해 샘플링할 최소 포인트 수 (보통 3개).
        num_iterations (int): RANSAC 알고리즘의 최대 반복 횟수.
        min_inliers_ratio (float): 추출된 inlier 수가 전체 포인트의 이 비율보다 작으면 평면 검출을 중단합니다.
        vertical_offset (float): 바닥 평면의 상한선을 수직으로 얼마나 더 올릴지 (미터 단위).
                                 양수 값은 바닥 영역을 위로 확장하여 바닥 제외 부분의 시작점을 높입니다.
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
    original_pcd = pcd # 원본 포인트 클라우드 보존
    original_total_points = len(pcd.points)

    # 원본 포인트 클라우드 시각화 (RANSAC 전)
    print("Visualizing original point cloud...")
    o3d.visualization.draw_geometries([original_pcd], window_name="Original Point Cloud")


    # --- RANSAC을 위해 현재 처리할 포인트 클라우드와 해당 포인트의 원본 인덱스 리스트를 초기화합니다. ---
    current_pcd_for_ransac = original_pcd
    current_original_indices = np.arange(original_total_points) # 처음에는 모든 원본 인덱스를 가집니다.

    plane_count = 0
    all_inlier_clouds = []
    floor_planes = []  # 바닥 평면들의 정보를 저장할 리스트

    # 출력 디렉토리 생성
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)
        print(f"Created output directory: {output_dir_base}")

    print(f"\nStarting RANSAC for multiple plane segmentation with parameters:")
    print(f"   Max Planes: {max_planes}")
    print(f"   Distance Threshold: {distance_threshold}")
    print(f"   RANSAC N: {ransac_n}")
    print(f"   Number of Iterations: {num_iterations}")
    print(f"   Minimum Inliers Ratio: {min_inliers_ratio}")
    print(f"   Vertical Offset for Floor: {vertical_offset:.3f} m") # <-- 추가된 파라미터 출력

    # --- 1. 다중 평면 검출 및 바닥 평면 식별 ---
    while plane_count < max_planes and len(current_pcd_for_ransac.points) > 0:
        print(f"\n--- Searching for Plane {plane_count + 1} ---")
        print(f"Current point cloud size for RANSAC: {len(current_pcd_for_ransac.points)} points.")

        # RANSAC을 이용한 평면 검출
        # `inliers`는 `current_pcd_for_ransac` 내에서의 인덱스입니다.
        plane_model, inliers = current_pcd_for_ransac.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        if len(inliers) == 0:
            print(f"No more significant planes found (0 inliers). Stopping.")
            break

        # 추출된 inlier 수가 너무 적으면 중단 (원본 포인트 대비 비율)
        if len(inliers) < original_total_points * min_inliers_ratio:
            print(f"Number of inliers ({len(inliers)}) is less than {min_inliers_ratio*100:.2f}% of total points ({original_total_points}). Stopping.")
            break

        # `inliers`는 `current_pcd_for_ransac`의 인덱스이므로,
        # 이를 `original_pcd`의 인덱스로 매핑해야 합니다.
        inliers_in_original_pcd = current_original_indices[inliers]


        [a, b, c, d] = plane_model
        print(f"Plane {plane_count + 1} equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        print(f"Number of inliers found for Plane {plane_count + 1}: {len(inliers_in_original_pcd)} (in original PCD)")

        # Inlier 포인트 클라우드 생성 (원본 PCD에서 인덱스를 사용하여)
        inlier_cloud = original_pcd.select_by_index(inliers_in_original_pcd)

        # 평면의 법선 벡터 계산
        normal_vector = np.array([a, b, c])
        normal_vector = normal_vector / np.linalg.norm(normal_vector)  # 정규화

        # 평면의 중심점 계산
        inlier_points = np.asarray(inlier_cloud.points)
        plane_center = np.mean(inlier_points, axis=0)

        print(f"Plane {plane_count + 1} normal vector: {normal_vector}")
        print(f"Plane {plane_count + 1} center point: {plane_center}")

        # --- 2. 바닥 평면 식별 (Y축과 거의 평행한 평면) ---
        # Y축 방향 벡터 [0, 1, 0]과 평면 법선 벡터의 각도 계산
        y_axis = np.array([0, 1, 0])
        angle_with_y = np.arccos(np.abs(np.dot(normal_vector, y_axis)))
        angle_degrees = np.degrees(angle_with_y)

        print(f"Plane {plane_count + 1} angle with Y-axis: {angle_degrees:.2f} degrees")

        # 각도가 30도 이하이면 바닥 평면으로 간주
        is_floor_plane = angle_degrees <= 30.0

        if is_floor_plane:
            print(f"Plane {plane_count + 1} is identified as a FLOOR plane.")
            floor_planes.append({
                'plane_model': plane_model,
                'inlier_cloud': inlier_cloud,
                'inlier_indices': inliers_in_original_pcd,
                'center': plane_center,
                'normal': normal_vector,
                'angle_with_y': angle_degrees
            })
        else:
            print(f"Plane {plane_count + 1} is identified as a NON-FLOOR plane.")

        # 모든 평면을 시각화용 리스트에 추가 (색상 구분)
        if is_floor_plane:
            inlier_cloud.paint_uniform_color([0.0, 1.0, 0.0])  # 바닥 평면은 초록색
        else:
            inlier_cloud.paint_uniform_color([1.0, 0.0, 0.0])  # 비바닥 평면은 빨간색

        all_inlier_clouds.append(inlier_cloud)

        # 현재 평면의 inlier들을 제거하고 다음 평면 검출을 위해 업데이트
        remaining_indices = np.setdiff1d(current_original_indices, inliers_in_original_pcd)
        current_pcd_for_ransac = original_pcd.select_by_index(remaining_indices)
        current_original_indices = remaining_indices

        plane_count += 1

    print(f"\nRANSAC completed. Found {plane_count} planes total, {len(floor_planes)} of which are floor planes.")

    # --- 3. 바닥 평면들 통합 및 저장 ---
    if len(floor_planes) > 0:
        print(f"\n--- Processing {len(floor_planes)} floor planes ---")
        
        # 모든 바닥 평면의 inlier 인덱스를 수집
        all_floor_indices = []
        for floor_plane in floor_planes:
            all_floor_indices.extend(floor_plane['inlier_indices'])
        
        all_floor_indices = np.unique(all_floor_indices)
        print(f"Total unique floor points: {len(all_floor_indices)}")

        # 바닥 평면 포인트 클라우드 생성
        floor_cloud = original_pcd.select_by_index(all_floor_indices)
        floor_cloud.paint_uniform_color([0.0, 1.0, 0.0])  # 초록색

        # 바닥 평면의 Y 좌표 범위 계산
        floor_points = np.asarray(floor_cloud.points)
        floor_y_min = np.min(floor_points[:, 1])
        floor_y_max = np.max(floor_points[:, 1])
        floor_y_range = floor_y_max - floor_y_min

        print(f"Floor Y range: {floor_y_min:.4f} to {floor_y_max:.4f} (range: {floor_y_range:.4f})")

        # vertical_offset 적용하여 바닥 영역 확장
        expanded_floor_y_max = floor_y_max + vertical_offset
        print(f"Expanded floor Y max (with {vertical_offset:.3f} offset): {expanded_floor_y_max:.4f}")

        # 확장된 바닥 영역에 포함되는 모든 포인트 찾기
        all_points = np.asarray(original_pcd.points)
        expanded_floor_mask = all_points[:, 1] <= expanded_floor_y_max
        expanded_floor_indices = np.where(expanded_floor_mask)[0]

        # 확장된 바닥 포인트 클라우드 생성
        expanded_floor_cloud = original_pcd.select_by_index(expanded_floor_indices)
        expanded_floor_cloud.paint_uniform_color([0.0, 1.0, 0.0])  # 초록색

        print(f"Expanded floor points: {len(expanded_floor_indices)} (original floor: {len(all_floor_indices)})")

        # 바닥 평면 파일 저장 (원본 바닥 평면들만)
        floor_output_path = os.path.join(output_dir_base, "floor_plane.pcd")
        try:
            o3d.io.write_point_cloud(floor_output_path, floor_cloud)
            print(f"[SUCCESS] 바닥 평면 포인트 클라우드 저장됨: {floor_output_path} ({len(floor_cloud.points)} 포인트)")
        except Exception as e:
            print(f"[ERROR] 바닥 평면 파일 저장 중 오류: {e}")

        # 확장된 바닥 영역 파일 저장
        expanded_floor_output_path = os.path.join(output_dir_base, "expanded_floor_plane.pcd")
        try:
            o3d.io.write_point_cloud(expanded_floor_output_path, expanded_floor_cloud)
            print(f"[SUCCESS] 확장된 바닥 영역 포인트 클라우드 저장됨: {expanded_floor_output_path} ({len(expanded_floor_cloud.points)} 포인트)")
        except Exception as e:
            print(f"[ERROR] 확장된 바닥 영역 파일 저장 중 오류: {e}")

        # 바닥 제외 포인트 클라우드 생성
        non_floor_indices = np.setdiff1d(np.arange(original_total_points), expanded_floor_indices)
        non_floor_cloud = original_pcd.select_by_index(non_floor_indices)
        non_floor_cloud.paint_uniform_color([1.0, 0.0, 0.0])  # 빨간색

        # 바닥 제외 파일 저장
        non_floor_output_path = os.path.join(output_dir_base, "topview.pcd")
        try:
            o3d.io.write_point_cloud(non_floor_output_path, non_floor_cloud)
            print(f"[SUCCESS] 바닥 제외 포인트 클라우드 저장됨: {non_floor_output_path} ({len(non_floor_cloud.points)} 포인트)")
        except Exception as e:
            print(f"[ERROR] 바닥 제외 파일 저장 중 오류: {e}")


    # --- 5. 시각화 ---
    # 모든 추출된 평면과 나머지 포인트 시각화
    if len(all_inlier_clouds) > 0:
        # RANSAC 후 남은 current_pcd_for_ransac (어떤 평면에도 속하지 않은 노이즈)도 시각화에 포함
        if len(current_pcd_for_ransac.points) > 0:
            current_pcd_for_ransac.paint_uniform_color([0.3, 0.3, 0.3]) # 회색으로 표시
            all_inlier_clouds.append(current_pcd_for_ransac)

        print("\nVisualizing all detected planes and remaining points from RANSAC process...")
        o3d.visualization.draw_geometries(
            all_inlier_clouds,
            window_name="RANSAC Multiple Plane Segmentation (All Detected Planes)"
        )
    else:
        print("No planes were detected by RANSAC.")

    # 바닥과 바닥 제외 부분을 함께 시각화
    visualization_clouds_final_segmentation = []
    if floor_cloud is not None:
        visualization_clouds_final_segmentation.append(floor_cloud)
    if non_floor_cloud is not None:
        visualization_clouds_final_segmentation.append(non_floor_cloud)
    
    if visualization_clouds_final_segmentation:
        print("\nVisualizing final Floor vs Non-Floor Segmentation...")
        o3d.visualization.draw_geometries(
            visualization_clouds_final_segmentation,
            window_name="Floor vs Non-Floor Segmentation"
        )
    else:
        print("No clouds to visualize for Floor vs Non-Floor segmentation.")

    # --- floor_plane.pcd(모든 바닥 평면 통합본) 시각화 ---
    floor_plane_path = os.path.join(output_dir_base, "floor_plane.pcd")
    if os.path.exists(floor_plane_path):
        try:
            floor_plane_cloud = o3d.io.read_point_cloud(floor_plane_path)
            print("\nVisualizing saved floor_plane.pcd (모든 바닥 평면 통합본)...")
            o3d.visualization.draw_geometries(
                [floor_plane_cloud],
                window_name="Saved Floor Plane (floor_plane.pcd)"
            )
        except Exception as e:
            print(f"[ERROR] floor_plane.pcd 시각화 중 오류: {e}")
    else:
        print("floor_plane.pcd 파일이 존재하지 않아 시각화하지 않습니다.")


if __name__ == "__main__":
    # 기존 코드 (nonowall 적용된 PCD)
    # input_pcd_file = "input/3BP_CS_model_Cloud.pcd"
    input_pcd_file = "../output/nonowall.pcd"
    output_base_directory = "../output/ransac"
    
    max_planes_to_find = 40
    min_inliers_ratio_threshold = 0.01

    # ======================================================================
    # ⭐ 바닥 윗부분의 가장 아랫부분을 조절하는 파라미터 (미터 단위) ⭐
    # 이 값을 양수로 늘리면 바닥 영역이 위로 확장되어,
    # 바닥 제외 (non_floor) 부분의 시작점이 더 높아집니다.
    # 즉, 바닥과 객체 간의 지저분한 연결 부분을 '잘라내는' 효과가 있습니다.
    # 0.01 (1cm), 0.02 (2cm), 0.05 (5cm), 0.1 (10cm) 등 작은 값부터 시작하여 테스트해보세요.
    # ======================================================================
    VERTICAL_OFFSET_topview = 1.5  # 5 -> 바닥 싹사라짐

    load_and_ransac_multiple_planes(
        input_pcd_file,
        output_base_directory,
        max_planes=max_planes_to_find,
        distance_threshold=0.02, # RANSAC 평면 감지 정밀도
        ransac_n=3,
        num_iterations=1000,
        min_inliers_ratio=min_inliers_ratio_threshold,
        vertical_offset=VERTICAL_OFFSET_topview # <-- 추가된 파라미터 전달
    )

    # --- 바닥 평면만 원본 PCD에서 별도 추출 ---
    input_pcd_file_original = "../output/nonowall.pcd"  # nonowall 적용 안 된 원본
    output_base_directory_floor_orig = "../output/ransac/floor"
    if not os.path.exists(output_base_directory_floor_orig):
        os.makedirs(output_base_directory_floor_orig)
    print("\n[원본에서 바닥 평면만 별도 추출]")
    load_and_ransac_multiple_planes(
        input_pcd_file_original,
        output_base_directory_floor_orig,
        max_planes=10,  # 바닥 평면만 추출할 것이므로 적당히
        distance_threshold=0.02,
        ransac_n=3,
        num_iterations=1000,
        min_inliers_ratio=0.01,
        vertical_offset=0.0  # 바닥만 추출할 때는 오프셋 없이
    ) 