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

        # 바닥 평면인지 판단 (정규 벡터가 특정 축과 얼마나 평행한지)
        normal_vector = np.array([a, b, c])
        normal_vector = normal_vector / np.linalg.norm(normal_vector) # 정규화
        z_axis = np.array([0, 0, 1])
        y_axis = np.array([0, 1, 0])
        x_axis = np.array([1, 0, 0])

        # 법선 벡터와 각 축과의 각도를 계산 (절대값 사용)
        angle_with_z = np.arccos(np.abs(np.dot(normal_vector, z_axis))) * 180 / np.pi
        angle_with_y = np.arccos(np.abs(np.dot(normal_vector, y_axis))) * 180 / np.pi
        angle_with_x = np.arccos(np.abs(np.dot(normal_vector, x_axis))) * 180 / np.pi

        print(f"   평면 {plane_count + 1}의 법선 벡터: [{a:.3f}, {b:.3f}, {c:.3f}]")
        print(f"   X축과의 각도: {angle_with_x:.1f}도, Y축과의 각도: {angle_with_y:.1f}도, Z축과의 각도: {angle_with_z:.1f}도")

        # 각 평면을 다른 색상으로 표시 (랜덤 색상)
        inlier_cloud.paint_uniform_color(np.random.rand(3))
        all_inlier_clouds.append(inlier_cloud)

        # 바닥 평면으로 판단 기준: 어느 축과의 각도가 25도 이하면 해당 축에 수직인 평면으로 간주
        # 그리고 해당 평면의 평균 높이를 계산하여 저장
        is_floor = False
        avg_coord = 0.0
        height_axis = ""

        if angle_with_z <= 25: # Z축과 평행 (바닥/천장)
            is_floor = True
            avg_coord = np.mean(np.asarray(inlier_cloud.points)[:, 2])
            height_axis = "Z"
        elif angle_with_y <= 25: # Y축과 평행 (벽/바닥 - 모델에 따라)
            is_floor = True
            avg_coord = np.mean(np.asarray(inlier_cloud.points)[:, 1])
            height_axis = "Y"
        elif angle_with_x <= 25: # X축과 평행 (벽/바닥 - 모델에 따라)
            is_floor = True
            avg_coord = np.mean(np.asarray(inlier_cloud.points)[:, 0])
            height_axis = "X"

        if is_floor:
            print(f"   → 평면 {plane_count + 1}을 바닥 후보로 분류 (최소 각도: {min(angle_with_x, angle_with_y, angle_with_z):.1f}도)")
            floor_planes.append({
                'plane_model': plane_model,
                'inliers_indices_original_pcd': inliers_in_original_pcd, # 원본 pcd에서의 인덱스 저장
                'avg_coord': avg_coord, # 'avg_height'를 'avg_coord'로 변경하여 일반화
                'plane_id': plane_count + 1,
                'height_axis': height_axis
            })
            print(f"     평면 {plane_count + 1}의 평균 좌표 ({height_axis}축): {avg_coord:.3f}")
        else:
            print(f"   → 평면 {plane_count + 1}을 벽면/기타 평면으로 분류 (최소 각도: {min(angle_with_x, angle_with_y, angle_with_z):.1f}도)")

        # Outlier 포인트 클라우드 업데이트 (다음 반복을 위해 inlier를 제거)
        # 중요: current_original_indices도 함께 업데이트해야 합니다.
        outliers_indices_relative_to_current = np.setdiff1d(np.arange(len(current_pcd_for_ransac.points)), inliers)
        current_pcd_for_ransac = current_pcd_for_ransac.select_by_index(outliers_indices_relative_to_current)
        current_original_indices = current_original_indices[outliers_indices_relative_to_current] # 원본 인덱스도 업데이트

        plane_count += 1

    print(f"\nFinished finding {plane_count} planes.")

    # --- 2. 모든 개별 바닥 평면 통합 저장 ---
    if floor_planes:
        # 모든 개별 평면의 포인트 인덱스를 합치기
        all_individual_indices = []
        for floor_plane in floor_planes:
            all_individual_indices.extend(floor_plane['inliers_indices_original_pcd'])
        
        # 중복 제거
        all_individual_indices = list(set(all_individual_indices))
        
        # 전체 합쳐진 포인트 클라우드 생성
        all_individual_floor_cloud = original_pcd.select_by_index(all_individual_indices)
        
        # 전체 합쳐진 파일 저장 (ransac 폴더 바로 안에)
        all_individual_filename = "floor_plane.pcd"
        all_individual_filepath = os.path.join(output_dir_base, all_individual_filename)
        
        try:
            o3d.io.write_point_cloud(all_individual_filepath, all_individual_floor_cloud)
            print(f"✅ 모든 개별 바닥 평면 통합 파일 저장됨: {all_individual_filename} ({len(all_individual_indices)} 포인트, {len(floor_planes)}개 평면)")
        except Exception as e:
            print(f"❌ 전체 개별 평면 통합 파일 저장 중 오류: {e}")

    # --- 3. 바닥 평면 합치기 ---
    print(f"\n=== 바닥 처리 시작 ===")
    print(f"검출된 바닥 후보 평면 수: {len(floor_planes)}개")

    floor_cloud = o3d.geometry.PointCloud() # 통합된 바닥 포인트 클라우드
    non_floor_cloud = o3d.geometry.PointCloud() # 바닥 제외 포인트 클라우드
    
    if floor_planes:
        # 각 바닥 평면의 정보 출력
        for i, floor_plane in enumerate(floor_planes):
            print(f"   바닥 후보 평면 {i+1}: 평면 {floor_plane['plane_id']}, 좌표({floor_plane['height_axis']}축): {floor_plane['avg_coord']:.3f}")

        # 가장 낮은 평균 좌표를 가진 평면을 실제 바닥으로 간주 (아래를 향하는 좌표계 기준)
        # 만약 Z축이 위로 향하고 있다면, 가장 '낮은' Z값이 바닥이 됩니다.
        # 실제 환경에 따라 Y축 또는 X축이 '높이'를 나타낼 수도 있으므로,
        # 'height_axis'를 기준으로 가장 낮은 좌표를 찾습니다.
        
        # 일단 Z축을 기본으로 가장 낮은 바닥을 찾고, 만약 Z축 바닥이 없다면 다른 축에서 찾도록 로직을 확장할 수 있습니다.
        # 현재 코드는 단순히 '최고 높이'를 찾는 것으로 되어 있으므로, 이를 가장 '낮은' 평면으로 수정하는 것이 더 일반적입니다.
        
        # 여기서는 모델의 Y나 X축이 높이를 나타낼 수 있으므로,
        # 모든 '바닥 후보' 평면 중 가장 '낮은' 좌표값을 가진 평면을 '주 바닥'으로 결정합니다.
        # 즉, min(avg_coord)를 사용합니다.
        
        # 모델의 Z축이 높이인 경우, 가장 낮은 Z가 바닥.
        # 모델의 Y축이 높이인 경우, 가장 낮은 Y가 바닥.
        # 모델의 X축이 높이인 경우, 가장 낮은 X가 바닥.
        
        # 여기서는 단순히 floor_planes 리스트에 있는 모든 평면들 중 'avg_coord'가 가장 작은 것을 선택합니다.
        # 이는 좌표계에 따라 가장 낮은 물리적 위치의 평면을 의미합니다.
        
        main_floor_plane = min(floor_planes, key=lambda x: x['avg_coord'])
        main_coord = main_floor_plane['avg_coord']
        height_axis = main_floor_plane['height_axis']
        print(f"\n기준 바닥 평면: 평면 {main_floor_plane['plane_id']} (좌표({height_axis}축): {main_coord:.3f})")

        # 이제 원본 포인트 클라우드를 기준으로 바닥과 바닥 제외를 분리합니다.
        all_points_np = np.asarray(original_pcd.points)

        # 어떤 축을 사용할지 결정
        if height_axis == "Z":
            height_coord_array = all_points_np[:, 2]
        elif height_axis == "Y":
            height_coord_array = all_points_np[:, 1]
        else:  # X
            height_coord_array = all_points_np[:, 0]
        
        # 바닥 포인트 정의: 기준 바닥 평면의 좌표 이하 + (RANSAC 오차 + vertical_offset)
        # vertical_offset을 추가하여 바닥으로 분류될 상한 좌표를 조절합니다.
        # 'main_coord'를 기준으로 `vertical_offset`만큼 위로 (좌표값이 커지는 방향) 확장합니다.
        floor_mask = height_coord_array <= main_coord + distance_threshold + vertical_offset # <-- vertical_offset 적용
        floor_indices = np.where(floor_mask)[0]
        
        print(f"기준 바닥 평면({height_axis}축: {main_coord:.3f}) 이하 (오프셋 포함) 포인트 수: {len(floor_indices)}개")
        print(f"전체 포인트 수: {original_total_points}개")
        print(f"바닥 비율: {len(floor_indices)/original_total_points*100:.1f}%")
        
        # 바닥 포인트들을 하나의 클라우드로 합치기
        floor_cloud = original_pcd.select_by_index(floor_indices)
        floor_cloud.paint_uniform_color([0.8, 0.8, 0.8])  # 밝은 회색으로 바닥 표시
        
        # 바닥 파일 저장
        floor_output_path = os.path.join(output_dir_base, "under_floor.pcd")
        try:
            o3d.io.write_point_cloud(floor_output_path, floor_cloud)
            print(f"✅ 바닥 포인트 클라우드 저장됨: {floor_output_path} ({len(floor_indices)} 포인트)")
        except Exception as e:
            print(f"❌ 바닥 파일 저장 중 오류: {e}")
    else:
        print("❌ 바닥 평면이 검출되지 않았습니다.")
        floor_cloud = None
        floor_indices = []

    # --- 4. 바닥을 제외한 나머지 포인트 처리 ---
    print(f"\n=== 바닥 제외 처리 시작 ===")
    if floor_cloud is not None and len(floor_indices) > 0:
        # 원본 포인트 클라우드에서 바닥 인덱스를 제외한 모든 포인트를 선택
        non_floor_indices = np.setdiff1d(np.arange(original_total_points), floor_indices)
        
        print(f"바닥 제외 포인트 수: {len(non_floor_indices)}개")
        print(f"바닥 제외 비율: {len(non_floor_indices)/original_total_points*100:.1f}%")
        
        if len(non_floor_indices) > 0:
            non_floor_cloud = original_pcd.select_by_index(non_floor_indices)
            non_floor_cloud.paint_uniform_color([0.2, 0.2, 0.8])  # 파란색으로 표시
            
            # 바닥 제외 파일 저장
            non_floor_output_path = os.path.join(output_dir_base, "above_floor.pcd")
            try:
                o3d.io.write_point_cloud(non_floor_output_path, non_floor_cloud)
                print(f"✅ 바닥 제외 포인트 클라우드 저장됨: {non_floor_output_path} ({len(non_floor_indices)} 포인트)")
            except Exception as e:
                print(f"❌ 바닥 제외 파일 저장 중 오류: {e}")
        else:
            print("❌ 바닥 제외 포인트가 없습니다.")
            non_floor_cloud = None
    else:
        print("❌ 바닥이 없거나 검출되지 않아 모든 포인트를 '바닥 제외'로 처리합니다.")
        non_floor_cloud = o3d.geometry.PointCloud(original_pcd) # 원본 PCD의 복사본을 만들어 색상 변경
        non_floor_cloud.paint_uniform_color([0.2, 0.2, 0.8])
        
        non_floor_output_path = os.path.join(output_dir_base, "above_floor.pcd")
        try:
            o3d.io.write_point_cloud(non_floor_output_path, non_floor_cloud)
            print(f"✅ 바닥 제외 포인트 클라우드 저장됨: {non_floor_output_path} ({original_total_points} 포인트)")
        except Exception as e:
            print(f"❌ 바닥 제외 파일 저장 중 오류: {e}")


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
            print(f"❌ floor_plane.pcd 시각화 중 오류: {e}")
    else:
        print("floor_plane.pcd 파일이 존재하지 않아 시각화하지 않습니다.")


if __name__ == "__main__":
    # 기존 코드 (nonowall 적용된 PCD)
    # input_pcd_file = "input/3BP_CS_model_Cloud.pcd"
    input_pcd_file = "output/nonowall.pcd"
    output_base_directory = "output/ransac"
    
    max_planes_to_find = 40
    min_inliers_ratio_threshold = 0.01

    # ======================================================================
    # ⭐ 바닥 윗부분의 가장 아랫부분을 조절하는 파라미터 (미터 단위) ⭐
    # 이 값을 양수로 늘리면 바닥 영역이 위로 확장되어,
    # 바닥 제외 (non_floor) 부분의 시작점이 더 높아집니다.
    # 즉, 바닥과 객체 간의 지저분한 연결 부분을 '잘라내는' 효과가 있습니다.
    # 0.01 (1cm), 0.02 (2cm), 0.05 (5cm), 0.1 (10cm) 등 작은 값부터 시작하여 테스트해보세요.
    # ======================================================================
    VERTICAL_OFFSET_ABOVE_FLOOR = 2  # 5 -> 바닥 싹사라짐

    load_and_ransac_multiple_planes(
        input_pcd_file,
        output_base_directory,
        max_planes=max_planes_to_find,
        distance_threshold=0.02, # RANSAC 평면 감지 정밀도
        ransac_n=3,
        num_iterations=1000,
        min_inliers_ratio=min_inliers_ratio_threshold,
        vertical_offset=VERTICAL_OFFSET_ABOVE_FLOOR # <-- 추가된 파라미터 전달
    )

    # --- 바닥 평면만 원본 PCD에서 별도 추출 ---
    input_pcd_file_original = "output/nonowall.pcd"  # nonowall 적용 안 된 원본
    output_base_directory_floor_orig = "output/ransac/floor_plane_from_original"
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