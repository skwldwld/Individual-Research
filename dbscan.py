import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt # 클러스터 시각화를 위한 색상 맵
from sklearn.cluster import DBSCAN

# Open3D 내장 DBSCAN 사용 (최적화된 구현)

def cluster_objects_direct_dbscan(
    pcd_path,
    output_base_directory,
    # DBSCAN 클러스터링 파라미터
    dbscan_eps=0.05,        # 이웃을 찾을 반경 (매우 중요, 물체 크기에 맞게 조정)
    dbscan_min_points=30,   # 클러스터 형성 최소 포인트 수 (매우 중요, 물체 밀도에 맞게 조정)
    min_cluster_points_ratio=0.001, # 전체 포인트 대비 최소 클러스터 비율
    min_cluster_absolute_points=100 # 클러스터의 최소 절대 포인트 수
):
    """
    포인트 클라우드를 불러와 scikit-learn의 DBSCAN을 사용하여 작은 물체들을 클러스터링하고 저장합니다.
    바닥 평면 제거 단계는 포함되지 않습니다.

    Args:
        pcd_path (str): 입력 포인트 클라우드 파일 경로 (.pcd, .ply 등).
        output_base_directory (str): 모든 결과물(클러스터)이 저장될 기본 디렉토리.
        dbscan_eps (float): DBSCAN의 'epsilon' 파라미터 (이웃을 찾을 반경).
        dbscan_min_points (int): DBSCAN의 'min_points' 파라미터 (클러스터를 형성할 최소 포인트 수).
        min_cluster_points_ratio (float): 전체 입력 포인트 수 대비 클러스터의 최소 포인트 비율.
        min_cluster_absolute_points (int): 클러스터의 최소 절대 포인트 수.
    """
    print(f"\n--- 포인트 클라우드 로드: {pcd_path} ---")
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
    except Exception as e:
        print(f"포인트 클라우드 로드 중 오류 발생: {e}")
        print("파일 경로가 올바르고 유효한 .pcd 또는 .ply 파일인지 확인하세요.")
        return

    if not pcd.has_points():
        print("로드된 포인트 클라우드에 포인트가 없습니다. 종료합니다.")
        return

    print(f"로드된 포인트 클라우드의 포인트 수: {len(pcd.points)}개.")
    original_total_points = len(pcd.points)

    # 포인트 클라우드 범위 분석
    print("\n--- 포인트 클라우드 범위 분석 ---")
    points = np.asarray(pcd.points)
    
    # 포인트 클라우드 범위 분석
    print("포인트 클라우드 범위 분석 중...")
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    range_coords = max_coords - min_coords
    
    print(f"  포인트 클라우드 범위:")
    print(f"    X: {min_coords[0]:.3f} ~ {max_coords[0]:.3f} (범위: {range_coords[0]:.3f})")
    print(f"    Y: {min_coords[1]:.3f} ~ {max_coords[1]:.3f} (범위: {range_coords[1]:.3f})")
    print(f"    Z: {min_coords[2]:.3f} ~ {max_coords[2]:.3f} (범위: {range_coords[2]:.3f})")
    
    # 전체 범위의 1%를 추천 eps로 사용
    recommended_eps = np.mean(range_coords) * 0.01
    print(f"  추천 eps 값 (전체 범위의 1%): {recommended_eps:.4f}")
    
    # 현재 eps와 비교
    if dbscan_eps < recommended_eps * 0.5:
        print(f"  ⚠️  현재 eps ({dbscan_eps})가 너무 작습니다. {recommended_eps:.4f} 정도로 늘려보세요.")
    elif dbscan_eps > recommended_eps * 2:
        print(f"  ⚠️  현재 eps ({dbscan_eps})가 너무 큽니다. {recommended_eps:.4f} 정도로 줄여보세요.")
    else:
        print(f"  ✅ 현재 eps ({dbscan_eps})가 적절한 범위에 있습니다.")

    # 출력 디렉토리 생성
    cluster_output_dir = output_base_directory
    os.makedirs(cluster_output_dir, exist_ok=True)

    # --- 1. scikit-learn DBSCAN 클러스터링 시작 ---
    print("\n--- 1. scikit-learn DBSCAN 클러스터링 시작 ---")
    print(f"  DBSCAN Epsilon (eps): {dbscan_eps}")
    print(f"  DBSCAN 최소 포인트 (min_points): {dbscan_min_points}")
    print(f"  최소 클러스터 포인트 비율: {min_cluster_points_ratio} (원본 대비)")
    print(f"  클러스터의 최소 절대 포인트 수: {min_cluster_absolute_points}개")
    print(f"  DBSCAN 적용 대상 포인트 수: {len(pcd.points)}개 (전체 포인트)")

    # scikit-learn DBSCAN 클러스터링 수행 (훨씬 빠름)
    print("scikit-learn DBSCAN 클러스터링 진행 중...")
    
    # scikit-learn DBSCAN 사용
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_points, n_jobs=-1)
    labels = dbscan.fit_predict(points)

    max_label = labels.max()
    print(f"scikit-learn DBSCAN이 {max_label + 1}개의 클러스터를 찾았습니다 (노이즈 포함).")

    # 클러스터별 색상 지정
    # tab20은 20가지 구별되는 색상을 제공합니다.
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels == -1] = [0.3, 0.3, 0.3, 0.5] # 노이즈 포인트: 어두운 회색
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3]) # 색상 적용 (알파 채널 무시)

    cluster_clouds_to_visualize = []
    found_object_count = 0

    for i in range(max_label + 1): # -1은 노이즈이므로 0부터 max_label까지 반복
        cluster_indices = np.where(labels == i)[0]
        cluster_size = len(cluster_indices)

        # 클러스터 유효성 검사: 최소 포인트 비율 또는 최소 절대 포인트 수
        if cluster_size > (original_total_points * min_cluster_points_ratio) and \
           cluster_size >= min_cluster_absolute_points:
            
            cluster_cloud = pcd.select_by_index(cluster_indices)
            cluster_clouds_to_visualize.append(cluster_cloud) # 시각화를 위해 클러스터 추가
            found_object_count += 1

            cluster_output_path = os.path.join(cluster_output_dir, f"object_cluster_{found_object_count:03d}.pcd")
            try:
                o3d.io.write_point_cloud(cluster_output_path, cluster_cloud)
                print(f"  클러스터 {found_object_count} ({i+1}번째 원래 클러스터) 저장됨: {cluster_output_path} ({cluster_size} 포인트)")
                
                # 개별 클러스터 시각화
                print(f"    클러스터 {found_object_count} 시각화 중...")
                cluster_cloud.paint_uniform_color([1, 0, 0])  # 빨간색으로 표시
                try:
                    o3d.visualization.draw_geometries(
                        [cluster_cloud],
                        window_name=f"클러스터 {found_object_count} ({cluster_size} 포인트)",
                        width=800,
                        height=600
                    )
                except Exception as viz_error:
                    print(f"    시각화 오류: {viz_error}")
                    print(f"    클러스터 {found_object_count}는 저장되었지만 시각화할 수 없습니다.")
                
            except Exception as e:
                print(f"  클러스터 {found_object_count} 저장 중 오류 발생: {e}")
        else:
            print(f"  클러스터 {i+1} ({cluster_size} 포인트)가 너무 작아서 저장하지 않음.")

    # 노이즈 포인트 (시각화 목적)
    noise_indices = np.where(labels == -1)[0]
    if len(noise_indices) > 0:
        noise_cloud = pcd.select_by_index(noise_indices)
        noise_cloud.paint_uniform_color([0.1, 0.1, 0.1]) # 노이즈는 아주 어두운 회색
        cluster_clouds_to_visualize.append(noise_cloud)
        print(f"  {len(noise_indices)}개의 노이즈 포인트를 찾았습니다 (별도의 객체로 저장되지 않음).")

    # 최종 결과 시각화
    if len(cluster_clouds_to_visualize) > 0:
        print("\n--- 최종 클러스터된 물체들 시각화 ---")
    else:
        print("시각화할 유의미한 클러스터가 없습니다.")

    print(f"\n--- 총 {found_object_count}개의 유의미한 객체 클러스터가 발견 및 저장되었습니다. ---")


if __name__ == "__main__":
    # --- 설정 파라미터 ---
    input_pcd_file = "output/ransac/above_floor.pcd" # 입력 포인트 클라우드 파일 경로
    output_base_directory = "output/dbscan" # 모든 결과물 저장될 기본 폴더

    # DBSCAN 클러스터링 (객체용) 파라미터 - 이 값들을 조정하여 최적화하세요!
    # 물체의 크기와 포인트 밀도를 고려하여 `dbscan_eps`와 `dbscan_min_points`를 신중하게 조정하세요.
    # 작은 물체(예: 의자, 테이블 다리)는 `eps`와 `min_points`를 작게, 큰 물체(예: 벽면 전체)는 크게 설정합니다.

    # DBSCAN_EPS (Epsilon): 이웃을 찾을 반경.
    # 값이 작을수록 더 작은/조밀한 클러스터 생성.
    # 너무 작으면 하나의 물체가 여러 클러스터로 쪼개지고, 너무 크면 여러 물체가 하나로 묶임.
    DBSCAN_EPS = 0.8 # 0.8 짱짱맨

    # DBSCAN_MIN_POINTS (Minimum Points): 클러스터를 형성할 최소 포인트 수.
    # 값이 작을수록 노이즈가 클러스터로 잡힐 수 있고, 값이 클수록 작은 물체가 노이즈로 분류됨.
    DBSCAN_MIN_POINTS = 10  # 10 짱짱맨

    # 클러스터 필터링 (너무 작은 클러스터 제외)
    # MIN_CLUSTER_POINTS_RATIO: 원본 포인트 수 대비 최소 비율
    # 0.001 (0.1%) 미만의 포인트는 무시 (더 작은 물체를 위해 낮게 설정)
    MIN_CLUSTER_POINTS_RATIO = 0.0005  # 0.002에서 0.0005로 감소 (더 관대하게)

    # MIN_CLUSTER_ABSOLUTE_POINTS: 클러스터에 포함되어야 할 최소 절대 포인트 수
    # 최소 50개 포인트 미만은 무시 (노이즈 필터링에 유용)
    MIN_CLUSTER_ABSOLUTE_POINTS = 600  # 450 이상 짱짱맨

    # 함수 호출
    cluster_objects_direct_dbscan(
        input_pcd_file,
        output_base_directory,
        dbscan_eps=DBSCAN_EPS,
        dbscan_min_points=DBSCAN_MIN_POINTS,
        min_cluster_points_ratio=MIN_CLUSTER_POINTS_RATIO,
        min_cluster_absolute_points=MIN_CLUSTER_ABSOLUTE_POINTS
    )