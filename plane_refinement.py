import numpy as np
import open3d as o3d
import os

def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """PLY 파일에서 point cloud 불러오기."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {path}")
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise RuntimeError(f"Point cloud 로드 실패: {path}")
    return pcd

def refine_plane_model(pts: np.ndarray) -> np.ndarray:
    """
    주어진 평면 위 점들로부터 PCA를 이용해 법선 벡터를 재추정.
    반환값: [a, b, c, d] 형태의 평면 방정식 계수 (ax+by+cz+d=0)
    """
    centroid = pts.mean(axis=0)
    cov = np.cov((pts - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, np.argmin(eigvals)]
    # Ensure normal points roughly same 방향
    # if normal[2] < 0:
    #     normal = -normal
    d = -normal.dot(centroid)
    return np.hstack((normal, d))

def segment_and_refine_planes(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    min_inlier_ratio: float = 0.05
):
    """
    RANSAC으로 반복적으로 평면 분할 → 각 평면을 refine → 결과 리스트로 반환.
    - distance_threshold: RANSAC inlier 거리 임계값 (m 단위)
    - min_inlier_ratio: 전체 점 대비 최소 inlier 비율
    """
    planes = []
    models = []
    colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1]]  # 평면별 표시 색
    rest = pcd

    total_points = np.asarray(pcd.points).shape[0]
    idx_color = 0
    current_min_ratio = min_inlier_ratio

    while True:
        model, inliers = rest.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        # 더 작은 평면과 수직 평면도 인식하도록 적응적 임계값 조정
        if len(inliers) < current_min_ratio * total_points:
            # 남은 점이 충분히 있으면 더 관대한 조건으로 재시도
            remaining_points = np.asarray(rest.points).shape[0]
            if remaining_points > 50:  # 최소 50개 점이 남아있으면
                current_min_ratio *= 0.5  # 임계값을 절반으로 낮춤
                if current_min_ratio < 0.001:  # 너무 작아지면 중단
                    break
                continue
            else:
                break

        # 분리된 평면 점군
        plane_pcd = rest.select_by_index(inliers)
        # 평면 모델 재추정
        pts = np.asarray(plane_pcd.points)
        refined = refine_plane_model(pts)

        # 컬러 적용
        plane_pcd.paint_uniform_color(colors[idx_color % len(colors)])
        planes.append(plane_pcd)
        models.append(refined)

        # 나머지 점군 갱신
        rest = rest.select_by_index(inliers, invert=True)
        idx_color += 1
        
        # 임계값을 원래대로 복원 (다음 평면 검출을 위해)
        current_min_ratio = min_inlier_ratio

    # 나머지 점은 회색으로
    rest.paint_uniform_color([0.5, 0.5, 0.5])
    return planes, models, rest

def detect_vertical_planes_specialized(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.01,
    min_inlier_ratio: float = 0.002
):
    """
    수직 평면 전용 검출 함수.
    수직 평면은 보통 점의 밀도가 낮고 분산이 클 수 있어서 더 관대한 파라미터 사용.
    """
    planes = []
    models = []
    rest = pcd
    colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1]]
    idx_color = 0
    
    total_points = np.asarray(pcd.points).shape[0]
    
    while True:
        model, inliers = rest.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=5000  # 수직 평면 검출을 위해 더 많은 반복
        )
        
        if len(inliers) < min_inlier_ratio * total_points:
            break
            
        # 검출된 평면이 실제로 수직인지 확인
        a, b, c, d = model
        normal = np.array([a, b, c])
        normal_unit = normal / np.linalg.norm(normal)
        z_component = abs(normal_unit[2])
        
        # z축과 45도 이상 각도를 이루는 평면만 수직 평면으로 간주
        if z_component < 0.7:
            plane_pcd = rest.select_by_index(inliers)
            pts = np.asarray(plane_pcd.points)
            refined = refine_plane_model(pts)
            
            plane_pcd.paint_uniform_color(colors[idx_color % len(colors)])
            planes.append(plane_pcd)
            models.append(refined)
            idx_color += 1
            
        # 수직 평면이 아니어도 제거 (다음 검출을 위해)
        rest = rest.select_by_index(inliers, invert=True)
    
    rest.paint_uniform_color([0.5, 0.5, 0.5])
    return planes, models, rest

def visualize_results(geometries):
    """분리된 평면 시각화 (주석처리: draw_geometries 사용 불가시)."""
    # o3d.visualization.draw_geometries(geometries)
    pass

if __name__ == "__main__":
    # 1) PLY 파일 경로
    ply_path = "input/3BP_ascii.pcd"

    # 2) point cloud 로드
    pcd = load_point_cloud(ply_path)
    print(f"로드된 점 개수: {len(pcd.points)}")

    # 3) 평면 분할 및 재추정 - 더 작은 평면과 수직 평면도 인식하도록 파라미터 조정
    planes, models, rest = segment_and_refine_planes(
        pcd,
        distance_threshold=0.008,  # 수직 평면을 위해 더 관대한 임계값
        ransac_n=3,
        num_iterations=3000,       # 더 많은 반복으로 정확도 향상
        min_inlier_ratio=0.003     # 매우 작은 평면과 수직 평면도 인식하도록 낮춤
    )

    # 4) 각 평면 모델 방정식 출력
    print(f"\n총 {len(models)}개의 평면이 검출되었습니다:")
    for i, m in enumerate(models):
        a, b, c, d = m
        normal = np.array([a, b, c])
        normal_norm = np.linalg.norm(normal)
        normal_unit = normal / normal_norm
        
        # 평면의 방향 분석 (수평/수직 판단)
        z_component = abs(normal_unit[2])
        if z_component > 0.7:  # z축과 45도 이내
            plane_type = "수평 평면"
        elif z_component < 0.3:  # z축과 45도 이상
            plane_type = "수직 평면"
        else:
            plane_type = "경사 평면"
            
        print(f"Plane {i} ({plane_type}): {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
        print(f"  법선 벡터: [{normal_unit[0]:.3f}, {normal_unit[1]:.3f}, {normal_unit[2]:.3f}]")
        print(f"  점 개수: {len(planes[i].points)}")

    # 5) 각 평면의 점을 projection하여 평평하게 만듦
    all_projected_points = []
    all_projected_colors = []
    
    for plane_pcd, model in zip(planes, models):
        pts = np.asarray(plane_pcd.points)
        colors = np.asarray(plane_pcd.colors)  # 원본 색상 가져오기
        a, b, c, d = model
        normal = np.array([a, b, c])
        
        # 법선 벡터 정규화
        norm_val = np.linalg.norm(normal)
        if norm_val == 0:  # 0으로 나누기 방지
            continue
        normal_normalized = normal / norm_val
        d_normalized = d / norm_val
        
        # 각 점을 평면에 projection
        projected_pts = pts - ((pts @ normal_normalized + d_normalized)[:, None]) * normal_normalized
        all_projected_points.append(projected_pts)
        all_projected_colors.append(colors)

    # 6) 검출된 평면들의 점만 합치기 (전체 점이 아닌)
    if all_projected_points:
        combined_projected_points = np.vstack(all_projected_points)
        combined_projected_colors = np.vstack(all_projected_colors)
        
        result_pcd = o3d.geometry.PointCloud()
        result_pcd.points = o3d.utility.Vector3dVector(combined_projected_points)
        result_pcd.colors = o3d.utility.Vector3dVector(combined_projected_colors)
        
        # 7) projection된 결과 시각화 (주석처리: draw_geometries 사용 불가시)
        # visualize_results([result_pcd])

        # 8) 검출된 평면들만 저장 (전체 점이 아닌)
        output_path = "output/output.pcd"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        o3d.io.write_point_cloud(output_path, result_pcd)
        print(f"\n검출된 평면들만 저장됨: {output_path}")
        print(f"평면화된 점 개수: {len(result_pcd.points)}")
    else:
        print("검출된 평면이 없습니다.")

    
    # 나머지 점들도 저장 (평면이 아닌 점들)
    if not rest.is_empty():
        o3d.io.write_point_cloud("output/remaining_points.pcd", rest)
        print(f"평면이 아닌 나머지 점들: {len(rest.points)}개")

    print("\n파라미터 조정 팁:")
    print("- distance_threshold를 낮추면 더 정밀한 평면 검출")
    print("- min_inlier_ratio를 낮추면 더 작은 평면도 검출")
    print("- num_iterations를 늘리면 더 정확한 검출")