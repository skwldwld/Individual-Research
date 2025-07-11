import open3d as o3d
import numpy as np
import cv2
import os
from sklearn.cluster import DBSCAN
import pickle

def extract_2d_outline_from_pcd_y_up(
    pcd_path, output_dir, scale_factor=0, min_contour_area=10, 
    dilate_kernel_size=1, dilate_iterations=0, contour_thickness=3, 
    contour_color=(0, 255, 255),
    dbscan_eps=0.03, dbscan_min_samples=30, dot_size=3
):
    """
    3D 포인트 클라우드에서 객체들을 2D 투영하여 윤곽선을 추출하고 이미지로 저장합니다.
    DBSCAN으로 노이즈(작은 클러스터) 제거 및 객체 분리.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"출력 디렉토리 생성됨: {output_dir}")

    print(f"\n--- 2D 윤곽선 추출 시작 (Y축 높이 가정, DBSCAN 분리) ---")
    print(f"대상 파일: {pcd_path}")

    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
    except Exception as e:
        print(f"🔴 오류: 포인트 클라우드 불러오기 실패: {e}")
        print("경로가 정확하고 파일이 손상되지 않았는지 확인하세요.")
        return

    if not pcd.has_points():
        print("🔴 오류: 불러온 포인트 클라우드에 포인트가 없습니다. 윤곽선을 추출할 수 없습니다.")
        return

    points = np.asarray(pcd.points)
    print(f"🟢 로딩된 총 포인트 수: {len(points)}")

    # 1. X-Z 평면으로 투영 (Y축이 높이이므로 XZ 평면이 탑뷰가 됨)
    projected_points = points[:, [0, 2]] # X, Z 좌표 사용

    # DBSCAN으로 클러스터 분리 (노이즈 제거)
    print(f"DBSCAN 클러스터링: eps={dbscan_eps}, min_samples={dbscan_min_samples}")
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels = dbscan.fit_predict(projected_points)
    unique_labels = set(labels)
    if -1 in unique_labels:
        print(f"  노이즈(라벨 -1) 포인트: {(labels==-1).sum()}개")
    print(f"  검출된 클러스터 수(노이즈 제외): {len(unique_labels) - (1 if -1 in unique_labels else 0)}")

    # 이미지 크기 계산
    min_proj_x, min_proj_z = np.min(projected_points, axis=0)
    max_proj_x, max_proj_z = np.max(projected_points, axis=0)
    range_proj_x = max_proj_x - min_proj_x
    range_proj_z = max_proj_z - min_proj_z
    img_width = int(range_proj_x * scale_factor) + 1
    img_height = int(range_proj_z * scale_factor) + 1
    min_img_dim = 10 
    if img_width < min_img_dim or img_height < min_img_dim:
        img_width = max(img_width, min_img_dim)
        img_height = max(img_height, min_img_dim)
    print(f"🟢 생성될 최종 이미지 크기: {img_width} x {img_height} 픽셀 (너비 x 높이)")

    # 전체 이진 이미지(시각화용)
    binary_image = np.zeros((img_height, img_width), dtype=np.uint8)
    contour_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    all_clean_contours = []
    cluster_count = 0
    # --- 추가: 2D 픽셀 <-> 3D 포인트 매핑 테이블 ---
    pixel_to_points = dict()
    #
    for label in unique_labels:
        if label == -1:
            continue  # 노이즈 클러스터는 무시
        cluster_mask = (labels == label)
        cluster_points = projected_points[cluster_mask]
        orig_points = points[cluster_mask]  # 원본 3D 포인트
        if len(cluster_points) < min_contour_area:
            continue  # 너무 작은 클러스터는 무시
        # 픽셀 좌표 변환
        pixel_points_x = np.round((cluster_points[:, 0] - min_proj_x) * scale_factor).astype(np.int32)
        pixel_points_y = np.round((cluster_points[:, 1] - min_proj_z) * scale_factor).astype(np.int32)
        pixel_points_y = img_height - 1 - pixel_points_y
        # --- 매핑 테이블 저장 ---
        for idx, (px, py) in enumerate(zip(pixel_points_x, pixel_points_y)):
            key = (int(px), int(py))
            if key not in pixel_to_points:
                pixel_to_points[key] = []
            pixel_to_points[key].append(orig_points[idx])
        # 클러스터별 이진 이미지
        cluster_img = np.zeros((img_height, img_width), dtype=np.uint8)
        for px, py in np.c_[pixel_points_x, pixel_points_y]:
            if 0 <= py < img_height and 0 <= px < img_width:
                cv2.rectangle(cluster_img, (px-dot_size//2, py-dot_size//2), (px+dot_size//2, py+dot_size//2), 255, -1)
        # 팽창 등 최소화 (붙지 않게)
        if dilate_iterations > 0:
            dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
            cluster_img = cv2.dilate(cluster_img, dilate_kernel, iterations=dilate_iterations)
        # 컨투어 추출
        contours, _ = cv2.findContours(cluster_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 면적 필터링 및 근사화
        clean_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area > min_contour_area:
                epsilon = 0.002 * perimeter
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                clean_contours.append(approx_contour)
        if clean_contours:
            all_clean_contours.extend(clean_contours)
            # 시각화용
            cv2.drawContours(contour_image, clean_contours, -1, contour_color, contour_thickness)
            cv2.drawContours(binary_image, clean_contours, -1, (255,255,255), -1)
            cluster_count += 1
    print(f"🟢 최종 추출된 클러스터 컨투어 개수: {len(all_clean_contours)} (클러스터 수: {cluster_count})")
    # --- 매핑 테이블 pickle로 저장 ---
    pixel_map_path = os.path.join(output_dir, "pixel_to_points.pkl")
    with open(pixel_map_path, "wb") as f:
        pickle.dump(pixel_to_points, f)
    print(f"✅ 2D 픽셀-3D포인트 매핑 저장됨: {pixel_map_path}")
    # 결과 이미지 저장
    output_image_path_binary = os.path.join(output_dir, "binary.png")
    output_image_path_contours = os.path.join(output_dir, "contours.png")
    cv2.imwrite(output_image_path_binary, binary_image)
    cv2.imwrite(output_image_path_contours, contour_image)
    print(f"✅ 이진 투영 이미지 저장됨: {output_image_path_binary}")
    print(f"✅ 추출된 윤곽선 이미지 저장됨: {output_image_path_contours}")
    if all_clean_contours is None:
        all_clean_contours = []
    return all_clean_contours, (img_width, img_height), (min_proj_x, min_proj_z), scale_factor

if __name__ == "__main__":
    input_pcd_for_outline = "output/ransac/above_floor.pcd" 
    output_outline_dir = "output/outline" 
    extracted_contours, img_dims, min_coords, scale = extract_2d_outline_from_pcd_y_up(
        input_pcd_for_outline,
        output_outline_dir,
        scale_factor=5,       # 이미지가 더 조밀해지도록 대폭 낮춤
        min_contour_area=60,   # 너무 작은 노이즈 클러스터 무시
        dilate_kernel_size=1,  # 팽창 최소화
        dilate_iterations=0,   # 팽창 없음
        contour_thickness=1,   # 윤곽선 선 두께
        contour_color=(0, 0, 255), # 윤곽선 색상 (BGR: 선명한 빨간색)
        dbscan_eps=3,
        dbscan_min_samples=20,
        dot_size=3
    )
    if extracted_contours is not None and len(extracted_contours) > 0:
        print(f"\n총 {len(extracted_contours)}개의 윤곽선이 성공적으로 추출되었습니다.")
    else:
        print("\n윤곽선을 추출하지 못했습니다. 위의 로그와 생성된 중간 이미지들을 자세히 확인하여 문제 원인을 파악하세요.")