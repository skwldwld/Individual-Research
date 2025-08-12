import open3d as o3d
import numpy as np
import cv2
import os
from sklearn.cluster import DBSCAN
import pickle

def extract_2d_outline_from_pcd_projection(
    pcd_path, output_dir,
    project_dims=(0, 2),          # (x, z) -> top view / (x, 1) -> x-y side view / (2, 1) -> z-y front view
    scale_factor=0,
    min_contour_area=10,
    dilate_kernel_size=1, dilate_iterations=0, contour_thickness=3,
    contour_color=(0, 255, 255),
    dbscan_eps=0.03, dbscan_min_samples=30, dot_size=3,
    plane_name="topview"
):
    """
    3D 포인트 클라우드를 임의의 평면으로 투영하여 2D 윤곽선을 추출
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"출력 디렉토리 생성됨: {output_dir}")

    print(f"\n--- 2D 윤곽선 추출 시작 ({plane_name}, DBSCAN 분리) ---")
    print(f"대상 파일: {pcd_path}")

    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
    except Exception as e:
        print(f"[ERROR] 포인트 클라우드 불러오기 실패: {e}")
        print("경로가 정확하고 파일이 손상되지 않았는지 확인하세요.")
        return

    max_points = 100000
    if len(pcd.points) > max_points:
        print(f"다운샘플링 전 포인트 수: {len(pcd.points)}")
        voxel = 0.03
        while True:
            temp_pcd = pcd.voxel_down_sample(voxel_size=voxel)
            if len(temp_pcd.points) <= max_points or voxel > 1.0:
                break
            voxel *= 1.5
        pcd = temp_pcd
        print(f"다운샘플링 적용됨! voxel_size={voxel:.4f}, 다운샘플링 후 포인트 수: {len(pcd.points)}")
    else:
        print(f"다운샘플링 불필요: 포인트 수 {len(pcd.points)}")

    if not pcd.has_points():
        print("[ERROR] 불러온 포인트 클라우드에 포인트가 없습니다. 윤곽선을 추출할 수 없습니다.")
        return

    points = np.asarray(pcd.points)
    print(f"[INFO] 로딩된 총 포인트 수: {len(points)}")

    projected_points = points[:, project_dims]  # 원하는 평면으로 투영

    print(f"DBSCAN 클러스터링: eps={dbscan_eps}, min_samples={dbscan_min_samples}")
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels = dbscan.fit_predict(projected_points)
    unique_labels = set(labels)
    if -1 in unique_labels:
        print(f"  노이즈(라벨 -1) 포인트: {(labels==-1).sum()}개")
    print(f"  검출된 클러스터 수(노이즈 제외): {len(unique_labels) - (1 if -1 in unique_labels else 0)}")

    min_proj_0, min_proj_1 = np.min(projected_points, axis=0)
    max_proj_0, max_proj_1 = np.max(projected_points, axis=0)
    range_proj_0 = max_proj_0 - min_proj_0
    range_proj_1 = max_proj_1 - min_proj_1
    img_width = int(range_proj_0 * scale_factor) + 1
    img_height = int(range_proj_1 * scale_factor) + 1
    min_img_dim = 10
    if img_width < min_img_dim or img_height < min_img_dim:
        img_width = max(img_width, min_img_dim)
        img_height = max(img_height, min_img_dim)
    print(f"[INFO] 생성될 최종 이미지 크기: {img_width} x {img_height} 픽셀 (너비 x 높이)")

    binary_image = np.zeros((img_height, img_width), dtype=np.uint8)
    contour_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    all_clean_contours = []
    cluster_count = 0
    pixel_to_points = dict()
    for label in unique_labels:
        if label == -1:
            continue
        cluster_mask = (labels == label)
        cluster_points = projected_points[cluster_mask]
        orig_points = points[cluster_mask]
        print(f"[INFO] 클러스터 라벨 {label}: 포인트 개수 = {len(cluster_points)}")
        if len(cluster_points) < min_contour_area:
            continue
        pixel_points_x = np.round((cluster_points[:, 0] - min_proj_0) * scale_factor).astype(np.int32)
        pixel_points_y = np.round((cluster_points[:, 1] - min_proj_1) * scale_factor).astype(np.int32)
        pixel_points_y = img_height - 1 - pixel_points_y
        for idx, (px, py) in enumerate(zip(pixel_points_x, pixel_points_y)):
            key = (int(px), int(py))
            if key not in pixel_to_points:
                pixel_to_points[key] = []
            pixel_to_points[key].append(orig_points[idx])
        cluster_img = np.zeros((img_height, img_width), dtype=np.uint8)
        for px, py in np.c_[pixel_points_x, pixel_points_y]:
            if 0 <= py < img_height and 0 <= px < img_width:
                cv2.rectangle(cluster_img, (px-dot_size//2, py-dot_size//2), (px+dot_size//2, py+dot_size//2), 255, -1)
        if dilate_iterations > 0:
            dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
            cluster_img = cv2.dilate(cluster_img, dilate_kernel, iterations=dilate_iterations)
            erode_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
            cluster_img = cv2.erode(cluster_img, erode_kernel, iterations=1)
        contours, _ = cv2.findContours(cluster_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            cv2.drawContours(contour_image, clean_contours, -1, contour_color, contour_thickness)
            cv2.drawContours(binary_image, clean_contours, -1, (255,255,255), -1)
            cluster_count += 1
    print(f"[INFO] 최종 추출된 클러스터 컨투어 개수: {len(all_clean_contours)} (클러스터 수: {cluster_count})")
    pixel_map_path = os.path.join(output_dir, "pixel_to_points.pkl")
    with open(pixel_map_path, "wb") as f:
        pickle.dump(pixel_to_points, f)
    print(f"[SUCCESS] 2D 픽셀-3D포인트 매핑 저장됨: {pixel_map_path}")
    output_image_path_binary = os.path.join(output_dir, "binary.png")
    output_image_path_contours = os.path.join(output_dir, "contours.png")
    cv2.imwrite(output_image_path_binary, binary_image)
    cv2.imwrite(output_image_path_contours, contour_image)
    print(f"[SUCCESS] 이진 투영 이미지 저장됨: {output_image_path_binary}")
    print(f"[SUCCESS] 추출된 윤곽선 이미지 저장됨: {output_image_path_contours}")

    _vis_image = contour_image.copy()
    for contour in all_clean_contours:
        for point in contour:
            x, y = point[0]
            cv2.circle(_vis_image, (x, y), dot_size, (0, 0, 255), -1)
    output_image_path_corners = os.path.join(output_dir, "corners_vis.png")
    cv2.imwrite(output_image_path_corners, _vis_image)
    print(f"[SUCCESS] 꼭짓점 시각화 이미지 저장됨: {output_image_path_corners}")
    if all_clean_contours is None:
        all_clean_contours = []
    return all_clean_contours, (img_width, img_height), (min_proj_0, min_proj_1), scale_factor

def main():
    # 바닥 outline (x-z 평면)
    extract_2d_outline_from_pcd_projection(
        pcd_path="../output/ransac/floor_plane.pcd",
        output_dir="../output/outline/floor",
        project_dims=(0, 2),
        scale_factor=3,
        min_contour_area=20,
        dilate_kernel_size=1,
        dilate_iterations=0,
        contour_thickness=1,
        contour_color=(0, 255, 0),
        dbscan_eps=2.0,
        dbscan_min_samples=8,
        dot_size=2,
        plane_name="floor_topview"
    )
    # 위에서 본 outline (x-z 평면, Y축 높이)
    extract_2d_outline_from_pcd_projection(
        pcd_path="../output/ransac/topview.pcd",
        output_dir="../output/outline/topview/material",
        project_dims=(0, 2),    # x, z 평면
        scale_factor=5,
        min_contour_area=400,
        dilate_kernel_size=1,
        dilate_iterations=0,
        contour_thickness=1,
        contour_color=(0, 0, 255),
        dbscan_eps=1.5,
        dbscan_min_samples=60,
        dot_size=2,
        plane_name="topview"
    )
    # 옆에서 본 outline (x-y 평면) ← 여기 추가
    extract_2d_outline_from_pcd_projection(
        pcd_path="../output/ransac/topview.pcd",
        output_dir="../output/outline/sideview/material",
        project_dims=(0, 1),   # x, y 평면 (옆에서 본다)
        scale_factor=5,
        min_contour_area=400,
        dilate_kernel_size=1,
        dilate_iterations=0,
        contour_thickness=1,
        contour_color=(255, 0, 255),   # 보라색
        dbscan_eps=1.5,
        dbscan_min_samples=30,
        dot_size=2,
        plane_name="sideview"
    )
    # 벽 top_view
    extract_2d_outline_from_pcd_projection(
        pcd_path="../output/removed_walls.pcd",
        output_dir="../output/outline/topview/wall",
        project_dims=(0, 2),   # x-z 평면 (top view)
        scale_factor=5,
        min_contour_area=400,
        dilate_kernel_size=1,
        dilate_iterations=0,
        contour_thickness=1,
        contour_color=(0, 0, 255),
        dbscan_eps=1.5,
        dbscan_min_samples=60,
        dot_size=2,
        plane_name="walls_topview"
    )
    # 벽 side_view
    extract_2d_outline_from_pcd_projection(
        pcd_path="../output/removed_walls.pcd",
        output_dir="../output/outline/sideview/wall",
        project_dims=(0, 1),     # x, y (Y축이 높이)
        scale_factor=6,
        min_contour_area=200,
        dilate_kernel_size=1, dilate_iterations=0,
        contour_thickness=2, contour_color=(255, 0, 255),
        dbscan_eps=1.2, dbscan_min_samples=40,  # sideview는 보통 eps 조금 더 타이트
        dot_size=2,
        plane_name="walls_sideview"
    )

    

if __name__ == "__main__":
    main()
