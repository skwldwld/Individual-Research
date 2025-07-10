import cv2
import numpy as np
import open3d as o3d

# --- 사용자 입력: outline.py에서 사용한 값 반드시 동일하게 입력! ---
scale_factor = 5  # outline.py에서 사용한 값
min_proj_x = 0.0  # outline.py 리턴값/로그 참고 (예시)
min_proj_z = 0.0  # outline.py 리턴값/로그 참고 (예시)
# morph 결과 이미지 경로
morph_img_path = "output/morph/morph_smoothed.png"
# 저장할 포인트클라우드 경로
output_pcd_path = "output/pcd/final_result.pcd"

# 1. morph 결과 이미지 로드
morph_img = cv2.imread(morph_img_path, cv2.IMREAD_GRAYSCALE)
if morph_img is None:
    print(f"이미지 로드 실패: {morph_img_path}")
    exit(1)
img_height, img_width = morph_img.shape

# 2. 흰색(255) 픽셀 좌표 추출
ys, xs = np.where(morph_img == 255)

# 3. 2D → 3D 변환 (Y는 임의값, 예: 0.0)
Y_value = 0.0
points_3d = []
for x, y in zip(xs, ys):
    X = x / scale_factor + min_proj_x
    Z = (img_height - 1 - y) / scale_factor + min_proj_z
    points_3d.append([X, Y_value, Z])
points_3d = np.array(points_3d)

# 4. 포인트클라우드로 저장 (색상 지정)
if len(points_3d) == 0:
    print("흰색 픽셀이 없습니다. 포인트클라우드를 생성할 수 없습니다.")
    exit(1)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)

# --- 모든 포인트에 파란색(RGB: 0, 0, 1) 지정 ---
color = np.array([[0, 0, 1]] * len(points_3d))  # Open3D는 0~1 범위
pcd.colors = o3d.utility.Vector3dVector(color)

o3d.io.write_point_cloud(output_pcd_path, pcd)
print(f"3D 포인트클라우드 저장 완료: {output_pcd_path}") 