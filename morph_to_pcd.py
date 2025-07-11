import cv2
import numpy as np
import open3d as o3d
import pickle
import os

# --- 사용자 입력: outline.py에서 사용한 값 반드시 동일하게 입력! ---
scale_factor = 5  # outline.py에서 사용한 값
min_proj_x = 0.0  # outline.py 리턴값/로그 참고 (예시)
min_proj_z = 0.0  # outline.py 리턴값/로그 참고 (예시)

# 파일 경로
morph_img_path = "output/morph/morph_smoothed.png"
pixel_map_path = "output/outline/pixel_to_points.pkl"  # 매핑 테이블 경로
output_pcd_path = "output/pcd/final_result.pcd"

# 출력 디렉토리 생성
output_dir = os.path.dirname(output_pcd_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. morph 결과 이미지 로드
morph_img = cv2.imread(morph_img_path, cv2.IMREAD_GRAYSCALE)
if morph_img is None:
    print(f"이미지 로드 실패: {morph_img_path}")
    exit(1)
img_height, img_width = morph_img.shape
print(f"이미지 크기: {img_width} x {img_height}")

# 2. 매핑 테이블 로드
if os.path.exists(pixel_map_path):
    print(f"매핑 테이블 로드 중: {pixel_map_path}")
    with open(pixel_map_path, "rb") as f:
        pixel_to_points = pickle.load(f)
    print(f"매핑 테이블 로드 완료: {len(pixel_to_points)} 개의 픽셀-포인트 매핑")
else:
    print(f"⚠️ 매핑 테이블을 찾을 수 없습니다: {pixel_map_path}")
    print("outline.py를 먼저 실행하여 매핑 테이블을 생성하세요.")
    exit(1)

# 3. 흰색(255) 픽셀 좌표 추출
ys, xs = np.where(morph_img == 255)
print(f"흰색 픽셀 수: {len(xs)}")

# 4. 2D → 3D 변환 (매핑된 픽셀만 사용)
points_3d = []
mapped_count = 0
unmapped_count = 0

for x, y in zip(xs, ys):
    pixel_key = (int(x), int(y))
    
    if pixel_key in pixel_to_points:
        # 매핑 테이블에서 해당 픽셀의 모든 3D 포인트 가져오기
        original_points = pixel_to_points[pixel_key]
        points_3d.extend(original_points)
        mapped_count += 1
    else:
        # 매핑되지 않은 픽셀은 무시
        unmapped_count += 1

points_3d = np.array(points_3d)

# 중복 포인트 제거 (동일한 좌표를 가진 포인트들)
print(f"중복 제거 전 포인트 수: {len(points_3d)}")

# 좌표를 문자열로 변환하여 중복 제거
unique_coords = set()
unique_points = []
for point in points_3d:
    coord_str = f"{point[0]:.6f},{point[1]:.6f},{point[2]:.6f}"
    if coord_str not in unique_coords:
        unique_coords.add(coord_str)
        unique_points.append(point)

points_3d = np.array(unique_points)
print(f"중복 제거 후 포인트 수: {len(points_3d)}")
print(f"제거된 중복 포인트 수: {len(points_3d) - len(unique_points)}")

print(f"매핑된 픽셀 수: {mapped_count}")
print(f"매핑되지 않은 픽셀 수: {unmapped_count}")
print(f"총 3D 포인트 수: {len(points_3d)}")

# 5. 포인트클라우드로 저장
if len(points_3d) == 0:
    print("3D 포인트가 없습니다. 포인트클라우드를 생성할 수 없습니다.")
    exit(1)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)

# 포인트 분포 확인
points_np = np.asarray(pcd.points)
print(f"3D 포인트 분포:")
print(f"  X: [{np.min(points_np[:, 0]):.3f}, {np.max(points_np[:, 0]):.3f}]")
print(f"  Y: [{np.min(points_np[:, 1]):.3f}, {np.max(points_np[:, 1]):.3f}]")
print(f"  Z: [{np.min(points_np[:, 2]):.3f}, {np.max(points_np[:, 2]):.3f}]")

# 색상 지정 (모든 포인트를 파란색으로)
colors = np.array([[0, 0, 1]] * len(points_3d))  # 모든 포인트를 파란색으로
pcd.colors = o3d.utility.Vector3dVector(colors)

# 포인트클라우드 저장
o3d.io.write_point_cloud(output_pcd_path, pcd)
print(f"✅ 3D 포인트클라우드 저장 완료: {output_pcd_path}")

# 시각화
print("\n포인트클라우드 시각화 중...")
try:
    o3d.visualization.draw_geometries([pcd], window_name="Morph to PCD Result")
except Exception as e:
    print(f"시각화 오류 (무시 가능): {e}")
    print("포인트클라우드는 정상적으로 저장되었습니다.") 