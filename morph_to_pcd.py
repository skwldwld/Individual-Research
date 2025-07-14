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

# --- 바닥 평면(floor_plane) morph 결과로 PCD 생성 ---
scale_factor_floor = 3  # outline.py, morph.py에서 사용한 값과 동일하게 맞춰야 함
min_proj_x_floor = 0.0  # outline.py 로그 참고 (필요시 수정)
min_proj_z_floor = 0.0  # outline.py 로그 참고 (필요시 수정)

morph_img_path_floor = "output/morph/floor_plane/morph_smoothed.png"
pixel_map_path_floor = "output/outline/floor_plane/pixel_to_points.pkl"
output_pcd_path_floor = "output/pcd/final_result_floor_plane.pcd"

# 출력 디렉토리 생성
output_dir_floor = os.path.dirname(output_pcd_path_floor)
if not os.path.exists(output_dir_floor):
    os.makedirs(output_dir_floor)

# 1. morph 결과 이미지 로드
morph_img_floor = cv2.imread(morph_img_path_floor, cv2.IMREAD_GRAYSCALE)
if morph_img_floor is None:
    print(f"이미지 로드 실패: {morph_img_path_floor}")
else:
    img_height_floor, img_width_floor = morph_img_floor.shape
    print(f"[floor_plane] 이미지 크기: {img_width_floor} x {img_height_floor}")

    # 2. 매핑 테이블 로드
    if os.path.exists(pixel_map_path_floor):
        print(f"[floor_plane] 매핑 테이블 로드 중: {pixel_map_path_floor}")
        with open(pixel_map_path_floor, "rb") as f:
            pixel_to_points_floor = pickle.load(f)
        print(f"[floor_plane] 매핑 테이블 로드 완료: {len(pixel_to_points_floor)} 개의 픽셀-포인트 매핑")
    else:
        print(f"⚠️ [floor_plane] 매핑 테이블을 찾을 수 없습니다: {pixel_map_path_floor}")
        print("outline.py를 먼저 실행하여 매핑 테이블을 생성하세요.")
        pixel_to_points_floor = None

    if pixel_to_points_floor is not None:
        # 3. 흰색(255) 픽셀 좌표 추출
        ys_floor, xs_floor = np.where(morph_img_floor == 255)
        print(f"[floor_plane] 흰색 픽셀 수: {len(xs_floor)}")

        # 4. 2D → 3D 변환 (매핑된 픽셀만 사용)
        points_3d_floor = []
        mapped_count_floor = 0
        unmapped_count_floor = 0

        for x, y in zip(xs_floor, ys_floor):
            pixel_key = (int(x), int(y))
            if pixel_key in pixel_to_points_floor:
                original_points = pixel_to_points_floor[pixel_key]
                points_3d_floor.extend(original_points)
                mapped_count_floor += 1
            else:
                unmapped_count_floor += 1

        points_3d_floor = np.array(points_3d_floor)
        print(f"[floor_plane] 중복 제거 전 포인트 수: {len(points_3d_floor)}")

        # 중복 제거
        unique_coords_floor = set()
        unique_points_floor = []
        for point in points_3d_floor:
            coord_str = f"{point[0]:.6f},{point[1]:.6f},{point[2]:.6f}"
            if coord_str not in unique_coords_floor:
                unique_coords_floor.add(coord_str)
                unique_points_floor.append(point)
        points_3d_floor = np.array(unique_points_floor)
        print(f"[floor_plane] 중복 제거 후 포인트 수: {len(points_3d_floor)}")
        print(f"[floor_plane] 매핑된 픽셀 수: {mapped_count_floor}")
        print(f"[floor_plane] 매핑되지 않은 픽셀 수: {unmapped_count_floor}")
        print(f"[floor_plane] 총 3D 포인트 수: {len(points_3d_floor)}")

        # 5. 포인트클라우드로 저장
        if len(points_3d_floor) == 0:
            print("[floor_plane] 3D 포인트가 없습니다. 포인트클라우드를 생성할 수 없습니다.")
        else:
            pcd_floor = o3d.geometry.PointCloud()
            pcd_floor.points = o3d.utility.Vector3dVector(points_3d_floor)
            # 색상 지정 (초록색)
            colors_floor = np.array([[0, 1, 0]] * len(points_3d_floor))
            pcd_floor.colors = o3d.utility.Vector3dVector(colors_floor)
            o3d.io.write_point_cloud(output_pcd_path_floor, pcd_floor)
            print(f"✅ [floor_plane] 3D 포인트클라우드 저장 완료: {output_pcd_path_floor}")
            # 시각화
            print("\n[floor_plane] 포인트클라우드 시각화 중...")
            try:
                # open3d 0.10~0.16
                o3d.visualization.draw_geometries([pcd_floor], window_name="Morph to PCD Result (floor_plane)")
            except AttributeError:
                try:
                    # open3d 0.17+
                    o3d.visualization.draw([pcd_floor], window_name="Morph to PCD Result (floor_plane)")
                except Exception as e:
                    print(f"[floor_plane] 시각화 오류 (무시 가능): {e}")
                    print("포인트클라우드는 정상적으로 저장되었습니다.")
            except Exception as e:
                print(f"[floor_plane] 시각화 오류 (무시 가능): {e}")
                print("포인트클라우드는 정상적으로 저장되었습니다.") 

# --- above_floor와 floor_plane PCD를 합쳐서 저장 및 시각화 ---
combined_pcd_path = "output/pcd/final_result_combined.pcd"
above_pcd_path = "output/pcd/final_result.pcd"
floor_pcd_path = "output/pcd/final_result_floor_plane.pcd"

if os.path.exists(above_pcd_path) and os.path.exists(floor_pcd_path):
    print("\n[combined] 위 영역과 바닥 평면 PCD를 합칩니다...")
    above_pcd = o3d.io.read_point_cloud(above_pcd_path)
    floor_pcd = o3d.io.read_point_cloud(floor_pcd_path)
    combined_pcd = o3d.geometry.PointCloud()
    # 포인트와 색상 합치기
    combined_pcd.points = o3d.utility.Vector3dVector(
        np.vstack((np.asarray(above_pcd.points), np.asarray(floor_pcd.points)))
    )
    if above_pcd.has_colors() and floor_pcd.has_colors():
        combined_pcd.colors = o3d.utility.Vector3dVector(
            np.vstack((np.asarray(above_pcd.colors), np.asarray(floor_pcd.colors)))
        )
    o3d.io.write_point_cloud(combined_pcd_path, combined_pcd)
    print(f"✅ [combined] 합쳐진 PCD 저장 완료: {combined_pcd_path}")
    # 시각화
    print("[combined] 합쳐진 포인트클라우드 시각화 중...")
    try:
        # open3d 0.10~0.16
        o3d.visualization.draw_geometries([combined_pcd], window_name="Morph to PCD Result (combined)")
    except AttributeError:
        try:
            # open3d 0.17+
            o3d.visualization.draw([combined_pcd], window_name="Morph to PCD Result (combined)")
        except Exception as e:
            print(f"[combined] 시각화 오류 (무시 가능): {e}")
            print("포인트클라우드는 정상적으로 저장되었습니다.")
    except Exception as e:
        print(f"[combined] 시각화 오류 (무시 가능): {e}")
        print("포인트클라우드는 정상적으로 저장되었습니다.")
else:
    print("[combined] 위 영역 또는 바닥 평면 PCD가 존재하지 않아 합칠 수 없습니다.") 