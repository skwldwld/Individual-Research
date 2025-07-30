import os
import cv2
import open3d as o3d
import numpy as np
import pickle

def img_to_pcd(morph_img_path, pixel_map_path, output_pcd_path, label="", color=None):
    """
    2D 바이너리 이미지와 픽셀-3D포인트 매핑 테이블을 이용해 PCD 파일을 생성합니다.
    color: (R, G, B) 0~1 float, 모든 포인트에 동일 색상 적용
    """
    morph_img = cv2.imread(morph_img_path, cv2.IMREAD_GRAYSCALE)
    if morph_img is None:
        print(f"{label} 이미지 로드 실패: {morph_img_path}")
        return False
    print(f"{label} 이미지 크기: {morph_img.shape[1]} x {morph_img.shape[0]}")

    if not os.path.exists(pixel_map_path):
        print(f"{label} 매핑 테이블 없음: {pixel_map_path}")
        return False
    with open(pixel_map_path, "rb") as f:
        pixel_to_points = pickle.load(f)
    print(f"{label} 매핑 테이블 로드 완료: {len(pixel_to_points)}개")

    white_pixels = np.argwhere(morph_img == 255)
    points_3d = []
    mapped_pixel_count = 0
    unmapped_pixel_count = 0
    for py, px in white_pixels:
        key = (int(px), int(py))
        if key in pixel_to_points:
            points_3d.extend(pixel_to_points[key])
            mapped_pixel_count += 1
        else:
            unmapped_pixel_count += 1
    if not points_3d:
        print(f"{label} 3D 포인트 없음")
        return False
    points_3d = np.unique(np.array(points_3d), axis=0)
    print(f"{label} 총 3D 포인트 수: {len(points_3d)} (매핑된 픽셀: {mapped_pixel_count}, 매핑X: {unmapped_pixel_count})")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    # 색상 지정
    if color is not None:
        colors = np.tile(np.array(color, dtype=np.float32), (len(points_3d), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    # PCD 저장 전 폴더 생성
    output_dir = os.path.dirname(output_pcd_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # PCD 저장
    result = o3d.io.write_point_cloud(output_pcd_path, pcd)
    if result:
        print(f"✅ {label} PCD 저장 완료: {output_pcd_path}")
    else:
        print(f"❌ {label} PCD 저장 실패: {output_pcd_path}")
    return result

def merge_pcds(pcd_paths, output_path):
    """
    여러 PCD 파일을 합쳐서 하나의 PCD로 저장합니다.
    색상 정보도 함께 병합합니다.
    """
    combined = o3d.geometry.PointCloud()
    for path in pcd_paths:
        if os.path.exists(path):
            pcd = o3d.io.read_point_cloud(path)
            combined += pcd
        else:
            print(f"병합 대상 파일 없음: {path}")
    if len(combined.points) > 0:
        o3d.io.write_point_cloud(output_path, combined)
        print(f"✅ 병합 PCD 저장 완료: {output_path}")
        print("[combined] 합쳐진 포인트클라우드 시각화 중...")
        o3d.visualization.draw_geometries([combined])
        return True
    else:
        print("병합할 PCD가 없습니다.")
        return False

def main():
    # 경로/라벨/색상 정의 (실제 존재하는 파일들만 포함)
    jobs = [
        {
            "label": "[floor_plane]",
            "morph_img": "../output/morph/floor_plane/morph_smoothed.png",
            "pixel_map": "../output/outline/floor_plane/pixel_to_points.pkl",
            "output_pcd": "../output/pcd/final_result_floor_plane.pcd",
            "color": [1, 0, 0]  # 빨간색
        }
    ]
    
    # above_floor 파일이 존재하면 추가 (front_view 대신)
    above_floor_morph = "../output/morph/above_floor/morph_smoothed.png"
    above_floor_pkl = "../output/outline/above_floor/pixel_to_points.pkl"
    if os.path.exists(above_floor_morph) and os.path.exists(above_floor_pkl):
        jobs.insert(0, {
            "label": "[above_floor]",
            "morph_img": above_floor_morph,
            "pixel_map": above_floor_pkl,
            "output_pcd": "../output/pcd/final_result_above_floor.pcd",
            "color": [0, 1, 0]  # 초록색
        })
        print("[INFO] above_floor 파일 발견, 처리에 포함합니다.")
    else:
        print("[INFO] above_floor 파일이 없어서 건너뜁니다.")
    # 각 영역별 PCD 생성
    created_pcds = []
    for job in jobs:
        if img_to_pcd(job["morph_img"], job["pixel_map"], job["output_pcd"], job["label"], color=job["color"]):
            created_pcds.append(job["output_pcd"])

    # 병합 (생성된 PCD들만)
    if len(created_pcds) > 1:
        merge_pcds(created_pcds, "../output/pcd/final_result.pcd")
    elif len(created_pcds) == 1:
        # 하나만 있으면 그대로 복사
        import shutil
        shutil.copy2(created_pcds[0], "../output/pcd/final_result.pcd")
        print(f"✅ 단일 PCD 복사 완료: {created_pcds[0]} -> ../output/pcd/final_result.pcd")
    else:
        print("❌ 생성된 PCD가 없습니다.")

if __name__ == "__main__":
    main() 