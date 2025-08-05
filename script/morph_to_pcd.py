import os
import cv2
import open3d as o3d
import numpy as np
import pickle

def pixel_to_world_top(px, py, min_x, min_z, scale, img_height):
    wx = px / scale + min_x
    wz = (img_height - 1 - py) / scale + min_z
    return wx, wz

def pixel_to_world_side(px, py, min_x, min_y, scale, img_height):
    wx = px / scale + min_x
    wy = (img_height - 1 - py) / scale + min_y
    return wx, wy

def img_to_pcd(morph_img_path, pixel_map_path, output_pcd_path, label="", color=None):
    morph_img = cv2.imread(morph_img_path, cv2.IMREAD_GRAYSCALE)
    if morph_img is None:
        print(f"{label} 이미지 로드 실패: {morph_img_path}")
        return False, None
    print(f"{label} 이미지 크기: {morph_img.shape[1]} x {morph_img.shape[0]}")
    if not os.path.exists(pixel_map_path):
        print(f"{label} 매핑 테이블 없음: {pixel_map_path}")
        return False, None
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
        return False, None
    points_3d = np.unique(np.array(points_3d), axis=0)
    print(f"{label} 총 3D 포인트 수: {len(points_3d)} (매핑된 픽셀: {mapped_pixel_count}, 매핑X: {unmapped_pixel_count})")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    if color is not None:
        colors = np.tile(np.array(color, dtype=np.float32), (len(points_3d), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if output_pcd_path:
        output_dir = os.path.dirname(output_pcd_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        result = o3d.io.write_point_cloud(output_pcd_path, pcd)
        if result:
            print(f"✅ {label} PCD 저장 완료: {output_pcd_path}")
        else:
            print(f"❌ {label} PCD 저장 실패: {output_pcd_path}")
        return result, pcd
    else:
        print(f"✅ {label} PCD 생성 완료 (파일 저장 안 함).")
        return True, pcd

def sideview_to_3d_pcd(
    morph_img_path, sideview_pkl, topview_pkl, output_pcd_path,
    min_x_side, min_y_side, scale_side, img_height_side,
    min_x_top, min_z_top, scale_top, img_height_top,
    color=[0,0,1]
):
    morph_img = cv2.imread(morph_img_path, cv2.IMREAD_GRAYSCALE)
    if morph_img is None:
        print(f"[sideview] 이미지 로드 실패: {morph_img_path}")
        return False, None
    if not os.path.exists(topview_pkl):
        print(f"[sideview] 탑뷰 매핑 테이블 없음: {topview_pkl}")
        return False, None
    with open(topview_pkl, "rb") as f:
        pixel_to_points_top = pickle.load(f)
    white_pixels = np.argwhere(morph_img == 255)
    points_3d = []
    matched_count = 0
    for py, px in white_pixels:
        x, y = pixel_to_world_side(px, py, min_x_side, min_y_side, scale_side, img_height_side)
        z_candidates = []
        for (px_top, py_top), pts in pixel_to_points_top.items():
            x_top, z_world_top = pixel_to_world_top(px_top, py_top, min_x_top, min_z_top, scale_top, img_height_top)
            if abs(x - x_top) < (1.0 / scale_top):
                for pt in pts:
                    z_candidates.append(pt[2])
        if z_candidates:
            z = np.median(z_candidates)
            points_3d.append([x, y, z])
            matched_count += 1
    if not points_3d:
        print("[sideview] 매핑된 3D 포인트 없음")
        return False, None
    points_3d = np.unique(np.array(points_3d), axis=0)
    print(f"[sideview] 총 3D 포인트 수: {len(points_3d)} (매핑된 픽셀: {matched_count})")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    if color is not None:
        colors = np.tile(np.array(color, dtype=np.float32), (len(points_3d), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if output_pcd_path:
        output_dir = os.path.dirname(output_pcd_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        result = o3d.io.write_point_cloud(output_pcd_path, pcd)
        if result:
            print(f"✅ [sideview] PCD 저장 완료: {output_pcd_path}")
        else:
            print(f"❌ [sideview] PCD 저장 실패: {output_pcd_path}")
        return result, pcd
    else:
        print(f"✅ [sideview] PCD 생성 완료 (파일 저장 안 함).")
        return True, pcd

def filter_pcd_by_y_range(pcd, min_y, max_y, label=""):
    if not pcd.has_points():
        print(f"[{label}] 필터링할 포인트가 없습니다.")
        return o3d.geometry.PointCloud()
    points = np.asarray(pcd.points)
    colors = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    y_coords = points[:, 1]
    filtered_indices = np.where((y_coords >= min_y) & (y_coords <= max_y))[0]
    if len(filtered_indices) == 0:
        print(f"[{label}] 필터링 후 남은 3D 포인트 없음.")
        return o3d.geometry.PointCloud()
    filtered_points = points[filtered_indices]
    filtered_colors = colors[filtered_indices] if colors is not None else None
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    if filtered_colors is not None:
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    print(f"[{label}] Y 범위 필터링 완료: {len(points)} -> {len(filtered_points)} 포인트 남음 (Y 범위: [{min_y:.2f}, {max_y:.2f}])")
    return filtered_pcd

def main():
    created_pcds = []

    # --- 1. topview (탑뷰) 점군 생성 (파일 저장X, 기준점)
    topview_morph = "../output/morph/topview/morph_smoothed.png"
    topview_pkl = "../output/outline/topview/pixel_to_points.pkl"
    topview_out_pcd = "../output/pcd/final_result_topview.pcd"
    topview_color = [0, 1, 0]

    if not (os.path.exists(topview_morph) and os.path.exists(topview_pkl)):
        print("[INFO] topview 파일이 없어 이후 작업을 건너뜁니다.")
        return

    print("[INFO] topview 파일 발견, 처리에 포함합니다.")
    success_raw, raw_topview_pcd = img_to_pcd(
        topview_morph, topview_pkl,
        None, "[topview_raw]", color=topview_color
    )
    if not (success_raw and raw_topview_pcd and raw_topview_pcd.has_points()):
        print("[INFO] topview PCD 생성 실패 또는 포인트 없음. 작업 종료.")
        return

    # --- 2. sideview 점군 생성 및 Y범위 기준점 구하기 ---
    sideview_morph = "../output/morph/sideview/morph_smoothed.png"
    sideview_pkl = "../output/outline/sideview/pixel_to_points.pkl"
    topview_pkl_for_sideview = topview_pkl
    sideview_out_pcd = "../output/pcd/final_result_sideview.pcd"
    min_x_side = -51.2
    min_y_side = 0.0
    scale_side = 5
    img_height_side = 100
    min_x_top = -51.2
    min_z_top = -33.8
    scale_top = 5
    img_height_top = 163

    if os.path.exists(sideview_morph) and os.path.exists(sideview_pkl) and os.path.exists(topview_pkl_for_sideview):
        print("[INFO] sideview 파일 발견, 처리에 포함합니다.")
        success_sideview, raw_sideview_pcd = sideview_to_3d_pcd(
            sideview_morph, sideview_pkl, topview_pkl_for_sideview,
            None,
            min_x_side, min_y_side, scale_side, img_height_side,
            min_x_top, min_z_top, scale_top, img_height_top,
            color=[0,0,1]
        )
        if success_sideview and raw_sideview_pcd and raw_sideview_pcd.has_points():
            # sideview의 Y범위 계산
            sideview_points = np.asarray(raw_sideview_pcd.points)
            sideview_y_min = np.min(sideview_points[:, 1])
            sideview_y_max = np.max(sideview_points[:, 1])
            print(f"[INFO] sideview 3D Y (높이) 범위: [{sideview_y_min:.2f}, {sideview_y_max:.2f}]")

            # sideview 바닥(Y최솟값)을 topview 바닥에 맞춤 (Y축 평행이동)
            topview_points = np.asarray(raw_topview_pcd.points)
            topview_y_min = np.min(topview_points[:, 1])
            y_offset = sideview_y_min - topview_y_min
            shifted_sideview_pcd = raw_sideview_pcd.translate((0, -y_offset, 0))

            # 이동된 sideview의 Y범위로 topview 필터링
            shifted_points = np.asarray(shifted_sideview_pcd.points)
            shifted_y_min = np.min(shifted_points[:, 1])
            shifted_y_max = np.max(shifted_points[:, 1])

            filtered_topview_pcd = filter_pcd_by_y_range(
                raw_topview_pcd, shifted_y_min, shifted_y_max,
                label="topview_filtered"
            )
            if filtered_topview_pcd.has_points():
                o3d.io.write_point_cloud(topview_out_pcd, filtered_topview_pcd)
                print(f"✅ [topview] sideview Y범위로 필터된 PCD 저장 완료: {topview_out_pcd}")
                created_pcds.append(topview_out_pcd)
            else:
                print("❌ [topview] 필터 후 남은 점이 없습니다.")

            o3d.io.write_point_cloud(sideview_out_pcd, shifted_sideview_pcd)
            print(f"✅ [sideview] 평행이동된 PCD 저장 완료: {sideview_out_pcd}")
            # created_pcds.append(sideview_out_pcd)
        else:
            print("[INFO] sideview PCD 생성 실패 또는 포인트 없음. 건너뜀.")
    else:
        print("[INFO] sideview 파일/매핑 없음, 건너뜀.")

    # --- 3. floor_plane 점군 생성 (그대로) ---
    floor_plane_morph = "../output/morph/floor_plane/morph_smoothed.png"
    floor_plane_pkl = "../output/outline/floor_plane/pixel_to_points.pkl"
    floor_plane_out_pcd = "../output/pcd/final_result_floor_plane.pcd"
    floor_plane_color = [1, 0, 0]
    success_floor, _ = img_to_pcd(
        floor_plane_morph, floor_plane_pkl, floor_plane_out_pcd,
        "[floor_plane]", color=floor_plane_color
    )
    if success_floor:
        created_pcds.append(floor_plane_out_pcd)

    # --- 4. PCD 파일 병합 및 시각화 ---
    if len(created_pcds) > 0:
        combined = o3d.geometry.PointCloud()
        for path in created_pcds:
            if os.path.exists(path):
                pcd = o3d.io.read_point_cloud(path)
                combined += pcd
            else:
                print(f"병합 대상 파일 없음: {path}")
        if len(combined.points) > 0:
            out_path = "../output/pcd/final_result.pcd"
            o3d.io.write_point_cloud(out_path, combined)
            print(f"✅ 병합 PCD 저장 완료: {out_path}")
            o3d.visualization.draw_geometries([combined])
        else:
            print("병합할 PCD가 없습니다.")
    else:
        print("❌ 생성된 PCD가 없습니다.")

if __name__ == "__main__":
    main()
