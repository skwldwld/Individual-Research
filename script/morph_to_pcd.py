import os
import cv2
import open3d as o3d
import numpy as np
import pickle

# ==============================
# 설정
# ==============================
VOXEL_SIZE = 0.01   # 1cm 다운샘플 (필요시 조절)
MERGE_AT_END = False  # 최종 병합 금지(원하면 True)

# ------------------------------
# 픽셀 ↔ 월드 좌표 변환 유틸
# ------------------------------
def pixel_to_world_top(px, py, min_x, min_z, scale, img_height):
    wx = px / scale + min_x
    wz = (img_height - 1 - py) / scale + min_z
    return wx, wz

def pixel_to_world_side(px, py, min_x, min_y, scale, img_height):
    wx = px / scale + min_x
    wy = (img_height - 1 - py) / scale + min_y
    return wx, wy

# ------------------------------
# 공통: PCD 다운샘플 + 저장
# ------------------------------
def _voxel_down(pcd, voxel_size=VOXEL_SIZE):
    if pcd is None or not pcd.has_points():
        return pcd
    return pcd.voxel_down_sample(voxel_size)

def _save_pcd(path, pcd, label=""):
    if pcd is None or not pcd.has_points():
        print(f"[{label}] 저장할 포인트가 없습니다.")
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = o3d.io.write_point_cloud(path, pcd)
    if ok:
        print(f"✅ [{label}] PCD 저장: {path}  (points={len(pcd.points)})")
    else:
        print(f"❌ [{label}] PCD 저장 실패: {path}")
    return ok

# ------------------------------
# 바이너리(morph) → PCD 변환
# ------------------------------
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

    # 다운샘플 후 저장/반환
    pcd = _voxel_down(pcd, VOXEL_SIZE)

    if output_pcd_path:
        ok = _save_pcd(output_pcd_path, pcd, label=label)
        return ok, pcd
    else:
        print(f"✅ {label} PCD 생성 완료 (파일 저장 안 함).")
        return True, pcd
    
def build_top_x_to_z_lookup(topview_pkl, min_x_top, min_z_top, scale_top, img_height_top):
    # 같은 world x-bin(= 픽셀 폭 1/scale_top) 안의 z들을 모아서 중앙값으로 요약
    with open(topview_pkl, "rb") as f:
        pixel_to_points_top = pickle.load(f)

    # bin 인덱스 = round((x_world - min_x_top) * scale_top)
    buckets = {}
    for (px_top, py_top), pts in pixel_to_points_top.items():
        x_world, _ = pixel_to_world_top(px_top, py_top, min_x_top, min_z_top, scale_top, img_height_top)
        bin_idx = int(round((x_world - min_x_top) * scale_top))
        zs = [p[2] for p in pts]
        if bin_idx in buckets:
            buckets[bin_idx].extend(zs)
        else:
            buckets[bin_idx] = list(zs)

    # 중앙값으로 축약
    x_to_zmed = {k: float(np.median(v)) for k, v in buckets.items() if len(v) > 0}
    return x_to_zmed


# ----------------------------------------
# sideview 2D → 3D 복원 (topview 매핑과 결합)
# ----------------------------------------
def sideview_to_3d_pcd(
    morph_img_path, sideview_pkl, topview_pkl, output_pcd_path,
    min_x_side, min_y_side, scale_side, img_height_side,
    min_x_top, min_z_top, scale_top, img_height_top,
    color=[0, 0, 1],
    stride=1  # 성능튜닝: 1=모두, 2/3/4=샘플링
):
    morph_img = cv2.imread(morph_img_path, cv2.IMREAD_GRAYSCALE)
    if morph_img is None:
        print(f"[sideview] 이미지 로드 실패: {morph_img_path}")
        return False, None

    if not os.path.exists(topview_pkl):
        print(f"[sideview] 탑뷰 매핑 테이블 없음: {topview_pkl}")
        return False, None

    # 1) 탑뷰를 world x-bin -> z중앙값 룩업으로 전처리 (한 번만!)
    x_to_zmed = build_top_x_to_z_lookup(topview_pkl, min_x_top, min_z_top, scale_top, img_height_top)

    # 2) 사이드 흰 픽셀 추출 + (선택) 샘플링
    white_pixels = np.argwhere(morph_img == 255)
    if stride > 1:
        white_pixels = white_pixels[::stride]  # 간단 샘플링

    if white_pixels.size == 0:
        print("[sideview] 흰 픽셀이 없습니다.")
        return False, None

    # 3) 벡터화로 side 픽셀→world (x,y)
    pys = white_pixels[:, 0].astype(np.int32)
    pxs = white_pixels[:, 1].astype(np.int32)
    xs_world = pxs / scale_side + min_x_side
    ys_world = (img_height_side - 1 - pys) / scale_side + min_y_side

    # 4) 준비한 룩업에서 z를 즉시 조회 (bin폭=1/scale_top)
    bin_idx = np.round((xs_world - min_x_top) * scale_top).astype(np.int32)
    zs_world = np.empty_like(xs_world, dtype=np.float32)
    mask_hit = np.zeros_like(xs_world, dtype=bool)
    for i, b in enumerate(bin_idx):
        z = x_to_zmed.get(int(b), None)
        if z is not None:
            zs_world[i] = z
            mask_hit[i] = True

    # 5) 유효한 것만 포인트로 구성
    if not np.any(mask_hit):
        print("[sideview] 매핑된 3D 포인트 없음")
        return False, None

    points_3d = np.stack([xs_world[mask_hit], ys_world[mask_hit], zs_world[mask_hit]], axis=1)
    points_3d = np.unique(points_3d, axis=0)
    print(f"[sideview] 총 3D 포인트 수: {len(points_3d)} (매핑된 픽셀: {np.count_nonzero(mask_hit)} / stride={stride})")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    if color is not None:
        colors = np.tile(np.array(color, dtype=np.float32), (len(points_3d), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # 다운샘플 후 저장/반환 (너 코드 스타일 유지)
    pcd = _voxel_down(pcd, VOXEL_SIZE)

    if output_pcd_path:
        ok = _save_pcd(output_pcd_path, pcd, label="sideview")
        return ok, pcd
    else:
        print(f"✅ [sideview] PCD 생성 완료 (파일 저장 안 함).")
        return True, pcd



# ------------------------------
# Y 범위로 PCD 필터링
# ------------------------------
def filter_pcd_by_y_range(pcd, min_y, max_y, label=""):
    if not pcd.has_points():
        print(f"[{label}] 필터링할 포인트가 없습니다.")
        return o3d.geometry.PointCloud()

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    y_coords = points[:, 1]
    filtered_indices = np.where((y_coords >= min_y) & (y_coords <= max_y))[0]

    if len(filtered_indices) == 0:
        print(f"[{label}] 필터링 후 남은 3D 포인트 없음.")
        return o3d.geometry.PointCloud()

    filtered_points = points[filtered_indices]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    if colors is not None:
        filtered_colors = colors[filtered_indices]
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    print(f"[{label}] Y 범위 필터링 완료: {len(points)} -> {len(filtered_points)} 포인트 남음 (Y 범위: [{min_y:.2f}, {max_y:.2f}])")
    return filtered_pcd

# ------------------------------
# 메인: 경로 구조 정리 및 처리
# ------------------------------
def main():
    created_pcds = []

    # side→3D 복원 시 사용하는 좌표/스케일 파라미터 (프로젝트 값으로 조정)
    min_x_side = -51.2
    min_y_side = 0.0
    scale_side = 5
    img_height_side = 100

    min_x_top = -51.2
    min_z_top = -33.8
    scale_top = 5
    img_height_top = 163

    # ===== 1) material / wall 각각 처리 (개별 저장만) =====
    for category, color_top, color_side in [
        ("material", [0, 1, 0], [0, 0, 1]),  # topview=green, sideview=blue
        ("wall",     [1, 0, 1], [0, 0, 1]),  # topview=magenta, sideview=blue
    ]:
        print(f"\n===== [{category}] 처리 시작 =====")

        # --- 1-1. topview (탑뷰) 기준 PCD 생성 ---
        topview_morph   = f"../output/morph/topview/{category}/morph_smoothed.png"
        topview_pkl     = f"../output/outline/topview/{category}/pixel_to_points.pkl"
        topview_out_pcd = f"../output/pcd/topview/{category}.pcd"

        if not (os.path.exists(topview_morph) and os.path.exists(topview_pkl)):
            print(f"[INFO] topview {category} 파일이 없어 이 카테고리는 건너뜁니다.")
            continue

        print(f"[INFO] topview {category} 파일 발견, 처리에 포함합니다.")
        success_raw, raw_topview_pcd = img_to_pcd(
            topview_morph, topview_pkl,
            None, f"[topview_raw_{category}]", color=color_top
        )
        if not (success_raw and raw_topview_pcd and raw_topview_pcd.has_points()):
            print(f"[INFO] topview {category} PCD 생성 실패 또는 포인트 없음. 이 카테고리 건너뜀.")
            continue

        # --- 1-2. sideview → 3D 복원, Y정렬, topview 필터 ---
        sideview_morph   = f"../output/morph/sideview/{category}/morph_smoothed.png"
        sideview_pkl     = f"../output/outline/sideview/{category}/pixel_to_points.pkl"
        sideview_out_pcd = f"../output/pcd/sideview/{category}.pcd"

        if os.path.exists(sideview_morph) and os.path.exists(sideview_pkl) and os.path.exists(topview_pkl):
            print(f"[INFO] sideview {category} 파일 발견, 처리에 포함합니다.")
            success_sideview, raw_sideview_pcd = sideview_to_3d_pcd(
                sideview_morph, sideview_pkl, topview_pkl,
                None,
                min_x_side, min_y_side, scale_side, img_height_side,
                min_x_top,  min_z_top,  scale_top,  img_height_top,
                color=color_side,
                stride=(3 if category == "wall" else 1)  # 벽만 3픽셀 간격 샘플링
            )
            if success_sideview and raw_sideview_pcd and raw_sideview_pcd.has_points():
                # sideview 바닥(Y최솟값)을 topview 바닥에 정렬 (Y축 평행이동)
                sideview_points = np.asarray(raw_sideview_pcd.points)
                sideview_y_min = np.min(sideview_points[:, 1])

                topview_points = np.asarray(raw_topview_pcd.points)
                topview_y_min = np.min(topview_points[:, 1])
                y_offset = sideview_y_min - topview_y_min
                shifted_sideview_pcd = raw_sideview_pcd.translate((0, -y_offset, 0))

                # 이동된 sideview의 Y범위로 topview 필터링 후 저장
                shifted_points = np.asarray(shifted_sideview_pcd.points)
                shifted_y_min = np.min(shifted_points[:, 1])
                shifted_y_max = np.max(shifted_points[:, 1])

                filtered_topview_pcd = filter_pcd_by_y_range(
                    raw_topview_pcd, shifted_y_min, shifted_y_max,
                    label=f"topview_filtered_{category}"
                )

                # 다운샘플 후 각각 저장
                if filtered_topview_pcd.has_points():
                    filtered_topview_pcd = _voxel_down(filtered_topview_pcd, VOXEL_SIZE)
                    _save_pcd(topview_out_pcd, filtered_topview_pcd, label=f"topview/{category}")
                    created_pcds.append(topview_out_pcd)
                else:
                    print(f"❌ [topview/{category}] 필터 후 남은 점이 없습니다.")

                shifted_sideview_pcd = _voxel_down(shifted_sideview_pcd, VOXEL_SIZE)
                _save_pcd(sideview_out_pcd, shifted_sideview_pcd, label=f"sideview/{category}")
                # 필요시 sideview도 병합 목록에 포함
                # created_pcds.append(sideview_out_pcd)
            else:
                print(f"[INFO] sideview {category} PCD 생성 실패 또는 포인트 없음. 이 단계 건너뜀.")
        else:
            print(f"[INFO] sideview {category} 파일/매핑 없음, 건너뜀.")

    # ===== 2) floor_plane (바닥) 별도 처리 =====
    floor_plane_morph   = "../output/morph/floor/morph_smoothed.png"
    floor_plane_pkl     = "../output/outline/floor/pixel_to_points.pkl"
    floor_plane_out_pcd = "../output/pcd/floor/floor_plane.pcd"
    floor_plane_color   = [1, 0, 0]

    success_floor, floor_pcd = img_to_pcd(
        floor_plane_morph, floor_plane_pkl, None,
        "[floor_plane]", color=floor_plane_color
    )
    if success_floor and floor_pcd is not None and floor_pcd.has_points():
        floor_pcd = _voxel_down(floor_pcd, VOXEL_SIZE)
        if _save_pcd(floor_plane_out_pcd, floor_pcd, label="floor_plane"):
            created_pcds.append(floor_plane_out_pcd)

    # main() 끝부분 병합 단계 위쪽에 추가
    # ===== 4) 생성된 PCD 시각화 =====
    print("[INFO] 시각화 시작...")
    pcds_for_viz = []
    for pcd_path in created_pcds:
        if os.path.exists(pcd_path):
            try:
                pcd = o3d.io.read_point_cloud(pcd_path)
                if pcd.has_points():
                    pcds_for_viz.append(pcd)
                    print(f"  [LOAD] {pcd_path} ({len(pcd.points)} points)")
                else:
                    print(f"  [SKIP] {pcd_path} -> no points")
            except Exception as e:
                print(f"  [ERROR] Failed to load {pcd_path}: {e}")

    if pcds_for_viz:
        # material / wall 색상 확인 위해 mesh_show_back_face=True 옵션
        o3d.visualization.draw_geometries(
            pcds_for_viz,
            window_name="Material & Wall PCD Visualization",
            mesh_show_back_face=True
        )
    else:
        print("[WARNING] 시각화할 PCD 없음.")


    # ===== 3) 최종 병합 저장/시각화 (기본 비활성) =====
    if MERGE_AT_END and len(created_pcds) > 0:
        combined = o3d.geometry.PointCloud()
        for path in created_pcds:
            if os.path.exists(path):
                pcd = o3d.io.read_point_cloud(path)
                combined += pcd
            else:
                print(f"병합 대상 파일 없음: {path}")

        if len(combined.points) > 0:
            out_path = "../output/pcd/final_result.pcd"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            o3d.io.write_point_cloud(out_path, combined)
            print(f"✅ 병합 PCD 저장 완료: {out_path}")
            # o3d.visualization.draw_geometries([combined])  # 필요시 시각화
        else:
            print("병합할 PCD가 없습니다.")
    else:
        print("병합 단계는 건너뜀.")

if __name__ == "__main__":
    main()
