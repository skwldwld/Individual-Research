# outline.py
import open3d as o3d
import numpy as np
import cv2
from pathlib import Path
import os
from sklearn.cluster import DBSCAN
import pickle

def extract_outline(
    input_path, output_dir,
    dim=(0, 2), scale_factor=5,
    contour_min=10,
    dilate_kernel_size=1, dilate_iterations=0,
    contour_thickness=2, contour_color=(0, 255, 255),
    dbscan_eps=0.03, dbscan_min=30, dot_size=2,
    plane_name="topview"
):
    if scale_factor <= 0:
        print("[ERROR] scale_factor must be > 0")
        return False

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- 2D 윤곽선 추출 시작: {plane_name} ---")
    pcd = o3d.io.read_point_cloud(str(input_path))
    if not pcd.has_points():
        print(f"[ERROR] 빈 포인트 클라우드: {input_path}")
        return False

    # 약 10만 pts 목표 다운샘플(안전)
    max_points = 100_000
    if len(pcd.points) > max_points:
        voxel = 0.03
        while True:
            tmp = pcd.voxel_down_sample(voxel_size=voxel)
            if len(tmp.points) <= max_points or voxel > 1.0:
                pcd = tmp
                break
            voxel *= 1.5
        print(f"[INFO] 다운샘플: {len(pcd.points)} pts (voxel={voxel:.4f})")

    pts = np.asarray(pcd.points)
    proj = pts[:, dim]  # Nx2

    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min)
    labels = db.fit_predict(proj)
    uniq = np.unique(labels)
    clusters = [k for k in uniq if k != -1]
    if (labels == -1).any():
        print(f"[INFO] 노이즈: {(labels == -1).sum()} pts")
    print(f"[INFO] 클러스터: {len(clusters)}")

    (min0, min1) = np.min(proj, axis=0)
    (max0, max1) = np.max(proj, axis=0)
    w = max(int((max0 - min0) * scale_factor) + 1, 10)
    h = max(int((max1 - min1) * scale_factor) + 1, 10)
    print(f"[INFO] 캔버스: {w} x {h}")

    binary = np.zeros((h, w), dtype=np.uint8)
    contour_img = np.zeros((h, w, 3), dtype=np.uint8)
    pixel_to_points = {}
    all_contours = []

    kernel = None
    if dilate_iterations > 0:
        k = int(max(1, dilate_kernel_size))
        kernel = np.ones((k, k), np.uint8)

    half = max(dot_size // 2, 0)
    for k in clusters:
        mask = (labels == k)
        cluster_xy = proj[mask]
        orig_points = pts[mask]
        if cluster_xy.shape[0] < contour_min:
            continue

        px = np.round((cluster_xy[:, 0] - min0) * scale_factor).astype(np.int32)
        py = np.round((cluster_xy[:, 1] - min1) * scale_factor).astype(np.int32)
        py = h - 1 - py  # 영상 좌표 보정

        # 픽셀→3D 포인트 매핑 저장(후속 단계 사용)
        for i, (x, y) in enumerate(zip(px, py)):
            key = (int(x), int(y))
            if 0 <= x < w and 0 <= y < h:
                if key not in pixel_to_points:
                    pixel_to_points[key] = []
                pixel_to_points[key].append(orig_points[i])

        canvas = np.zeros((h, w), dtype=np.uint8)
        if half > 0:
            for x, y in zip(px, py):
                if 0 <= x < w and 0 <= y < h:
                    cv2.rectangle(canvas, (x - half, y - half), (x + half, y + half), 255, -1)
        else:
            valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            canvas[py[valid], px[valid]] = 255

        if kernel is not None:
            canvas = cv2.dilate(canvas, kernel, iterations=dilate_iterations)
            canvas = cv2.erode(canvas, kernel, iterations=1)

        contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned = []
        for c in contours:
            if cv2.contourArea(c) <= contour_min:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.002 * peri, True)
            cleaned.append(approx)
        if not cleaned:
            continue

        all_contours.extend(cleaned)
        cv2.drawContours(contour_img, cleaned, -1, contour_color, contour_thickness)
        cv2.drawContours(binary, cleaned, -1, 255, -1)

    # 저장
    (output_dir / "binary.png").parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "binary.png"), binary)
    cv2.imwrite(str(output_dir / "contours.png"), contour_img)
    with open(output_dir / "pixel_to_points.pkl", "wb") as f:
        pickle.dump(pixel_to_points, f)

    print(f"[OK] {plane_name} → {output_dir}")
    print(f"     - binary.png / contours.png / pixel_to_points.pkl")
    return True


def resolve(p: str) -> Path:
    return (Path(__file__).resolve().parent / p).resolve()

def main():
    # RANSAC 산출물을 기본 입력으로 사용
    floor_pcd = resolve("../output/ransac/floor_plane.pcd")
    top_pcd   = resolve("../output/ransac/topview.pcd")
    walls_pcd = resolve("../output/removed_walls.pcd")  # remove.py가 저장

    # 1) 바닥 (top view: x-z)
    if floor_pcd.exists():
        extract_outline(
            input_path=floor_pcd,
            output_dir=resolve("../output/outline/floor"),
            dim=(0, 2),
            scale_factor=3,
            contour_min=20,
            dilate_kernel_size=1, dilate_iterations=0,
            contour_thickness=1, contour_color=(0, 255, 0),
            dbscan_eps=2.0, dbscan_min=8, dot_size=2,
            plane_name="floor_topview"
        )
    else:
        print(f"[SKIP] floor not found: {floor_pcd}")

    # 2) 상부 (top view / side view) — 입력: ransac/topview.pcd
    if top_pcd.exists():
        extract_outline(
            input_path=top_pcd,
            output_dir=resolve("../output/outline/topview/material"),
            dim=(0, 2),
            scale_factor=10,
            contour_min=400,
            dilate_kernel_size=1, dilate_iterations=0,
            contour_thickness=1, contour_color=(0, 0, 255),
            dbscan_eps=1.5, dbscan_min=60, dot_size=3,
            plane_name="topview"
        )
        extract_outline(
            input_path=top_pcd,
            output_dir=resolve("../output/outline/sideview/material"),
            dim=(0, 1),
            scale_factor=5,
            contour_min=400,
            dilate_kernel_size=1, dilate_iterations=0,
            contour_thickness=1, contour_color=(255, 0, 255),
            dbscan_eps=1.5, dbscan_min=30, dot_size=2,
            plane_name="sideview"
        )
    else:
        print(f"[SKIP] topview not found: {top_pcd}")

    # 3) 벽 (top/side) — 입력: output/removed_walls.pcd (remove.py)
    if walls_pcd.exists():
        extract_outline(
            input_path=walls_pcd,
            output_dir=resolve("../output/outline/topview/wall"),
            dim=(0, 2),
            scale_factor=5,
            contour_min=400,
            dilate_kernel_size=1, dilate_iterations=0,
            contour_thickness=1, contour_color=(0, 0, 255),
            dbscan_eps=1.5, dbscan_min=60, dot_size=4,
            plane_name="walls_topview"
        )
        extract_outline(
            input_path=walls_pcd,
            output_dir=resolve("../output/outline/sideview/wall"),
            dim=(0, 1),
            scale_factor=6,
            contour_min=200,
            dilate_kernel_size=1, dilate_iterations=0,
            contour_thickness=2, contour_color=(255, 0, 255),
            dbscan_eps=1.2, dbscan_min=40, dot_size=2,
            plane_name="walls_sideview"
        )
    else:
        print(f"[SKIP] walls not found: {walls_pcd}")

if __name__ == "__main__":
    main()
