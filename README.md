npx -y @modelcontextprotocol/server-filesystem C:/Users/yejim/Desktop/yeji/Individual-Research


# 3D 포인트클라우드 기반 의미 있는 물체 추출 및 미니맵 생성 파이프라인

## 개요

이 프로젝트는 3D 포인트클라우드(예: `.ply`, `.pcd`)에서  
- **RANSAC**으로 바닥 검출  
- **DBSCAN**으로 의미 있는 물체 분리  
- **2D 윤곽선 추출 및 모폴로지 영상처리**  
- **최종적으로 3D 미니맵(의미 있는 물체 + 바닥)**  
을 자동으로 생성하는 파이프라인입니다.

---

## 폴더 구조 및 주요 입출력

```
input/
  └─ 3BP_CS_model_cut_180.ply, 3BP_ascii.pcd 등 (입력 포인트클라우드)
output/
  ├─ ransac/
  │    ├─ floor.pcd         # 바닥만 추출
  │    └─ non_floor.pcd     # 바닥 제외 나머지
  ├─ outline/
  │    ├─ binary.png        # 2D 투영 이진 이미지
  │    ├─ contours.png      # 2D 윤곽선 시각화
  │    └─ pixel_to_points.pkl # 2D-3D 매핑
  ├─ morph/
  │    ├─ eroded.png, dilated.png, morph_smoothed.png # 모폴로지 처리 결과
  └─ pcd/
       ├─ final_result.pcd  # 의미 있는 물체만 남은 3D 결과
       └─ with_floor.pcd    # 바닥 포함 최종 3D 결과
```

---

## 설치

```bash
pip install -r requirements.txt
```

필요 패키지:  
- open3d
- numpy
- opencv-python
- scikit-learn
- matplotlib

---

## 전체 파이프라인 요약

### 1. 바닥/비바닥 분리 (RANSAC 등)
- (예시) `ransac.py` 또는 별도 코드로 바닥/비바닥 분리
- 결과: `output/ransac/floor.pcd`, `output/ransac/non_floor.pcd`

### 2. 2D 윤곽선 추출 및 DBSCAN 클러스터링

```bash
python outline.py
```
- 입력: `output/ransac/non_floor.pcd`
- 출력: `output/outline/binary.png`, `output/outline/contours.png`, `output/outline/pixel_to_points.pkl`

### 3. 모폴로지 영상처리

```bash
python morph.py
```
- 입력: `output/outline/binary.png`
- 출력: `output/morph/morph_smoothed.png` 등

### 4. 2D → 3D 포인트클라우드 복원

```bash
python morph_to_pcd.py
```
- 입력: `output/morph/morph_smoothed.png`
- 출력: `output/pcd/final_result.pcd`  
  (의미 있는 물체만 남은 3D 포인트클라우드, 파란색)

### 5. 바닥면과 합치기 (선택)

```python
import open3d as o3d
import numpy as np

obj = o3d.io.read_point_cloud("output/pcd/final_result.pcd")
floor = o3d.io.read_point_cloud("output/ransac/floor.pcd")
merged = o3d.geometry.PointCloud()
merged.points = o3d.utility.Vector3dVector(np.vstack((np.asarray(obj.points), np.asarray(floor.points))))
# (필요시 색상도 합치기)
o3d.io.write_point_cloud("output/pcd/with_floor.pcd", merged)
```

---

## 각 코드 설명

### outline.py
- 3D 포인트클라우드를 XZ 평면(Top-Down)으로 투영
- DBSCAN으로 클러스터 분리, 2D 윤곽선 추출
- 2D-3D 매핑 정보(`pixel_to_points.pkl`) 저장

### morph.py
- 2D 바이너리 이미지에 모폴로지 연산(침식, 팽창, 열림-닫힘) 적용
- 노이즈 제거 및 윤곽 부드럽게

### morph_to_pcd.py
- 모폴로지 결과 이미지의 흰색 픽셀을 3D로 복원
- (Y값은 임의, XZ는 원래 좌표계와 일치)
- 모든 포인트에 파란색(RGB: 0,0,1) 지정

---

## 참고/팁

- 바닥면에도 동일한 파이프라인 적용 가능 (파인 부분 등 분석)
- DBSCAN, 모폴로지 파라미터는 데이터에 따라 조정 필요
- 3D 결과는 Open3D 등으로 시각화 가능
