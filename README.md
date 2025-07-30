# 3D Building Mesh Generation Pipeline

3D 포인트 클라우드에서 건물의 완전한 3D 메쉬를 생성하는 파이프라인입니다.

## 📁 프로젝트 구조

```
Individual-Research/
├── input/                          # 입력 파일
│   ├── 3BP_CS_model_Cloud.pcd     # 원본 포인트 클라우드
│   └── 3BP_CS_model.ply           # 원본 메쉬
├── script/                         # 실행 스크립트
│   ├── nonowall.py                # 벽 제거
│   ├── ransac.py                  # 바닥 평면 분리
│   ├── outline.py                 # 2D 윤곽선 추출
│   ├── morph.py                   # 형태학적 처리
│   ├── pcd_to_mesh.py             # 3D 메쉬 생성
│   └── run_full_pipeline.py       # 전체 파이프라인 실행
├── output/                         # 출력 결과
│   ├── nonowall.pcd               # 벽 제거된 PCD
│   ├── ransac/                    # RANSAC 결과
│   ├── outline/                   # 2D 윤곽선
│   ├── morph/                     # 형태학 처리 결과
│   ├── pcd/                       # 최종 PCD 파일
│   └── mesh/                      # 최종 메쉬 파일
└── requirements.txt               # 필요한 패키지
```

## 🚀 실행 방법

### 전체 파이프라인 실행:
```bash
cd script
python run_full_pipeline.py
```

### 개별 단계 실행:
```bash
cd script
python nonowall.py    # 1단계: 벽 제거
python ransac.py      # 2단계: 바닥 평면 분리
python outline.py     # 3단계: 2D 윤곽선 추출
python morph.py       # 4단계: 형태학적 처리
python morph_to_pcd.py # 5단계: 
python pcd_to_mesh.py # 6단계: 3D 메쉬 생성
```

## 🔬 각 과정의 원리와 작동 과정

### 1. **nonowall.py - 벽 제거 (Wall Removal)**

#### **원리:**
- 건물의 외벽을 제거하여 내부 구조만 남기는 과정
- 박스 필터링 기반으로 경계 영역의 점들을 제거

#### **작동 과정:**
1. **포인트 클라우드 로드**: `input/3BP_CS_model_Cloud.pcd` 로드
2. **경계 영역 식별**: X, Z 방향의 경계에서 가장 밀도가 높은 영역 탐지
3. **박스 필터링**: 
   - 경계 두께: 9.0m
   - 상위 2개 방향 선택 (예: 'z-', 'z+')
   - 해당 영역의 점들을 벽으로 식별
4. **벽 제거**: 벽으로 식별된 점들을 제거
5. **다운샘플링**: voxel_size=0.03으로 다운샘플링
6. **결과 저장**: `output/nonowall.pcd`

#### **핵심 파라미터:**
- `boundary_thickness`: 9.0m (벽 두께)
- `voxel_size`: 0.03 (다운샘플링 크기)

---

### 2. **ransac.py - RANSAC 바닥 평면 분리**

#### **원리:**
- RANSAC (Random Sample Consensus) 알고리즘을 사용하여 다중 평면 검출
- Y축과 평행한 평면들을 바닥으로 식별
- 바닥 영역을 확장하여 바닥과 객체 간의 연결 부분 제거

#### **작동 과정:**
1. **포인트 클라우드 로드**: `output/nonowall.pcd` 로드
2. **RANSAC 평면 검출**:
   - 최대 40개 평면 검출
   - `distance_threshold`: 0.02m (평면 정밀도)
   - `ransac_n`: 3 (최소 샘플 수)
   - `num_iterations`: 1000 (반복 횟수)
3. **바닥 평면 식별**:
   - Y축과의 각도가 30도 이하인 평면을 바닥으로 식별
   - 법선 벡터와 Y축 [0,1,0] 간의 각도 계산
4. **바닥 영역 확장**:
   - `vertical_offset`: 1.5m (바닥 영역 위로 확장)
   - 바닥과 객체 간의 지저분한 연결 부분 제거
5. **결과 저장**:
   - `output/ransac/floor_plane.pcd`: 바닥 평면
   - `output/ransac/above_floor.pcd`: 바닥 위 영역

#### **핵심 파라미터:**
- `max_planes`: 40 (최대 평면 수)
- `distance_threshold`: 0.02m
- `vertical_offset`: 1.5m (바닥 영역 확장)

---

### 3. **outline.py - 2D 윤곽선 추출**

#### **원리:**
- 3D 포인트 클라우드를 Y축 위쪽에서 바라보는 2D 투영
- DBSCAN 클러스터링으로 노이즈 제거 및 객체 분리
- 컨투어 추출을 통한 2D 윤곽선 생성

#### **작동 과정:**
1. **포인트 클라우드 로드**: `output/ransac/above_floor.pcd`, `output/ransac/floor_plane.pcd`
2. **2D 투영**:
   - Y축 위쪽에서 바라보기 위해 X-Z 평면으로 투영
   - 3D 좌표 (x,y,z) → 2D 좌표 (x,z)
3. **스케일링**: 적절한 이미지 크기를 위한 스케일 팩터 계산
4. **DBSCAN 클러스터링**:
   - 노이즈 제거 (label = -1인 점들)
   - 객체 분리 및 클러스터링
   - 파라미터: `eps=1.5`, `min_samples=15`
5. **컨투어 추출**:
   - 각 클러스터별로 이진 이미지 생성
   - `cv2.findContours`로 윤곽선 추출
   - 면적 필터링 및 근사화
6. **결과 저장**:
   - `output/outline/top_view/binary.png`: 이진 이미지
   - `output/outline/top_view/contours.png`: 윤곽선 이미지
   - `output/outline/top_view/pixel_to_points.pkl`: 픽셀-3D점 매핑

#### **핵심 파라미터:**
- `scale_factor`: 5 (이미지 스케일)
- `dbscan_eps`: 1.5 (클러스터링 거리)
- `min_contour_area`: 60 (최소 컨투어 면적)

---

### 4. **morph.py - 형태학적 처리**

#### **원리:**
- OpenCV의 형태학적 연산을 사용하여 윤곽선을 부드럽게 처리
- 침식(Erosion)과 팽창(Dilation)을 조합하여 노이즈 제거

#### **작동 과정:**
1. **이진 이미지 로드**: `output/outline/*/binary.png`
2. **침식(Erosion)**:
   - `cv2.erode`로 경계를 축소
   - 노이즈 제거 효과
3. **팽창(Dilation)**:
   - `cv2.dilate`로 경계를 확장
   - 침식 후 팽창으로 경계 선명화
4. **열림-닫힘 연산**:
   - `cv2.MORPH_OPEN`: 침식 후 팽창 (노이즈 제거)
   - `cv2.MORPH_CLOSE`: 팽창 후 침식 (구멍 메우기)
5. **결과 저장**: `output/morph/*/morph_smoothed.png`

#### **핵심 파라미터:**
- `kernel_size`: 5 (형태학 연산 커널 크기)
- `iterations`: 2 (반복 횟수)

---

### 5. **pcd_to_mesh.py - 3D 메쉬 생성**

#### **원리:**
- 2D 윤곽선을 3D 메쉬로 변환
- 바닥, 천장, 벽을 각각 생성 후 병합
- Delaunay 삼각분할과 윤곽선 제약 조건을 사용

#### **작동 과정:**
1. **이미지 로드**: `output/morph/top_view/morph_smoothed.png`
2. **윤곽선 추출**:
   - `cv2.findContours`로 윤곽선 추출
   - `cv2.approxPolyDP`로 근사화
3. **좌표 변환**:
   - 픽셀 좌표를 3D 월드 좌표로 변환
   - `pixel_to_points.pkl` 파일을 사용한 정확한 매핑
4. **바닥 메쉬 생성**:
   - Poisson 메쉬 생성: `output/pcd/final_result_floor_plane.pcd`
   - 바닥 Y 레벨 감지
5. **천장/바닥 메쉬 생성**:
   - `create_constrained_flat_mesh` 함수 사용
   - Delaunay 삼각분할 + 윤곽선 제약 조건
   - 바닥: Y = floor_y_level + z_fighting_offset
   - 천장: Y = floor_y_level + fixed_height + z_fighting_offset
6. **벽 메쉬 생성**:
   - 윤곽선을 Y축 방향으로 압출(Extrusion)
   - 바닥에서 천장까지의 수직 벽 생성
7. **메쉬 병합**:
   - 바닥 + 천장 + 벽 메쉬 병합
   - 최종 결과: `output/pcd/extruded_building_mesh.ply`
8. **시각화**: Open3D를 사용한 3D 메쉬 시각화

#### **핵심 파라미터:**
- `fixed_height`: 10.0m (건물 높이)
- `z_fighting_offset`: 0.001m (Z-fighting 방지)
- `sphere_radius`: 1.0m (디버그 구 크기)

---

## 📊 출력 파일 설명

### **중간 결과:**
- `output/nonowall.pcd`: 벽이 제거된 포인트 클라우드
- `output/ransac/floor_plane.pcd`: 바닥 평면 포인트 클라우드
- `output/ransac/above_floor.pcd`: 바닥 위 영역 포인트 클라우드
- `output/outline/*/binary.png`: 2D 이진 투영 이미지
- `output/outline/*/pixel_to_points.pkl`: 픽셀-3D점 매핑 테이블
- `output/morph/*/morph_smoothed.png`: 형태학 처리된 이미지

### **최종 결과:**
- `output/pcd/extruded_building_mesh.ply`: 완전한 3D 건물 메쉬
- `output/mesh/merged_result_solid.ply`: 바닥과 병합된 메쉬
- `output/coordinate_mapping.json`: 좌표 매핑 정보

## 🔧 주요 알고리즘

### **RANSAC (Random Sample Consensus)**
- 노이즈가 있는 데이터에서 수학적 모델을 찾는 반복적 방법
- 평면 검출에 사용하여 바닥 평면을 식별

### **DBSCAN (Density-Based Spatial Clustering)**
- 밀도 기반 클러스터링 알고리즘
- 2D 투영에서 노이즈 제거 및 객체 분리에 사용

### **Delaunay 삼각분할**
- 점 집합을 삼각형으로 분할하는 알고리즘
- 2D 윤곽선을 3D 메쉬로 변환할 때 사용

### **형태학적 연산**
- 침식(Erosion): 경계 축소, 노이즈 제거
- 팽창(Dilation): 경계 확장, 구멍 메우기
- 열림(Opening): 침식 후 팽창
- 닫힘(Closing): 팽창 후 침식

## 📋 시스템 요구사항

### **Python 패키지:**
```
open3d>=0.17.0
numpy>=1.21.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

### **설치:**
```bash
pip install -r requirements.txt
```

## 🎯 사용 팁

1. **입력 파일**: `input/` 폴더에 PCD 파일을 넣어주세요
2. **메모리**: 대용량 포인트 클라우드의 경우 충분한 RAM 필요
3. **GPU**: Open3D 시각화는 GPU 가속 지원
4. **파라미터 조정**: 각 스크립트의 파라미터를 데이터에 맞게 조정 가능

## 🔍 문제 해결

### **일반적인 문제:**
- **메모리 부족**: 포인트 클라우드 다운샘플링 파라미터 조정
- **윤곽선 부정확**: DBSCAN 파라미터 조정
- **메쉬 품질**: Delaunay 삼각분할 파라미터 조정

### **Windows 호환성:**
- 모든 스크립트는 Windows에서 테스트됨
- 유니코드 이모지 제거로 인코딩 문제 해결
- cp949 인코딩 지원
