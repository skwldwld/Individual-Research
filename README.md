# 3D Point Cloud Processing Pipeline

이 프로젝트는 3D 포인트 클라우드 데이터를 처리하여 메시로 변환하는 파이프라인을 제공합니다.

## 파이프라인 순서

1. **Remove** - 벽과 천장 제거
2. **RANSAC** - 평면 검출 및 바닥 분리
3. **Outline** - 2D 윤곽선 추출
4. **Morph** - 이미지 모폴로지 처리
5. **Morph to PCD** - 이미지를 3D 포인트로 변환
6. **PCD to Mesh** - 포인트 클라우드를 메시로 변환

## GUI 사용법

### 설치

```bash
pip install -r requirements.txt
```

### 실행

```bash
python script/gui_runner.py
```

### GUI 기능

- **입력 파일**: PCD, PLY, OBJ 형식의 3D 파일 선택
- **파라미터 설정**: 각 단계별 파라미터를 탭으로 구분하여 설정
- **개별 실행**: 각 단계를 개별적으로 실행 가능
- **전체 파이프라인**: 모든 단계를 순차적으로 자동 실행
- **실시간 로그**: 실행 과정을 실시간으로 확인
- **출력 디렉토리**: 결과물 저장 위치 설정

### 파라미터 설명

#### Remove
- **벽 두께**: 제거할 벽의 두께 (미터)
- **천장 두께**: 제거할 천장의 두께 (미터)
- **복셀 크기**: 다운샘플링을 위한 복셀 크기
- **높이 축**: 천장 판단에 사용할 축 (x, y, z)

#### RANSAC
- **최대 평면 수**: 검출할 최대 평면 수
- **거리 임계값**: 평면 검출을 위한 거리 임계값
- **RANSAC N**: RANSAC 알고리즘의 N 값
- **반복 횟수**: RANSAC 반복 횟수
- **최소 인라이어 비율**: 평면으로 인정할 최소 포인트 비율
- **수직 오프셋**: 바닥 컷 높이 조정
- **위쪽 축**: 위쪽 방향 축 설정
- **자동 기울기 보정**: 자동으로 기울기 보정 수행
- **시각화**: 결과 시각화 여부

#### Outline
- **스케일 팩터**: 2D 이미지 변환 시 스케일
- **윤곽선 최소 크기**: 윤곽선 검출 시 최소 크기
- **DBSCAN eps**: 클러스터링을 위한 거리 임계값
- **DBSCAN min_samples**: 클러스터 최소 샘플 수

#### Morph
- **커널 크기**: 모폴로지 연산을 위한 커널 크기

#### Morph to PCD
- **복셀 크기**: 포인트 클라우드 다운샘플링 크기
- **최종 병합**: 모든 결과를 하나로 병합 여부

#### PCD to Mesh
- **메시 품질**: 생성할 메시의 품질 (low, medium, high)
- **시각화**: 결과 메시 시각화 여부

## 명령줄 사용법

각 스크립트는 개별적으로도 실행할 수 있습니다:

```bash
# Remove
python script/remove.py --input input.pcd --out output_dir

# RANSAC
python script/ransac.py --input nonowall.pcd --out ransac_output

# Outline
python script/outline.py

# Morph
python script/morph.py

# Morph to PCD
python script/morph_to_pcd.py

# PCD to Mesh
python script/pcd_to_mesh.py
```

## 출력 구조

```
output/
├── nonowall.pcd          # 벽과 천장이 제거된 포인트 클라우드
├── removed_walls.pcd     # 제거된 벽 포인트
├── ransac/               # RANSAC 결과
│   ├── floor_plane.pcd
│   ├── expanded_floor_plane.pcd
│   └── topview.pcd
├── outline/              # 2D 윤곽선 결과
│   ├── floor/
│   ├── topview/
│   └── sideview/
├── morph/                # 모폴로지 처리 결과
├── pcd/                  # 이미지에서 변환된 포인트 클라우드
└── mesh/                 # 최종 메시 파일
```

## 주의사항

- 입력 파일은 바이너리 형식의 PCD 파일을 권장합니다
- 각 단계는 이전 단계의 출력을 입력으로 사용합니다
- GUI에서 전체 파이프라인을 실행하면 자동으로 순차 실행됩니다
- 개별 단계 실행 시 이전 단계가 완료되었는지 확인하세요
