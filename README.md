<<<<<<< HEAD
# Individual-Research
=======
# RANSAC 평면 추정 및 다듬기

이 프로젝트는 PLY 포인트 클라우드 파일에서 RANSAC 알고리즘을 사용하여 평면을 추정하고 다듬는 Python 코드입니다.

## 기능

- PLY 파일 로드 및 기본 정보 출력
- RANSAC을 사용한 단일 평면 추정
- 여러 평면의 순차적 추출
- 결과 시각화 (Open3D 뷰어)
- 추출된 평면들을 개별 PLY 파일로 저장
- 평면 모델 정보를 텍스트 파일로 저장

## 설치

필요한 패키지들을 설치합니다:

```bash
pip install -r requirements.txt
```

## 사용법

1. `3BP_CS_model_cut_180.ply` 파일이 현재 디렉토리에 있는지 확인
2. 다음 명령어로 코드 실행:

```bash
python plane_refinement.py
```

## 주요 매개변수

- `distance_threshold`: 평면으로부터의 최대 거리 (기본값: 0.01)
- `ransac_n`: RANSAC에 사용할 포인트 수 (기본값: 3)
- `num_iterations`: RANSAC 반복 횟수 (기본값: 1000)
- `num_planes`: 추출할 평면의 개수 (기본값: 5)

## 출력

- `refined_planes/` 폴더에 추출된 평면들이 저장됩니다:
  - `plane_1.ply`, `plane_2.ply`, ...: 각 평면의 포인트 클라우드
  - `plane_1_model.txt`, `plane_2_model.txt`, ...: 각 평면의 수학적 모델 정보

## 평면 모델

평면은 `ax + by + cz + d = 0` 형태로 표현되며, 각 파일에는 다음 정보가 포함됩니다:
- a, b, c: 평면의 법선 벡터 성분
- d: 원점으로부터의 거리
- 포인트 개수

## 시각화

코드 실행 시 Open3D 뷰어가 열리며 다음 색상으로 표시됩니다:
- 회색: 원본 포인트 클라우드
- 빨간색, 초록색, 파란색, 노란색, 마젠타, 시안: 각 평면
- 검은색: 평면에 속하지 않는 나머지 포인트들 
>>>>>>> f9febe62 (first commit)
