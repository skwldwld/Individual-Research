import cv2
import os

# 입력 이미지 경로 (outline.py에서 생성된 결과)
input_dir = "output"
binary_img_path = os.path.join(input_dir, "outline/binary.png")
contour_img_path = os.path.join(input_dir, "outline/contours.png")

# 결과 저장 경로
output_eroded_path = os.path.join(input_dir, "morph/eroded.png")
output_dilated_path = os.path.join(input_dir, "morph/dilated.png")
output_morph_path = os.path.join(input_dir, "morph/morph_smoothed.png")

# 이미지 불러오기 (binary.png 기준)
img = cv2.imread(binary_img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"이미지 로드 실패: {binary_img_path}")
    exit(1)

# 커널 설정 (5x5 사각형)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 1. 침식 (Erosion) - 작은 돌출 제거
eroded = cv2.erode(img, kernel, iterations=1)
cv2.imwrite(output_eroded_path, eroded)
print(f"침식 결과 저장: {output_eroded_path}")

# 2. 팽창 (Dilation) - 움푹 들어간 부분 메움
dilated = cv2.dilate(eroded, kernel, iterations=1)
cv2.imwrite(output_dilated_path, dilated)
print(f"팽창 결과 저장: {output_dilated_path}")

# 3. 열림-닫힘(Opening-Closing) 연산으로 부드럽게
# (침식 후 팽창: 작은 노이즈 제거, 팽창 후 침식: 구멍 메움)
morph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=2)
cv2.imwrite(output_morph_path, morph)
print(f"모폴로지(열림-닫힘) 결과 저장: {output_morph_path}")
