import cv2
import os

def morph_process(binary_img_path, output_dir, kernel_size=5):
    """
    바이너리 이미지를 받아 침식, 팽창, 모폴로지 연산을 수행하고 결과를 저장합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_eroded_path = os.path.join(output_dir, "eroded.png")
    output_dilated_path = os.path.join(output_dir, "dilated.png")
    output_morph_path = os.path.join(output_dir, "morph_smoothed.png")

    img = cv2.imread(binary_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"이미지 로드 실패: {binary_img_path}")
        return False

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # 1. 침식 (Erosion)
    eroded = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite(output_eroded_path, eroded)
    print(f"침식 결과 저장: {output_eroded_path}")

    # 2. 팽창 (Dilation)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    cv2.imwrite(output_dilated_path, dilated)
    print(f"팽창 결과 저장: {output_dilated_path}")

    # 3. 열림-닫힘(Opening-Closing)
    morph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imwrite(output_morph_path, morph)
    print(f"모폴로지(열림-닫힘) 결과 저장: {output_morph_path}")
    return True

def main():
    outline_targets = [
        ("above_floor", "../output/outline/above_floor/binary.png"),
        ("floor_plane", "../output/outline/floor_plane/binary.png"),
    ]
    for name, binary_img_path in outline_targets:
        output_dir = os.path.join("../output", "morph", name)
        morph_process(binary_img_path, output_dir)

if __name__ == "__main__":
    main() 