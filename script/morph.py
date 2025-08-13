import cv2
import os

def morph_process(binary_img_path, output_dir, kernel_size=7):
    os.makedirs(output_dir, exist_ok=True)
    output_eroded_path = os.path.join(output_dir, "eroded.png")
    output_dilated_path = os.path.join(output_dir, "dilated.png")
    output_morph_path = os.path.join(output_dir, "morph_smoothed.png")

    img = cv2.imread(binary_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"이미지 로드 실패: {binary_img_path}")
        return False

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # 커널 생성 부분 교체
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))  # 가로로 틈 메움
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))  # 세로로 틈 메움
    k_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 가시 제거용 약한 커널


    # eroded = cv2.erode(img, kernel, iterations=1)
    # cv2.imwrite(output_eroded_path, eroded)
    # print(f"침식 결과 저장: {output_eroded_path}")

    # dilated = cv2.dilate(eroded, kernel, iterations=1)
    # cv2.imwrite(output_dilated_path, dilated)
    # print(f"팽창 결과 저장: {output_dilated_path}")

    # 기존 CLOSE/OPEN 삭제하고 아래로 교체
    tmp   = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k_h, iterations=1)
    morph = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, k_v, iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN,  k_cross, iterations=1)  # 너무 지워지면 이 줄은 주석
    _, morph = cv2.threshold(morph, 127, 255, cv2.THRESH_BINARY)

    cv2.imwrite(output_morph_path, morph)
    print(f"morphology 결과 저장: {output_morph_path}")
    return True

def make_output_dir(name: str) -> str:
    """
    'topview_material' -> '../output/morph/topview/material'
    'sideview_wall'    -> '../output/morph/sideview/wall'
    'floor_plane'      -> '../output/morph/floor'
    """
    base = os.path.join("..", "output", "morph")
    if "_" in name:
        view, category = name.split("_", 1)
        return os.path.join(base, view, category)
    return os.path.join(base, name)

def main():
    outline_targets = [
        ("floor", "../output/outline/floor/binary.png"),
        ("topview_material", "../output/outline/topview/material/binary.png"),
        ("sideview_material", "../output/outline/sideview/material/binary.png"),
        ("topview_wall", "../output/outline/topview/wall/binary.png"),
        ("sideview_wall", "../output/outline/sideview/wall/binary.png"),
    ]
    for name, binary_img_path in outline_targets:
        out_dir = make_output_dir(name)
        morph_process(binary_img_path, out_dir, kernel_size=5)

if __name__ == "__main__":
    main()
