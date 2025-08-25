import cv2
import os

def morph_process(binary_img_path, output_dir, kernel_size=7, enable_open=True):
    os.makedirs(output_dir, exist_ok=True)
    output_morph_path = os.path.join(output_dir, "morph_smoothed.png")

    img = cv2.imread(binary_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"이미지 로드 실패: {binary_img_path}")
        return False

    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    k_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    tmp = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k_h, iterations=1)
    morph = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, k_v, iterations=1)
    
    # enable_open이 True일 때만 MORPH_OPEN 연산 적용
    if enable_open:
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, k_cross, iterations=1)
        
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
    import argparse
    
    parser = argparse.ArgumentParser(description="모폴로지 처리")
    parser.add_argument("--kernel-size", type=int, default=5, help="커널 크기 (기본값: 5)")
    args = parser.parse_args()
    
    # wall과 sideview material에 대해 MORPH_OPEN을 비활성화하는 로직 추가
    outline_targets = [
        ("floor", "../output/outline/floor/binary.png", True),
        ("topview_material", "../output/outline/topview/material/binary.png", True),
        ("sideview_material", "../output/outline/sideview/material/binary.png", False), # MORPH_OPEN 비활성화
        ("topview_wall", "../output/outline/topview/wall/binary.png", False), # MORPH_OPEN 비활성화
        ("sideview_wall", "../output/outline/sideview/wall/binary.png", True),
    ]
    for name, binary_img_path, enable_open in outline_targets:
        out_dir = make_output_dir(name)
        morph_process(binary_img_path, out_dir, kernel_size=args.kernel_size, enable_open=enable_open)

if __name__ == "__main__":
    main()
