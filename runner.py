import subprocess

pipeline = [
    ("nonowall.py", "벽 제거 (nonowall.py)"),
    ("ransac.py", "RANSAC 평면 분리 (ransac.py)"),
    ("outline.py", "윤곽선 추출 (outline.py)"),
    ("morph.py", "모폴로지 처리 (morph.py)"),
    ("morph_to_pcd.py", "2D→3D 변환 및 병합 (morph_to_pcd.py)")
]

for idx, (filename, desc) in enumerate(pipeline):
    print(f"\n==== {desc} 실행 ====")
    result = subprocess.run(["python", filename])
    if result.returncode != 0:
        print(f"{filename} 실행 중 오류가 발생했습니다. 파이프라인을 중단합니다.")
        break
    if idx < len(pipeline) - 1:
        answer = input("다음 파일도 실행할까요? (y/n): ").strip().lower()
        if answer != "y":
            print("파이프라인 실행을 중단합니다.")
            break
print("\n파이프라인 실행이 종료되었습니다.") 