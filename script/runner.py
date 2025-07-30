"""
전체 3D 메쉬 생성 파이프라인을 순차적으로 실행하는 스크립트
순서: nonowall -> ransac -> outline -> morph -> pcd_to_mesh
"""
import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(command, step_name):
    """명령어를 실행하고 결과를 반환합니다."""
    print(f"\n{'='*60}")
    print(f"[START] {step_name}")
    print(f"[CMD] 실행 명령어: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, encoding='cp949', errors='replace')
        
        print("[OUTPUT] 출력:")
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("[ERROR] 경고/에러:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"[FAIL] {step_name} 실패! (반환코드: {result.returncode})")
            return False
        else:
            print(f"[SUCCESS] {step_name} 성공!")
            return True
            
    except Exception as e:
        print(f"[ERROR] {step_name} 실행 중 오류: {e}")
        return False

def check_file_exists(file_path, description):
    """파일이 존재하는지 확인합니다."""
    if os.path.exists(file_path):
        print(f"[SUCCESS] {description} 확인됨: {file_path}")
        return True
    else:
        print(f"[ERROR] {description} 없음: {file_path}")
        return False

def create_directories():
    """필요한 출력 디렉토리들을 생성합니다."""
    directories = [
        "../output",
        "../output/pcd",
        "../output/mesh",
        "../output/morph",
        "../output/morph/floor_plane",
        "../output/morph/above_floor",
        "../output/outline",
        "../output/outline/floor_plane",
        "../output/outline/above_floor",
        "../output/ransac",
        "../output/ransac/floor_plane_from_original"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[INFO] 디렉토리 생성/확인: {directory}")

def main():
    print("[START] 3D 메쉬 생성 전체 파이프라인 시작!")
    print("[INFO] 순서: nonowall -> ransac -> outline -> morph -> pcd_to_mesh")
    
    # 1. 필요한 디렉토리 생성
    print("\n[INFO] 출력 디렉토리 생성 중...")
    create_directories()
    
    # 2. 입력 파일 확인
    print("\n[INFO] 입력 파일 확인 중...")
    required_files = [
        ("../input/3BP_CS_model_Cloud.pcd", "원본 PCD 파일"),
        ("../input/3BP_CS_model.ply", "원본 PLY 파일")
    ]
    
    missing_files = []
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n[ERROR] 필수 입력 파일이 없습니다:")
        for file_path in missing_files:
            print(f"   • {file_path}")
        print("\n   input/ 폴더에 파일을 넣어주세요.")
        return False
    
    # 3. nonowall 실행 (벽 제거)
    if not run_command("python nonowall.py", "벽 제거 (nonowall)"):
        return False
    
    if not check_file_exists("../output/nonowall.pcd", "벽 제거된 PCD"):
        return False
    
    # 4. 잠시 대기 (파일 시스템 동기화)
    time.sleep(1)
    
    # 5. RANSAC으로 바닥 평면 분리
    if not run_command("python ransac.py", "RANSAC 바닥 평면 분리"):
        return False
    
    # 6. RANSAC 결과 확인
    ransac_results = [
        ("../output/ransac/floor_plane.pcd", "바닥 평면 PCD"),
        ("../output/ransac/above_floor.pcd", "바닥 위 PCD")
    ]
    
    for file_path, description in ransac_results:
        if not check_file_exists(file_path, description):
            return False
    
    # 7. 윤곽선 추출
    if not run_command("python outline.py", "2D 윤곽선 추출"):
        return False
    
    # 8. 윤곽선 결과 확인
    outline_results = [
        ("../output/outline/above_floor/binary.png", "상단 뷰 이진 이미지"),
        ("../output/outline/above_floor/pixel_to_points.pkl", "상단 뷰 픽셀-점 매핑"),
        ("../output/outline/floor_plane/binary.png", "바닥 평면 이진 이미지"),
        ("../output/outline/floor_plane/pixel_to_points.pkl", "바닥 평면 픽셀-점 매핑")
    ]
    
    for file_path, description in outline_results:
        if not check_file_exists(file_path, description):
            return False
    
    # 9. morph 실행 (형태학적 처리)
    if not run_command("python morph.py", "형태학적 처리 (morph)"):
        return False
    
    # 10. morph 결과 확인
    morph_results = [
        ("../output/morph/above_floor/morph_smoothed.png", "상단 뷰 형태학 처리 결과"),
        ("../output/morph/floor_plane/morph_smoothed.png", "바닥 평면 형태학 처리 결과")
    ]
    
    for file_path, description in morph_results:
        if not check_file_exists(file_path, description):
            return False
    
    # 11. PCD에서 메쉬 생성
    if not run_command("python pcd_to_mesh.py", "PCD에서 3D 메쉬 생성"):
        return False
    
    # 12. 최종 결과 확인
    print("\n[SUCCESS] 파이프라인 완료! 최종 결과 확인 중...")
    
    final_results = [
        ("../output/pcd/extruded_building_mesh.ply", "최종 건물 메쉬"),
        ("../output/mesh/merged_result_solid.ply", "병합된 메쉬"),
        ("../output/coordinate_mapping.json", "좌표 매핑 정보")
    ]
    
    success_count = 0
    for file_path, description in final_results:
        if check_file_exists(file_path, description):
            success_count += 1
    
    print(f"\n[INFO] 최종 결과 요약:")
    print(f"   • 성공한 결과 파일: {success_count}/{len(final_results)}")
    
    if success_count == len(final_results):
        print("\n[SUCCESS] 모든 단계가 성공적으로 완료되었습니다!")
        print("\n[INFO] 생성된 주요 파일들:")
        print("  • ../output/nonowall.pcd - 벽 제거된 PCD")
        print("  • ../output/ransac/floor_plane.pcd - 바닥 평면 PCD")
        print("  • ../output/ransac/above_floor.pcd - 바닥 위 PCD")
        print("  • ../output/morph/above_floor/morph_smoothed.png - 상단 뷰 형태학 처리")
        print("  • ../output/pcd/extruded_building_mesh.ply - 최종 건물 메쉬")
        print("  • ../output/mesh/merged_result_solid.ply - 바닥과 병합된 메쉬")
        print("  • ../output/coordinate_mapping.json - 좌표 매핑 정보")
    else:
        print(f"\n[WARNING] 일부 결과 파일이 생성되지 않았습니다. ({success_count}/{len(final_results)})")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 