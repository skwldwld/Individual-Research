import os
import sys
import threading
import traceback
import importlib
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

# ------------------------------
# 프로젝트 루트/스크립트 경로 설정
# ------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = THIS_DIR  # 본 파일과 같은 폴더에 스크립트들이 있다고 가정
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ------------------------------
# 파이프라인 단계 정의 (모듈명, 호출함수, 표시명)
#  - main()을 인자 없이 호출 → 각 파일 안에 하드코딩된 기본값 사용
#  - 필요 시 함수명을 바꾸거나, 인자 지원하려면 여기서 조정
# ------------------------------
PIPELINE = [
    ("remove",        "main", "1) Remove (벽/천장 분리 + nonowall 저장)"),
    ("ransac",        "__main__", "2) RANSAC (바닥/상부 분리)"),  # __main__은 모듈 실행 가드가 있는 경우를 위해 아래에서 처리
    ("outline",       "main", "3) Outline (윤곽선/매핑 저장)"),
    ("morph",         "main", "4) Morph (모폴로지 보정)"),
    ("morph_to_pcd",  "main", "5) Morph→PCD (wall/material/floor PCD 생성)"),
    ("pcd_to_mesh",   "main", "6) PCD→Mesh (바닥 Poisson + 벽/재질 Extrude)")
]

# ------------------------------
# 유틸: 로그 출력
# ------------------------------
class Logger:
    def __init__(self, widget: ScrolledText):
        self.widget = widget
        self.lock = threading.Lock()

    def write(self, msg: str):
        with self.lock:
            self.widget.insert(tk.END, msg)
            self.widget.see(tk.END)
            self.widget.update_idletasks()

    def println(self, msg: str):
        self.write(msg + "\n")

# ------------------------------
# 단계 실행기
# ------------------------------
def run_module(module_name: str, func_name: str, logger: Logger):
    """모듈을 importlib로 로드 후 함수 호출. func_name == '__main__' 인 경우 모듈의 엔트리포인트 수행 시도."""
    try:
        logger.println(f"[LOAD] {module_name}.py")
        mod = importlib.import_module(module_name)
        # 코드 최신 반영
        importlib.reload(mod)

        # 1) '__main__' 처리: 파일 하단의 if __name__ == "__main__": 블록을 그대로 실행하고 싶을 때
        if func_name == "__main__":
            # 일부 스크립트는 __main__ 블록에서 parse_args()를 호출하므로, argv를 비워 기본값만 사용하도록 함
            old_argv = sys.argv[:]
            try:
                sys.argv = [module_name]
                if hasattr(mod, "__name__") and mod.__name__:
                    # 모듈을 스크립트처럼 실행하는 가장 간단한 방법: runpy를 써도 되지만, 여기서는 main 함수를 우선 시도
                    # ransac.py가 if __name__ == "__main__": 아래에서 parse_args() 후 ransac_main(...)을 호출하는 구조라면
                    # 아래와 같은 접근으로 main 진입을 유도한다.
                    if hasattr(mod, "main"):
                        logger.println("[CALL] main() (기본값 사용)")
                        mod.main()  # 인자 없이 호출 → 하드코딩/기본값
                    else:
                        # main이 없다면, 공개된 엔트리 함수를 유추해서 호출
                        # 예: ransac_main 같은 함수가 있으면 그걸 기본 인자 없이 호출
                        for cand in [
                            "main", "ransac_main", "run", "entry", "cli"
                        ]:
                            if hasattr(mod, cand):
                                logger.println(f"[CALL] {cand}() (기본값 사용)")
                                getattr(mod, cand)()
                                break
                        else:
                            logger.println("[WARN] 실행 가능한 엔트리 함수를 찾지 못했습니다. (스킵)")
                else:
                    logger.println("[WARN] 모듈 이름 확인 실패")
            finally:
                sys.argv = old_argv
            return

        # 2) 명시 함수 호출
        if not hasattr(mod, func_name):
            # 대안 함수 이름 시도
            for cand in [func_name, "main", "run", "entry", "cli"]:
                if hasattr(mod, cand):
                    func_name = cand
                    break
            else:
                logger.println(f"[WARN] {module_name}.py : 실행 함수 '{func_name}'를 찾지 못했습니다. (스킵)")
                return

        func = getattr(mod, func_name)
        logger.println(f"[CALL] {module_name}.{func_name}() (인자 없이 호출 → 하드코딩/기본값 사용)")
        ret = func()  # 인자 없이 호출
        if ret is not None:
            logger.println(f"[RET] {module_name}.{func_name} -> {ret}")
    except SystemExit as e:
        # argparse가 종료시킨 경우 등
        logger.println(f"[INFO] {module_name} 종료 (SystemExit: {e})")
    except Exception:
        logger.println("[ERROR] 실행 중 예외 발생:\n" + traceback.format_exc())


def run_selected(steps, logger: Logger, run_button: ttk.Button):
    def worker():
        try:
            run_button.config(state=tk.DISABLED)
            for module_name, func_name, label in steps:
                logger.println("\n" + "="*80)
                logger.println(f"[STEP] {label}")
                logger.println("="*80)
                run_module(module_name, func_name, logger)
            logger.println("\n[ALL DONE] 파이프라인 실행 완료")
        finally:
            run_button.config(state=tk.NORMAL)
    t = threading.Thread(target=worker, daemon=True)
    t.start()

# ------------------------------
# GUI
# ------------------------------
class App:
    def __init__(self, root):
        root.title("3D Pipeline Launcher (Hardcoded-Defaults)")
        root.geometry("860x600")

        # 상단 프레임: 단계 선택
        top = ttk.LabelFrame(root, text="실행 단계 선택 (기본값=전부)")
        top.pack(fill=tk.X, padx=10, pady=10)

        self.vars = []
        for i, (module_name, func_name, label) in enumerate(PIPELINE):
            v = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(top, text=label, variable=v)
            cb.grid(row=i//2, column=i%2, sticky=tk.W, padx=8, pady=4)
            self.vars.append((v, module_name, func_name, label))

        # 실행 버튼
        self.run_btn = ttk.Button(root, text="선택 단계 실행", command=self.on_run)
        self.run_btn.pack(padx=10, pady=(0,10))

        # 로그 영역
        self.log = ScrolledText(root, height=24)
        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        self.logger = Logger(self.log)

        # 경로 안내
        self.logger.println("[INFO] 실행 기준 폴더: " + SCRIPT_DIR)
        self.logger.println("[INFO] 각 스크립트의 하드코딩 경로/기본값을 그대로 사용합니다.")
        self.logger.println("[TIP] 경로가 상대경로인 경우, 이 GUI를 스크립트들과 같은 폴더에서 실행하세요.")

    def on_run(self):
        selected = [(m, f, l) for (v, m, f, l) in self.vars if v.get()]
        if not selected:
            self.logger.println("[WARN] 선택된 단계가 없습니다.")
            return
        run_selected(selected, self.logger, self.run_btn)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
