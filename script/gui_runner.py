import sys
import os
import re
import json
import subprocess
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QTextEdit, QFileDialog, QGroupBox, QSpinBox,
                             QDoubleSpinBox, QComboBox, QCheckBox, QMessageBox,
                             QTabWidget)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ------------------------
# Worker
# ------------------------
class ScriptRunner(QThread):
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, script_py_path, args, workdir):
        super().__init__()
        self.script_py_path = script_py_path
        self.args = args
        self.workdir = workdir

    def run(self):
        try:
            cmd = [sys.executable, "-u", self.script_py_path] + self.args
            self.output_signal.emit(f"$ {' '.join(cmd)}\n")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=self.workdir
            )
            for line in iter(proc.stdout.readline, ''):
                if line:
                    self.output_signal.emit(line.rstrip())
            proc.wait()
            ok = (proc.returncode == 0)
            self.finished_signal.emit(ok, f"{os.path.basename(self.script_py_path)} code={proc.returncode}")
        except Exception as e:
            self.finished_signal.emit(False, f"ERR {os.path.basename(self.script_py_path)}: {e}")

# ------------------------
# GUI
# ------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Point Cloud Processing GUI")
        self.setGeometry(100, 100, 1200, 800)

        central = QWidget(); self.setCentralWidget(central)
        main = QHBoxLayout(central)

        # 기준 경로(레포 루트)
        self.repo_dir = os.path.dirname(os.path.abspath(__file__))

        # Left: params
        left = QVBoxLayout()

        # 입력
        grp_in = QGroupBox("입력 파일"); vin = QVBoxLayout(grp_in)
        self.input_path_edit = QLineEdit(); self.input_path_edit.setPlaceholderText("PCD/PLY 선택")
        btn_browse_in = QPushButton("파일 선택"); btn_browse_in.clicked.connect(self.browse_input_file)
        vin.addWidget(QLabel("PCD 파일:")); vin.addWidget(self.input_path_edit); vin.addWidget(btn_browse_in)
        left.addWidget(grp_in)

        # Tabs
        self.tab = QTabWidget(); left.addWidget(self.tab)

        # Remove 탭
        self.tab.addTab(self._tab_remove(), "Remove")
        # RANSAC 탭
        self.tab.addTab(self._tab_ransac(), "RANSAC")
        # Outline 탭
        self.tab.addTab(self._tab_outline(), "Outline")
        # Morph 탭
        self.tab.addTab(self._tab_morph(), "Morph")
        # Morph->PCD 탭
        self.tab.addTab(self._tab_morph_to_pcd(), "Morph to PCD")
        # Mesh 탭
        self.tab.addTab(self._tab_mesh(), "PCD to Mesh")

        # 실행
        grp_run = QGroupBox("실행"); vrun = QVBoxLayout(grp_run)
        self.run_all_btn = QPushButton("전체 파이프라인 실행")
        self.run_all_btn.setStyleSheet("QPushButton { background:#4CAF50; color:#fff; font-weight:bold; padding:10px; }")
        self.run_all_btn.clicked.connect(self.run_all_pipeline)
        vrun.addWidget(self.run_all_btn)

        # 개별 실행
        btns = [
            ("벽, 천장 제거", self.run_remove),
            ("바닥 추정", self.run_ransac),
            ("윤곽선 추출", self.run_outline),
            ("모폴로지 처리", self.run_morph),
            ("PCD 생성", self.run_morph_to_pcd),
            ("Mesh 생성", self.run_pcd_to_mesh),
        ]
        for t, fn in btns:
            b = QPushButton(t); b.clicked.connect(fn); vrun.addWidget(b)
        left.addWidget(grp_run)

        # Right: 출력 경로 + 로그
        right = QVBoxLayout()
        grp_out = QGroupBox("출력 디렉토리"); vout = QVBoxLayout(grp_out)
        self.output_path_edit = QLineEdit("../output")
        btn_browse_out = QPushButton("폴더 선택"); btn_browse_out.clicked.connect(self.browse_output_dir)
        vout.addWidget(QLabel("출력 폴더:")); vout.addWidget(self.output_path_edit); vout.addWidget(btn_browse_out)
        right.addWidget(grp_out)

        grp_log = QGroupBox("실행 로그"); vlog = QVBoxLayout(grp_log)
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True); self.log_text.setFont(QFont("Consolas", 9))
        vlog.addWidget(self.log_text)
        btn_clear = QPushButton("로그 클리어"); btn_clear.clicked.connect(self.log_text.clear)
        vlog.addWidget(btn_clear)
        right.addWidget(grp_log)

        main.addLayout(left, 0); main.addLayout(right, 1)

        # runner/queue
        self.current_runner = None
        self.pipeline_queue = []  # [(script_abs, args, workdir), ...]

        # 1) 스크립트 하드코딩 기본값 로드
        self.defaults = self._load_defaults_from_scripts()
        # 2) 위젯에 기본값 반영
        self._apply_defaults_to_widgets()

    # ------------------------
    # Defaults loader
    # ------------------------
    def _read_text(self, fname):
        try:
            with open(os.path.join(self.repo_dir, fname), "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def _rx(self, pattern, text, flags=re.MULTILINE|re.DOTALL):
        m = re.search(pattern, text, flags)
        return m.group(1) if m else None

    def _load_defaults_from_scripts(self):
        d = {}

        # remove_rect.py (직사각 전제)
        txt = self._read_text("remove_rect.py")
        if txt:
            v = self._rx(r'add_argument\([^\n]*"--voxel"[^\n]*default\s*=\s*([0-9.]+)', txt)
            d["remove_rect.voxel"] = float(v) if v else 0.03
            ht = self._rx(r'add_argument\([^\n]*"--height-axis"[^\n]*default\s*=\s*["\']([xyz])["\']', txt)
            d["remove_rect.up"] = ht if ht else "z"
            wa = self._rx(r'add_argument\([^\n]*"--wall-axis"[^\n]*default\s*=\s*["\']([xyz])["\']', txt)
            d["remove_rect.wall_axis"] = wa if wa else "z"
            ws = self._rx(r'add_argument\([^\n]*"--wall-sides"[^\n]*default\s*=\s*["\'](both|min|max)["\']', txt)
            d["remove_rect.wall_sides"] = ws if ws else "both"

        # remove.py (강화 일반형)
        txt = self._read_text("remove.py")
        if txt:
            pv = self._rx(r'add_argument\([^\n]*"--post-voxel"[^\n]*default\s*=\s*([0-9.]+)', txt)
            d["remove.post_voxel"] = float(pv) if pv else 0.03
            ht = self._rx(r'add_argument\([^\n]*"--height-axis"[^\n]*default\s*=\s*["\']([xyz])["\']', txt)
            d["remove.up"] = ht if ht else "z"

        # ransac.py (argparse defaults)
        txt = self._read_text("ransac.py")
        if txt:
            def _argdef(name, default_fallback):
                m = re.search(rf'add_argument\([^\n]*"{name}"[^\n]*default\s*=\s*([^\s,)]+)', txt)
                if not m: m = re.search(rf"add_argument\([^\n]*'{name}'[^\n]*default\s*=\s*([^\s,)]+)", txt)
                if m:
                    raw = m.group(1).strip().strip("'\"")
                    try:
                        if raw.lower() in ["x","y","z"]:
                            return raw
                        if re.match(r"^-?\d+$", raw): return int(raw)
                        return float(raw)
                    except Exception:
                        return raw
                return default_fallback

            d["ransac.max_planes"] = _argdef("--max-planes", 40)
            d["ransac.dist"]       = _argdef("--dist", 0.035)
            d["ransac.ransac_n"]   = _argdef("--ransac-n", 3)
            d["ransac.iters"]      = _argdef("--iters", 1000)
            d["ransac.min_ratio"]  = _argdef("--min-ratio", 0.01)
            d["ransac.offset"]     = _argdef("--offset", 1.0)
            d["ransac.up_axis"]    = _argdef("--up-axis", "y")

        # outline.py (argparse defaults)
        txt = self._read_text("outline.py")
        if txt:
            def _argdef_outline(name, fallback):
                m = re.search(rf'add_argument\([^\n]*"{name}"[^\n]*default\s*=\s*([^\s,)]+)', txt)
                if not m: m = re.search(rf"add_argument\([^\n]*'{name}'[^\n]*default\s*=\s*([^\s,)]+)", txt)
                if m:
                    raw = m.group(1).strip().strip("'\"")
                    try:
                        if re.match(r"^-?\d+$", raw): return int(raw)
                        return float(raw)
                    except Exception:
                        return raw
                return fallback

            d["outline.scale_factor"] = _argdef_outline("--scale-factor", 5.0)
            d["outline.contour_min"]  = int(_argdef_outline("--contour-min", 10))
            d["outline.dbscan_eps"]   = _argdef_outline("--dbscan-eps", 0.03)
            d["outline.dbscan_min"]   = int(_argdef_outline("--dbscan-min", 30))

        # morph.py (argparse defaults)
        txt = self._read_text("morph.py")
        if txt:
            m = re.search(r'add_argument\([^\n]*"--kernel-size"[^\n]*default\s*=\s*([0-9]+)', txt)
            d["morph.kernel_size"] = int(m.group(1)) if m else 5

        # morph_to_pcd.py (상수 VOXEL_SIZE, MERGE_AT_END)
        txt = self._read_text("morph_to_pcd.py")
        if txt:
            v = self._rx(r"VOXEL_SIZE\s*=\s*([0-9.]+)", txt)
            mer = self._rx(r"MERGE_AT_END\s*=\s*(True|False)", txt)
            d["m2p.voxel"] = float(v) if v else 0.01
            d["m2p.merge"] = True if (mer == "True") else False

        # pcd_to_mesh.py (main(visualize=True))
        txt = self._read_text("pcd_to_mesh.py")
        if txt:
            vis = self._rx(r"def\s+main\(\s*visualize\s*=\s*(True|False)\s*\)", txt)
            d["mesh.visualize"] = True if (vis == "True") else True  # fallback True

        return d

    def _apply_defaults_to_widgets(self):
        # Remove
        self.wall_thickness_spin.setValue(float(self.defaults.get("remove_rect.wall_thickness", 5.0)) if "remove_rect.wall_thickness" in self.defaults else self.wall_thickness_spin.value())
        self.ceiling_thickness_spin.setValue(float(self.defaults.get("remove_rect.ceiling_thickness", 5.0)) if "remove_rect.ceiling_thickness" in self.defaults else self.ceiling_thickness_spin.value())
        self.voxel_size_spin.setValue(float(self.defaults.get("remove.post_voxel", self.defaults.get("remove_rect.voxel", self.voxel_size_spin.value()))))
        axis = str(self.defaults.get("remove.up", self.defaults.get("remove_rect.up", self.height_axis_combo.currentText())))
        idx = max(0, self.height_axis_combo.findText(axis))
        self.height_axis_combo.setCurrentIndex(idx)

        # 신규: 벽 축/벽 쪽
        wall_axis = str(self.defaults.get("remove_rect.wall_axis", "z"))
        idx = max(0, self.wall_axis_combo.findText(wall_axis))
        self.wall_axis_combo.setCurrentIndex(idx)

        wall_sides = str(self.defaults.get("remove_rect.wall_sides", "both"))
        # 매핑: both→0, min→1, max→2
        sides_map = {"both":0, "min":1, "max":2}
        self.wall_sides_combo.setCurrentIndex(sides_map.get(wall_sides, 0))

        # RANSAC
        self.max_planes_spin.setValue(int(self.defaults.get("ransac.max_planes", self.max_planes_spin.value())))
        self.dist_threshold_spin.setValue(float(self.defaults.get("ransac.dist", self.dist_threshold_spin.value())))
        self.ransac_n_spin.setValue(int(self.defaults.get("ransac.ransac_n", self.ransac_n_spin.value())))
        self.iterations_spin.setValue(int(self.defaults.get("ransac.iters", self.iterations_spin.value())))
        self.min_ratio_spin.setValue(float(self.defaults.get("ransac.min_ratio", self.min_ratio_spin.value())))
        self.offset_spin.setValue(float(self.defaults.get("ransac.offset", self.offset_spin.value())))
        up = str(self.defaults.get("ransac.up_axis", self.up_axis_combo.currentText()))
        idx = max(0, self.up_axis_combo.findText(up))
        self.up_axis_combo.setCurrentIndex(idx)

        # Outline
        self.scale_factor_spin.setValue(float(self.defaults.get("outline.scale_factor", self.scale_factor_spin.value())))
        self.contour_min_spin.setValue(int(self.defaults.get("outline.contour_min", self.contour_min_spin.value())))
        self.dbscan_eps_spin.setValue(float(self.defaults.get("outline.dbscan_eps", self.dbscan_eps_spin.value())))
        self.dbscan_min_spin.setValue(int(self.defaults.get("outline.dbscan_min", self.dbscan_min_spin.value())))

        # Morph
        self.kernel_size_spin.setValue(int(self.defaults.get("morph.kernel_size", self.kernel_size_spin.value())))

        # Morph->PCD
        self.voxel_size_morph_spin.setValue(float(self.defaults.get("m2p.voxel", self.voxel_size_morph_spin.value())))
        self.merge_at_end_checkbox.setChecked(bool(self.defaults.get("m2p.merge", self.merge_at_end_checkbox.isChecked())))

        # Mesh
        self.mesh_visualize_checkbox.setChecked(bool(self.defaults.get("mesh.visualize", self.mesh_visualize_checkbox.isChecked())))

    # ---- Tabs ----
    def _tab_remove(self):
        w = QWidget(); L = QVBoxLayout(w)

        row = QHBoxLayout(); row.addWidget(QLabel("벽 두께(m):"))
        self.wall_thickness_spin = QDoubleSpinBox(); self.wall_thickness_spin.setRange(0.01, 50.0); self.wall_thickness_spin.setValue(0.15)
        row.addWidget(self.wall_thickness_spin); L.addLayout(row)

        row = QHBoxLayout(); row.addWidget(QLabel("천장 두께(m):"))
        self.ceiling_thickness_spin = QDoubleSpinBox(); self.ceiling_thickness_spin.setRange(0.1, 50.0); self.ceiling_thickness_spin.setValue(5.0)
        row.addWidget(self.ceiling_thickness_spin); L.addLayout(row)

        row = QHBoxLayout(); row.addWidget(QLabel("복셀(m):"))
        self.voxel_size_spin = QDoubleSpinBox(); self.voxel_size_spin.setRange(0.001, 1.0); self.voxel_size_spin.setDecimals(3); self.voxel_size_spin.setValue(0.03)
        row.addWidget(self.voxel_size_spin); L.addLayout(row)

        row = QHBoxLayout(); row.addWidget(QLabel("높이 축(천장):"))
        self.height_axis_combo = QComboBox(); self.height_axis_combo.addItems(["y","z","x"])
        row.addWidget(self.height_axis_combo); L.addLayout(row)

        # 신규: 벽 축/제거할 벽
        row = QHBoxLayout(); row.addWidget(QLabel("벽 축:"))
        self.wall_axis_combo = QComboBox(); self.wall_axis_combo.addItems(["z","x","y"])
        row.addWidget(self.wall_axis_combo); L.addLayout(row)

        row = QHBoxLayout(); row.addWidget(QLabel("제거할 벽:"))
        self.wall_sides_combo = QComboBox()
        self.wall_sides_combo.addItems(["양쪽(min,max)", "min쪽만", "max쪽만"])
        row.addWidget(self.wall_sides_combo); L.addLayout(row)

        L.addStretch()
        return w

    def _tab_ransac(self):
        w = QWidget(); L = QVBoxLayout(w)

        row = QHBoxLayout(); row.addWidget(QLabel("최대 평면 수:"))
        self.max_planes_spin = QSpinBox(); self.max_planes_spin.setRange(1, 1000); self.max_planes_spin.setValue(40)
        row.addWidget(self.max_planes_spin); L.addLayout(row)

        row = QHBoxLayout(); row.addWidget(QLabel("거리 임계값(m):"))
        self.dist_threshold_spin = QDoubleSpinBox(); self.dist_threshold_spin.setRange(0.0001, 1.0); self.dist_threshold_spin.setDecimals(4); self.dist_threshold_spin.setValue(0.02)
        row.addWidget(self.dist_threshold_spin); L.addLayout(row)

        row = QHBoxLayout(); row.addWidget(QLabel("RANSAC N:"))
        self.ransac_n_spin = QSpinBox(); self.ransac_n_spin.setRange(3,10); self.ransac_n_spin.setValue(3)
        row.addWidget(self.ransac_n_spin); L.addLayout(row)

        row = QHBoxLayout(); row.addWidget(QLabel("반복 횟수:"))
        self.iterations_spin = QSpinBox(); self.iterations_spin.setRange(100, 100000); self.iterations_spin.setValue(1000)
        row.addWidget(self.iterations_spin); L.addLayout(row)

        row = QHBoxLayout(); row.addWidget(QLabel("최소 인라이어 비율:"))
        self.min_ratio_spin = QDoubleSpinBox(); self.min_ratio_spin.setRange(0.000, 1.0); self.min_ratio_spin.setDecimals(3); self.min_ratio_spin.setValue(0.01)
        row.addWidget(self.min_ratio_spin); L.addLayout(row)

        row = QHBoxLayout(); row.addWidget(QLabel("수직 오프셋(m):"))
        self.offset_spin = QDoubleSpinBox(); self.offset_spin.setRange(-10.0, 50.0); self.offset_spin.setDecimals(3); self.offset_spin.setValue(1.0)
        row.addWidget(self.offset_spin); L.addLayout(row)

        row = QHBoxLayout(); row.addWidget(QLabel("Up 축:"))
        self.up_axis_combo = QComboBox(); self.up_axis_combo.addItems(["y","x","z"])
        row.addWidget(self.up_axis_combo); L.addLayout(row)

        self.autotilt_checkbox = QCheckBox("자동 기울기 보정"); self.autotilt_checkbox.setChecked(False); L.addWidget(self.autotilt_checkbox)
        self.visualize_checkbox = QCheckBox("시각화"); self.visualize_checkbox.setChecked(True); L.addWidget(self.visualize_checkbox)
        L.addStretch(); return w

    def _tab_outline(self):
        w = QWidget(); L = QVBoxLayout(w)

        row = QHBoxLayout(); row.addWidget(QLabel("스케일 팩터:"))
        self.scale_factor_spin = QDoubleSpinBox(); self.scale_factor_spin.setRange(0.1, 100.0); self.scale_factor_spin.setDecimals(3); self.scale_factor_spin.setValue(5.0)
        row.addWidget(self.scale_factor_spin); L.addLayout(row)

        row = QHBoxLayout(); row.addWidget(QLabel("윤곽 최소 크기(px):"))
        self.contour_min_spin = QSpinBox(); self.contour_min_spin.setRange(1, 100000); self.contour_min_spin.setValue(10)
        row.addWidget(self.contour_min_spin); L.addLayout(row)

        row = QHBoxLayout(); row.addWidget(QLabel("DBSCAN eps:"))
        self.dbscan_eps_spin = QDoubleSpinBox(); self.dbscan_eps_spin.setRange(0.0001, 10.0); self.dbscan_eps_spin.setDecimals(4); self.dbscan_eps_spin.setValue(0.03)
        row.addWidget(self.dbscan_eps_spin); L.addLayout(row)

        row = QHBoxLayout(); row.addWidget(QLabel("DBSCAN min_samples:"))
        self.dbscan_min_spin = QSpinBox(); self.dbscan_min_spin.setRange(1, 100000); self.dbscan_min_spin.setValue(30)
        row.addWidget(self.dbscan_min_spin); L.addLayout(row)
        L.addStretch(); return w

    def _tab_morph(self):
        w = QWidget(); L = QVBoxLayout(w)
        row = QHBoxLayout(); row.addWidget(QLabel("커널 크기:"))
        self.kernel_size_spin = QSpinBox(); self.kernel_size_spin.setRange(1, 99); self.kernel_size_spin.setValue(5)
        row.addWidget(self.kernel_size_spin); L.addLayout(row)
        L.addStretch(); return w

    def _tab_morph_to_pcd(self):
        w = QWidget(); L = QVBoxLayout(w)
        row = QHBoxLayout(); row.addWidget(QLabel("복셀(m):"))
        self.voxel_size_morph_spin = QDoubleSpinBox(); self.voxel_size_morph_spin.setRange(0.001,0.1); self.voxel_size_morph_spin.setDecimals(3); self.voxel_size_morph_spin.setValue(0.01)
        row.addWidget(self.voxel_size_morph_spin); L.addLayout(row)
        self.merge_at_end_checkbox = QCheckBox("최종 병합"); self.merge_at_end_checkbox.setChecked(False); L.addWidget(self.merge_at_end_checkbox)
        L.addStretch(); return w

    def _tab_mesh(self):
        w = QWidget(); L = QVBoxLayout(w)
        row = QHBoxLayout(); row.addWidget(QLabel("메시 품질:"))
        self.mesh_quality_combo = QComboBox(); self.mesh_quality_combo.addItems(["low","medium","high"]); self.mesh_quality_combo.setCurrentText("medium")
        row.addWidget(self.mesh_quality_combo); L.addLayout(row)
        self.mesh_visualize_checkbox = QCheckBox("시각화"); self.mesh_visualize_checkbox.setChecked(True); L.addWidget(self.mesh_visualize_checkbox)
        L.addStretch(); return w

    # ---- Helpers ----
    def browse_input_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "PCD 파일 선택", "", "Point Cloud (*.pcd *.ply *.obj)")
        if path: self.input_path_edit.setText(path)

    def browse_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "출력 디렉토리 선택")
        if d: self.output_path_edit.setText(d)

    def log(self, msg):
        self.log_text.append(msg); self.log_text.ensureCursorVisible()

    def _run_async(self, script_file, args):
        if self.current_runner and self.current_runner.isRunning():
            QMessageBox.warning(self, "경고", "이미 다른 스크립트 실행 중.")
            return
        script_abs = os.path.join(self.repo_dir, script_file)
        self.current_runner = ScriptRunner(script_abs, args, self.repo_dir)
        self.current_runner.output_signal.connect(self.log)
        self.current_runner.finished_signal.connect(self._on_any_finished)
        self.current_runner.start()

    # ---- Rect / General 선택 다이얼로그 ----
    def _choose_remove_script(self) -> str:
        ret = QMessageBox.question(
            self,
            "모델 형태 선택",
            "모델의 평면 모양이 네모(직사각형)인가요?\n\n"
            "예(Yes): 직사각 전제 - 빠름, 간단 (remove_rect.py)\n"
            "아니오(No): 일반형 - 꺾인 벽/복잡형 대응 (remove.py)",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        return "remove_rect.py" if ret == QMessageBox.Yes else "remove.py"

    # ---- Individual run ----
    def run_remove(self):
        ipath = self.input_path_edit.text()
        if not ipath:
            QMessageBox.warning(self, "경고", "입력 파일을 선택해주세요.")
            return
        out = self.output_path_edit.text()

        script = self._choose_remove_script()
        if script == "remove_rect.py":
            # GUI → CLI 매핑
            sides_idx = self.wall_sides_combo.currentIndex()
            wall_sides = "both" if sides_idx == 0 else ("min" if sides_idx == 1 else "max")

            args = [
                "--input", ipath,
                "--out", out,
                "--wall-thickness", str(self.wall_thickness_spin.value()),
                "--ceiling-thickness", str(self.ceiling_thickness_spin.value()),
                "--voxel", str(self.voxel_size_spin.value()),
                "--height-axis", self.height_axis_combo.currentText(),
                "--wall-axis", self.wall_axis_combo.currentText(),
                "--wall-sides", wall_sides
            ]
        else:
            args = [
                "--input", ipath,
                "--out", out,
                "--wall-thickness", str(self.wall_thickness_spin.value()),   # 하위호환 → wall-band로 매핑
                "--ceiling-thickness", str(self.ceiling_thickness_spin.value()),
                "--post-voxel", str(self.voxel_size_spin.value()),
                "--height-axis", self.height_axis_combo.currentText(),
                "--band-method", "erosion",
                "--only-vertical",
                "--vertical-angle-deg", "20",
                "--grid-cell", "0.05",
                "--morph-k", "3",
                "--min-area-px", "100"
            ]
        self._run_async(script, args)

    def run_ransac(self):
        out = self.output_path_edit.text()
        nonowall = os.path.join(out, "nonowall.pcd")
        if not os.path.exists(nonowall):
            QMessageBox.warning(self, "경고", "먼저 Remove를 돌려서 nonowall.pcd를 만들어라.")
            return
        args = [
            "--input", nonowall,
            "--out", os.path.join(out, "ransac"),
            "--max-planes", str(self.max_planes_spin.value()),
            "--dist", str(self.dist_threshold_spin.value()),
            "--ransac-n", str(self.ransac_n_spin.value()),
            "--iters", str(self.iterations_spin.value()),
            "--min-ratio", str(self.min_ratio_spin.value()),
            "--offset", str(self.offset_spin.value()),
            "--up-axis", self.up_axis_combo.currentText(),
        ]
        if self.autotilt_checkbox.isChecked(): args.append("--autotilt")
        if not self.visualize_checkbox.isChecked(): args.append("--no-vis")
        self._run_async("ransac.py", args)

    def run_outline(self):
        args = [
            "--scale-factor", str(self.scale_factor_spin.value()),
            "--contour-min", str(self.contour_min_spin.value()),
            "--dbscan-eps", str(self.dbscan_eps_spin.value()),
            "--dbscan-min", str(self.dbscan_min_spin.value())
        ]
        self._run_async("outline.py", args)

    def run_morph(self):
        args = ["--kernel-size", str(self.kernel_size_spin.value())]
        self._run_async("morph.py", args)

    def run_morph_to_pcd(self):
        args = ["--voxel-size", str(self.voxel_size_morph_spin.value())]
        if self.merge_at_end_checkbox.isChecked():
            args.append("--merge-at-end")
        self._run_async("morph_to_pcd.py", args)

    def run_pcd_to_mesh(self):
        args = []
        if self.mesh_visualize_checkbox.isChecked():
            args.append("--visualize")
        else:
            args.append("--no-vis")
        self._run_async("pcd_to_mesh.py", args)

    # ---- Pipeline ----
    def run_all_pipeline(self):
        ipath = self.input_path_edit.text()
        if not ipath:
            QMessageBox.warning(self, "경고", "입력 파일부터 선택해.")
            return
        out = self.output_path_edit.text()
        self.log("=== 전체 파이프라인 시작 ===")

        self.pipeline_queue = []

        # Remove (분기)
        script = self._choose_remove_script()
        if script == "remove_rect.py":
            sides_idx = self.wall_sides_combo.currentIndex()
            wall_sides = "both" if sides_idx == 0 else ("min" if sides_idx == 1 else "max")

            self.pipeline_queue.append((
                "remove_rect.py",
                ["--input", ipath, "--out", out,
                 "--wall-thickness", str(self.wall_thickness_spin.value()),
                 "--ceiling-thickness", str(self.ceiling_thickness_spin.value()),
                 "--voxel", str(self.voxel_size_spin.value()),
                 "--height-axis", self.height_axis_combo.currentText(),
                 "--wall-axis", self.wall_axis_combo.currentText(),
                 "--wall-sides", wall_sides],
                self.repo_dir
            ))
        else:
            self.pipeline_queue.append((
                "remove.py",
                ["--input", ipath, "--out", out,
                 "--wall-thickness", str(self.wall_thickness_spin.value()),  # 하위호환
                 "--ceiling-thickness", str(self.ceiling_thickness_spin.value()),
                 "--post-voxel", str(self.voxel_size_spin.value()),
                 "--height-axis", self.height_axis_combo.currentText(),
                 "--band-method", "erosion",
                 "--only-vertical",
                 "--vertical-angle-deg", "20",
                 "--grid-cell", "0.05",
                 "--morph-k", "3",
                 "--min-area-px", "100"],
                self.repo_dir
            ))

        # RANSAC
        self.pipeline_queue.append((
            "ransac.py",
            ["--input", os.path.join(out, "nonowall.pcd"),
             "--out", os.path.join(out, "ransac"),
             "--max-planes", str(self.max_planes_spin.value()),
             "--dist", str(self.dist_threshold_spin.value()),
             "--ransac-n", str(self.ransac_n_spin.value()),
             "--iters", str(self.iterations_spin.value()),
             "--min-ratio", str(self.min_ratio_spin.value()),
             "--offset", str(self.offset_spin.value()),
             "--up-axis", self.up_axis_combo.currentText()] +
            (["--autotilt"] if self.autotilt_checkbox.isChecked() else []) +
            (["--no-vis"] if not self.visualize_checkbox.isChecked() else []),
            self.repo_dir
        ))

        # Outline / Morph / Morph->PCD / Mesh
        self.pipeline_queue += [
            ("outline.py", [
                "--scale-factor", str(self.scale_factor_spin.value()),
                "--contour-min", str(self.contour_min_spin.value()),
                "--dbscan-eps", str(self.dbscan_eps_spin.value()),
                "--dbscan-min", str(self.dbscan_min_spin.value())
            ], self.repo_dir),
            ("morph.py", [
                "--kernel-size", str(self.kernel_size_spin.value())
            ], self.repo_dir),
            ("morph_to_pcd.py", [
                "--voxel-size", str(self.voxel_size_morph_spin.value())
            ] + (["--merge-at-end"] if self.merge_at_end_checkbox.isChecked() else []), self.repo_dir),
            ("pcd_to_mesh.py", [
                "--visualize" if self.mesh_visualize_checkbox.isChecked() else "--no-vis"
            ], self.repo_dir),
        ]

        self._run_next_in_queue()

    def _run_next_in_queue(self):
        if not self.pipeline_queue:
            self.log("=== 전체 파이프라인 완료 ===")
            return
        script, args, wdir = self.pipeline_queue.pop(0)
        script_abs = os.path.join(self.repo_dir, script)
        self.current_runner = ScriptRunner(script_abs, args, wdir)
        self.current_runner.output_signal.connect(self.log)
        self.current_runner.finished_signal.connect(self._on_pipeline_step_finished)
        self.current_runner.start()

    # ---- Slots ----
    def _on_any_finished(self, ok, msg):
        self.log(("[OK] " if ok else "[ERROR] ") + msg)
        self.current_runner = None

    def _on_pipeline_step_finished(self, ok, msg):
        self.log(("[OK] " if ok else "[ERROR] ") + msg)
        self.current_runner = None
        if not ok:
            self.log("파이프라인 중단.")
            self.pipeline_queue.clear()
            return
        self._run_next_in_queue()

# ------------------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
