"""SlicerTMS: Interactive TMS E-field visualization using TMSWarp FEM.

Connects to TMSService.py (RPyC) running as a PythonSlicer subprocess.
Move the 'TMS Probe' linear transform to update the E-field in real time.
"""

import logging
import multiprocessing.shared_memory
import multiprocessing.spawn
import os
import signal
import subprocess
import sys
import time

import numpy
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *

_MODULE_DIR  = os.path.dirname(os.path.realpath(__file__))
_REPO_ROOT   = os.path.abspath(os.path.join(_MODULE_DIR, "..", ".."))
_SERVICE_PATH = os.path.join(_REPO_ROOT, "Experiments", "TMSService.py")
_TMSWARP_ROOT = os.path.join(_REPO_ROOT, "TMSWarp")
_TMSWARP_SRC  = os.path.join(_TMSWARP_ROOT, "src")

DEFAULT_PORT   = 18892
_TEST_CACHE    = os.path.join(os.path.expanduser("~"), ".cache", "SlicerTMS")
_LOG_DIR       = os.path.join(_TEST_CACHE, "logs")

# Fix for PythonSlicer: sys._base_executable is '' in the embedded interpreter,
# which causes multiprocessing.resource_tracker to exec an empty path and die.
# Setting the spawn executable to sys.executable lets SharedMemory work normally.
# NOTE: This must NOT run at module load time — Slicer sets sys.executable AFTER
# loading scripted modules, so it would be empty.  Call _fix_spawn_executable()
# before the first SharedMemory use instead.
_spawn_exe_fixed = False

def _fix_spawn_executable():
    global _spawn_exe_fixed
    if _spawn_exe_fixed:
        return
    _spawn_exe_fixed = True
    if not multiprocessing.spawn.get_executable() or \
       multiprocessing.spawn.get_executable() == b'':
        exe = sys.executable
        if exe:
            multiprocessing.spawn.set_executable(exe)
            logging.getLogger("SlicerTMS").info(
                f"Fixed multiprocessing spawn executable: {exe}"
            )

# ---------------------------------------------------------------------------
# Session logging — keeps the last 10 session logs
# ---------------------------------------------------------------------------

def _setup_session_logger():
    """Create a per-session file logger under ~/.cache/SlicerTMS/logs/.

    Rotates old logs so only the 10 most recent are kept.
    Returns the logger instance.
    """
    import datetime, glob

    os.makedirs(_LOG_DIR, exist_ok=True)

    # Rotate: keep only last 9 (the new one will make 10)
    existing = sorted(glob.glob(os.path.join(_LOG_DIR, "tms_session_*.log")))
    while len(existing) >= 10:
        try:
            os.remove(existing.pop(0))
        except OSError:
            pass

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(_LOG_DIR, f"tms_session_{stamp}.log")

    logger = logging.getLogger("SlicerTMS")
    logger.setLevel(logging.DEBUG)
    logger.propagate = True  # also send to Slicer's log handler

    # Remove any stale file handlers from a previous module reload
    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(fh)
    logger.info(f"Session log: {log_path}")
    return logger

log = _setup_session_logger()

# QProcess enum → human-readable name (pattern from SlicerParallelProcessing)
_QProcessStateNames = {
    qt.QProcess.NotRunning: "NotRunning",
    qt.QProcess.Starting:   "Starting",
    qt.QProcess.Running:    "Running",
}
_QProcessErrorNames = {
    qt.QProcess.FailedToStart: "FailedToStart",
    qt.QProcess.Crashed:       "Crashed",
    qt.QProcess.Timedout:      "Timedout",
    qt.QProcess.WriteError:    "WriteError",
    qt.QProcess.ReadError:     "ReadError",
    qt.QProcess.UnknownError:  "UnknownError",
}


def _kill_processes_on_port(port):
    """Find and kill any processes listening on *port* (zombie cleanup).

    Tries multiple tools in order of reliability:
      1. ``fuser`` (Linux) — directly reports PIDs on a TCP port
      2. ``lsof`` (macOS/Linux) — ``-ti :PORT`` for terse PID output
      3. ``pgrep`` — matches TMSService command lines with ``--port PORT``
    """
    pids = set()
    my_pid = os.getpid()

    # --- fuser (Linux) ---
    try:
        result = subprocess.run(
            ["fuser", f"{port}/tcp"],
            capture_output=True, text=True, timeout=5,
        )
        # fuser writes PIDs to stderr
        for tok in (result.stdout + " " + result.stderr).split():
            tok = tok.strip().rstrip("e")  # fuser may append 'e' for established
            if tok.isdigit():
                pids.add(int(tok))
    except (FileNotFoundError, Exception):
        pass

    # --- lsof (macOS, some Linux) ---
    if not pids:
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5,
            )
            for tok in result.stdout.split():
                if tok.strip().isdigit():
                    pids.add(int(tok))
        except (FileNotFoundError, Exception):
            pass

    # --- pgrep fallback (matches TMSService command line) ---
    if not pids:
        try:
            result = subprocess.run(
                ["pgrep", "-f", f"TMSService.*--port {port}"],
                capture_output=True, text=True, timeout=5,
            )
            for tok in result.stdout.split():
                if tok.strip().isdigit():
                    pids.add(int(tok))
        except (FileNotFoundError, Exception):
            pass

    killed = []
    for pid in pids:
        if pid == my_pid:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
            log.info(f"Killed zombie process {pid} on port {port}")
        except ProcessLookupError:
            pass
        except PermissionError:
            log.warning(f"No permission to kill PID {pid} on port {port}")
    return killed


def _cleanup_orphaned_shm(name):
    """Try to unlink an orphaned POSIX shared-memory segment with *name*."""
    _fix_spawn_executable()
    try:
        shm = multiprocessing.shared_memory.SharedMemory(name=name, create=False)
        shm.close()
        shm.unlink()
        log.info(f"Cleaned up orphaned shared memory '{name}'")
    except FileNotFoundError:
        pass
    except Exception as exc:
        log.warning(f"Could not clean shared memory '{name}': {exc}")


# ============================================================
# Module descriptor
# ============================================================

class SlicerTMS(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SlicerTMS"
        self.parent.categories = ["TMS"]
        self.parent.dependencies = []
        self.parent.contributors = ["SlicerTMS developers"]
        self.parent.helpText = (
            "Interactive TMS E-field visualization driven by TMSWarp FEM solvers. "
            "Move the 'TMS Probe' transform to update the E-field in real time. "
            "Solver backends: NumPy (pre-factorized LU), Warp CPU, Warp GPU (CUDA)."
        )
        self.parent.acknowledgementText = ""


# ============================================================
# Widget
# ============================================================

class SlicerTMSWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = None

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = SlicerTMSLogic()

        # ---- Service section -----------------------------------------------
        svcSection = ctk.ctkCollapsibleButton()
        svcSection.text = "TMS Service"
        self.layout.addWidget(svcSection)
        svcForm = qt.QFormLayout(svcSection)

        # Mesh file selector
        meshRow = qt.QHBoxLayout()
        self.meshPathEdit = ctk.ctkPathLineEdit()
        self.meshPathEdit.filters = ctk.ctkPathLineEdit.Files
        self.meshPathEdit.nameFilters = ["TMSWarp mesh (*.npz)"]
        meshRow.addWidget(self.meshPathEdit)
        autoBtn = qt.QPushButton("Auto")
        autoBtn.setFixedWidth(45)
        autoBtn.setToolTip("Auto-detect ernie_data.npz or sphere3_data.npz in the TMSWarp directory")
        autoBtn.clicked.connect(self._autoDetectMesh)
        meshRow.addWidget(autoBtn)
        dlBtn = qt.QPushButton("Download")
        dlBtn.setFixedWidth(70)
        dlBtn.setToolTip(
            "Download ernie_data.npz (requires SimNIBS; runs scripts/fetch_ernie.py)"
        )
        dlBtn.clicked.connect(self._downloadErnie)
        meshRow.addWidget(dlBtn)
        svcForm.addRow("Mesh (.npz):", meshRow)

        # One-click ernie setup — low-res (mesh only) or full (mesh + T1 MRI)
        ernieRow = qt.QHBoxLayout()
        ernieLowBtn = qt.QPushButton("Setup Ernie (low-res, ~394 MB)")
        ernieLowBtn.setToolTip(
            "Download the low-resolution mesh only. Fast download, no MRI."
        )
        ernieLowBtn.clicked.connect(lambda: self._setupErnieMesh(full=False))
        ernieRow.addWidget(ernieLowBtn)
        ernieFullBtn = qt.QPushButton("Setup Ernie + T1 MRI (~1.12 GB)")
        ernieFullBtn.setToolTip(
            "Download the full dataset with T1 MRI for background visualization."
        )
        ernieFullBtn.clicked.connect(lambda: self._setupErnieMesh(full=True))
        ernieRow.addWidget(ernieFullBtn)
        svcForm.addRow(ernieRow)

        # Solver combo — availability updated after creation
        self.solverCombo = qt.QComboBox()
        self.solverCombo.addItem("NumPy  —  pre-factorized LU,  CPU",       "numpy")
        self.solverCombo.addItem("Warp   —  CG iterative,       CPU",       "warp_cpu")
        self.solverCombo.addItem("Warp   —  CG iterative,       GPU / CUDA","warp_gpu")
        svcForm.addRow("Solver:", self.solverCombo)

        # Start / Stop
        svcBtnRow = qt.QHBoxLayout()
        self.startSvcBtn = qt.QPushButton("Start Service")
        self.startSvcBtn.clicked.connect(self._startService)
        svcBtnRow.addWidget(self.startSvcBtn)
        self.stopSvcBtn = qt.QPushButton("Stop Service")
        self.stopSvcBtn.setEnabled(False)
        self.stopSvcBtn.clicked.connect(self._stopService)
        svcBtnRow.addWidget(self.stopSvcBtn)
        svcForm.addRow(svcBtnRow)

        self.installDepsBtn = qt.QPushButton("Install / Check Dependencies")
        self.installDepsBtn.setToolTip("Install rpyc and warp-lang into Slicer's Python if missing")
        self.installDepsBtn.clicked.connect(self._installDependencies)
        svcForm.addRow(self.installDepsBtn)

        self.svcStatusLabel = qt.QLabel("Status: idle")
        svcForm.addRow("", self.svcStatusLabel)

        # ---- Visualization section -----------------------------------------
        vizSection = ctk.ctkCollapsibleButton()
        vizSection.text = "Visualization"
        self.layout.addWidget(vizSection)
        vizForm = qt.QFormLayout(vizSection)

        # Mesh model node — subject hierarchy combo for selecting loaded meshes
        self.meshNodeSelector = slicer.qMRMLSubjectHierarchyComboBox()
        self.meshNodeSelector.setMRMLScene(slicer.mrmlScene)
        self.meshNodeSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.meshNodeSelector.showRootItem = False
        self.meshNodeSelector.noneEnabled = True
        self.meshNodeSelector.setToolTip(
            "Head mesh model node carrying the E-field scalar overlay. "
            "Auto-created as 'TMS E-field' when the service initializes."
        )
        vizForm.addRow("Mesh node:", self.meshNodeSelector)

        # Probe transform — drives coil position
        self.probeSelector = slicer.qMRMLNodeComboBox()
        self.probeSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
        self.probeSelector.addEnabled = True
        self.probeSelector.removeEnabled = False
        self.probeSelector.noneEnabled = True
        self.probeSelector.showHidden = False
        self.probeSelector.setMRMLScene(slicer.mrmlScene)
        self.probeSelector.setToolTip(
            "Linear transform used as the TMS coil probe. "
            "Translate/rotate this node to move the coil and update the E-field."
        )
        self.probeSelector.currentNodeChanged.connect(self._onProbeNodeChanged)
        vizForm.addRow("Probe transform:", self.probeSelector)

        # Live solver switching
        switchRow = qt.QHBoxLayout()
        self.liveSolverCombo = qt.QComboBox()
        self.liveSolverCombo.addItem("NumPy  —  pre-factorized LU,  CPU",       "numpy")
        self.liveSolverCombo.addItem("Warp   —  CG iterative,       CPU",       "warp_cpu")
        self.liveSolverCombo.addItem("Warp   —  CG iterative,       GPU / CUDA","warp_gpu")
        switchRow.addWidget(self.liveSolverCombo)
        switchBtn = qt.QPushButton("Switch")
        switchBtn.setFixedWidth(55)
        switchBtn.setToolTip("Switch to the selected solver without reloading the mesh")
        switchBtn.clicked.connect(self._switchSolver)
        switchRow.addWidget(switchBtn)
        vizForm.addRow("Live solver:", switchRow)

        self.vizStatusLabel = qt.QLabel("")
        vizForm.addRow("", self.vizStatusLabel)

        self.layout.addStretch(1)

        # Auto-detect mesh and set solver availability on startup
        self._autoDetectMesh()
        self._updateSolverAvailability()

    def cleanup(self):
        if self.logic:
            try:
                self.logic.stopService()
            except Exception:
                pass  # best-effort during Slicer shutdown

    # ------------------------------------------------------------------
    # Slot helpers
    # ------------------------------------------------------------------

    def _setStatus(self, msg, label=None):
        (label or self.svcStatusLabel).setText(f"Status: {msg}")
        slicer.app.processEvents()

    def _autoDetectMesh(self):
        path = self.logic.autoDetectMesh()
        if path:
            self.meshPathEdit.currentPath = path
            self._setStatus(f"Mesh: {os.path.basename(path)}")
        else:
            self._setStatus("No mesh found — click Download or browse for .npz")

    def _downloadErnie(self):
        self._setStatus("Running scripts/fetch_ernie.py …")
        try:
            path = self.logic.downloadErnie()
            self.meshPathEdit.currentPath = path
            self._setStatus(f"Ready: {os.path.basename(path)}")
        except Exception as exc:
            self._setStatus(f"Download failed: {exc}")
            slicer.util.errorDisplay(str(exc))

    def _setupErnieMesh(self, full=False):
        """Download ernie_data.npz if needed, load into scene, create probe, pick solver."""
        if full:
            self._setStatus("Setting up ernie mesh + T1 MRI (may download ~1.12 GB) …")
        else:
            self._setStatus("Setting up ernie mesh (may download ~394 MB) …")
        try:
            path = self.logic.setupErnieMesh(full=full)
            self.meshPathEdit.currentPath = path
        except Exception as exc:
            self._setStatus(f"Ernie setup failed: {exc}")
            slicer.util.errorDisplay(str(exc))
            return

        # Load mesh into scene with conductivity/tag1/Enorm scalars
        self._setStatus("Loading mesh into scene …")
        try:
            modelNode = self.logic.loadMeshToScene(path)
        except Exception as exc:
            self._setStatus(f"Mesh loading failed: {exc}")
            slicer.util.errorDisplay(str(exc))
            return

        # Select the model node in the mesh node combo
        self.meshNodeSelector.setCurrentNode(modelNode)

        # Load T1 MRI as background volume if available (same space as mesh)
        t1Path = self.logic.findErnieT1()
        if t1Path:
            try:
                volumeNode = slicer.util.loadVolume(t1Path)
                slicer.util.setSliceViewerLayers(background=volumeNode)
            except Exception as exc:
                log.warning(f"Could not load T1 MRI: {exc}")

        # Create or find the probe transform and position it above the head
        probe = slicer.mrmlScene.GetFirstNodeByName("TMS Probe")
        if probe is None:
            probe = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLinearTransformNode", "TMS Probe"
            )
            probe.CreateDefaultDisplayNodes()
            probe.GetDisplayNode().SetEditorVisibility(True)

        mat = vtk.vtkMatrix4x4()   # identity — coil above head at origin
        mat.SetElement(2, 3, 100.0)  # z = 100 mm (above head)
        probe.SetMatrixTransformToParent(mat)
        self.probeSelector.setCurrentNode(probe)

        # Pick the best available solver
        self._updateSolverAvailability()
        self._setStatus(
            "Ernie mesh loaded (conductivity display). Click 'Start Service' to begin "
            f"(solver: {self.solverCombo.currentData})."
        )

    def _updateSolverAvailability(self):
        """Gray out solver options that are not currently usable."""
        warp_ok = False
        cuda_ok = False
        try:
            import warp as wp
            warp_ok = True
            cuda_ok = wp.is_cuda_available()
        except Exception:
            pass

        available = {"numpy": True, "warp_cpu": warp_ok, "warp_gpu": cuda_ok}
        tips = {
            "numpy":    "Pre-factorized LU direct solver — always available",
            "warp_cpu": "Warp CG on CPU — requires warp-lang (click Install Dependencies)",
            "warp_gpu": "Warp CG on GPU — requires CUDA GPU + warp-lang",
        }

        for combo in (self.solverCombo, self.liveSolverCombo):
            for i in range(combo.count):
                solver = combo.itemData(i)
                item   = combo.model().item(i)
                item.setEnabled(available.get(solver, False))
                item.setToolTip(tips.get(solver, ""))

            # Select best available: GPU > numpy > warp_cpu
            order = [2, 0, 1] if cuda_ok else [0, 1, 2]
            for idx in order:
                if available.get(combo.itemData(idx), False):
                    combo.setCurrentIndex(idx)
                    break

    def _installDependencies(self):
        self._setStatus("Installing dependencies …")
        try:
            self.logic.ensureDependencies()
            self._updateSolverAvailability()
            self._setStatus("Dependencies OK")
        except Exception as exc:
            self._setStatus(f"Error: {exc}")
            slicer.util.errorDisplay(str(exc))

    def _startService(self):
        meshPath = self.meshPathEdit.currentPath
        solver   = self.solverCombo.currentData
        try:
            self._setStatus("Installing dependencies …")
            self.logic.ensureDependencies()
            self._updateSolverAvailability()
            self._setStatus("Starting TMSService …")
            self.logic.startService(meshPath, solver, DEFAULT_PORT)
            self._setStatus("Connecting …")
            self.logic.connectToService(DEFAULT_PORT)
            self._setStatus("Initializing FEM (may take several minutes for ernie/numpy) …")
            self.logic.initializeFEM(meshPath, solver)
            self._setStatus("Setting up visualization …")
            self.logic.setupVisualization()
            # Select the mesh node and probe in the combos
            if self.logic._meshNode is not None:
                self.meshNodeSelector.setCurrentNode(self.logic._meshNode)
            probe = slicer.mrmlScene.GetFirstNodeByName("TMS Probe")
            if probe:
                self.probeSelector.setCurrentNode(probe)
            self._setStatus("Running — move 'TMS Probe' transform to update the E-field.")
            self.startSvcBtn.setEnabled(False)
            self.stopSvcBtn.setEnabled(True)
        except Exception as exc:
            self._setStatus(f"Error: {exc}")
            slicer.util.errorDisplay(str(exc))

    def _stopService(self):
        self.logic.stopService()
        self._setStatus("Stopped")
        self.startSvcBtn.setEnabled(True)
        self.stopSvcBtn.setEnabled(False)

    def _onProbeNodeChanged(self, node):
        self.logic.setProbeTransformNode(node)

    def _switchSolver(self):
        solver = self.liveSolverCombo.currentData
        self._setStatus(f"Switching to {solver} …", self.vizStatusLabel)
        try:
            self.logic.switchSolver(solver)
            self._setStatus(f"Solver: {solver}", self.vizStatusLabel)
        except Exception as exc:
            self._setStatus(f"Error: {exc}", self.vizStatusLabel)
            slicer.util.errorDisplay(str(exc))


# ============================================================
# Logic
# ============================================================

class SlicerTMSLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self._process      = None   # qt.QProcess for TMSService
        self._tms          = None   # RPyC connection
        self._shm          = None   # shared memory block
        self._sharedEnorm  = None   # numpy view into shared memory (scalar |E|)
        self._shmName      = f"tmsSharedE_{os.getpid()}_{id(self):x}"
        self._port         = None   # port used by current service
        self._meshNode     = None   # vtkMRMLModelNode for E-field display
        self._probeNode    = None   # vtkMRMLLinearTransformNode
        self._probeObsTag  = None   # observer tag
        self._probeMatrix  = vtk.vtkMatrix4x4()
        self._surfaceCellMap = None  # maps surface triangle → original tet index
        # QProcess signal slots (stored to prevent GC)
        self._onStateChangedSlot = None
        self._onReadyReadOutSlot = None
        self._onReadyReadErrSlot = None
        self._onFinishedSlot     = None

    # ------------------------------------------------------------------
    # Dependency / environment helpers
    # ------------------------------------------------------------------

    @staticmethod
    def hasCudaGpu():
        """Return True if warp detects at least one CUDA device."""
        try:
            import warp as wp
            return wp.is_cuda_available()
        except Exception:
            return False

    _TMSWARP_GIT = "https://github.com/pieper/TMSWarp.git"

    @staticmethod
    def ensureDependencies():
        """Install TMSWarp and service dependencies into Slicer's Python.

        Installs/upgrades tmswarp from git so that a fresh machine only needs
        the SlicerTMS repo — no manual TMSWarp checkout required.
        Also installs warp-lang and rpyc which are needed by the service but
        are not core tmswarp dependencies.
        """
        # Always upgrade tmswarp from git to pick up latest changes.
        # numpy and scipy come in automatically via tmswarp's pyproject.toml.
        slicer.util.pip_install(
            f"--upgrade git+{SlicerTMSLogic._TMSWARP_GIT}"
        )

        # warp-lang and rpyc are not in tmswarp's core deps
        for pkg, mod in [("warp-lang", "warp"), ("rpyc", "rpyc")]:
            try:
                __import__(mod)
            except ImportError:
                slicer.util.pip_install(pkg)

    # ------------------------------------------------------------------
    # Mesh helpers
    # ------------------------------------------------------------------

    @staticmethod
    def autoDetectMesh():
        """Return path to first found .npz mesh file, or None."""
        for name in ("ernie_data.npz", "sphere3_data.npz"):
            path = os.path.join(_TMSWARP_ROOT, name)
            if os.path.isfile(path):
                return path
        return None

    @staticmethod
    def downloadErnie():
        """Run scripts/fetch_ernie.py to build ernie_data.npz.

        Requires SimNIBS to be installed; see fetch_ernie.py for details.
        """
        fetchScript = os.path.join(_TMSWARP_ROOT, "scripts", "fetch_ernie.py")
        if not os.path.isfile(fetchScript):
            raise FileNotFoundError(
                f"fetch_ernie.py not found at {fetchScript}"
            )
        # Try SimNIBS Python first (it has mesh_io), then sys.executable
        simnibs_python = os.path.join(
            os.path.expanduser("~"), "Applications",
            "SimNIBS-4.5", "simnibs_env", "bin", "python"
        )
        python = simnibs_python if os.path.isfile(simnibs_python) else sys.executable
        result = subprocess.run(
            [python, fetchScript],
            capture_output=True, text=True, cwd=_TMSWARP_ROOT
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"fetch_ernie.py failed (returncode={result.returncode}):\n"
                f"{result.stderr}\n\n"
                "Hint: fetch_ernie.py requires SimNIBS to read the .msh file.\n"
                "Run it manually with the SimNIBS Python:\n"
                "  /path/to/SimNIBS-4.5/simnibs_env/bin/python "
                f"{fetchScript}"
            )
        outPath = os.path.join(_TMSWARP_ROOT, "ernie_data.npz")
        if not os.path.isfile(outPath):
            raise FileNotFoundError("ernie_data.npz not found after fetch_ernie.py ran")
        return outPath

    @staticmethod
    def setupErnieMesh(full=False):
        """Return path to ernie_data.npz, downloading and converting if needed.

        Parameters
        ----------
        full : bool
            If True, download the full SimNIBS v4.1 dataset (~1.12 GB) which
            includes the T1 MRI.  If False (default), download the low-res
            mesh-only zip (~394 MB).

        Search order:
          1. _TMSWARP_ROOT/ernie_data.npz  (local TMSWarp checkout)
          2. _TEST_CACHE/ernie_data.npz    (previously cached download)
          3. Download zip + convert via meshio
        """
        _ERNIE_URL_LOWRES = (
            "https://github.com/simnibs/example-dataset/releases/"
            "download/v4.0-lowres/ernie_lowres_V2.zip"
        )
        _ERNIE_URL_FULL = (
            "https://github.com/simnibs/example-dataset/releases/"
            "download/v4.1/simnibs4_examples.zip"
        )
        _ERNIE_URL = _ERNIE_URL_FULL if full else _ERNIE_URL_LOWRES
        _CONDUCTIVITY_MAP = {
            1: 0.126,   # white matter
            2: 0.275,   # gray matter
            3: 1.654,   # CSF
            4: 0.010,   # skull
            5: 0.465,   # scalp
            6: 0.500,   # eye balls
        }

        t1_cached = os.path.join(_TEST_CACHE, "ernie_T1.nii.gz")
        need_t1 = full and not os.path.isfile(t1_cached)

        # 1. Local checkout
        local = os.path.join(_TMSWARP_ROOT, "ernie_data.npz")
        if os.path.isfile(local) and not need_t1:
            return local

        # 2. Cache
        os.makedirs(_TEST_CACHE, exist_ok=True)
        cached = os.path.join(_TEST_CACHE, "ernie_data.npz")
        if os.path.isfile(cached) and not need_t1:
            return cached

        # 3. Download + convert
        # Try SimNIBS Python first (has mesh_io, avoids meshio dependency)
        # Skip this path when we need T1 — fetch_ernie.py only gets the low-res zip
        fetch_script = os.path.join(_TMSWARP_ROOT, "scripts", "fetch_ernie.py")
        simnibs_python = os.path.join(
            os.path.expanduser("~"), "Applications",
            "SimNIBS-4.5", "simnibs_env", "bin", "python",
        )
        if not need_t1 and os.path.isfile(simnibs_python) and os.path.isfile(fetch_script):
            result = subprocess.run(
                [simnibs_python, fetch_script],
                capture_output=True, text=True, cwd=_TMSWARP_ROOT,
            )
            if result.returncode == 0 and os.path.isfile(local):
                return local

        # Fall back: download zip and convert with meshio (no SimNIBS needed)
        slicer.util.pip_install("meshio")
        import meshio
        import shutil
        import tempfile
        import urllib.request
        import zipfile
        import numpy as _np

        print(f"Downloading ernie dataset from {_ERNIE_URL} …")
        tmp_zip = tempfile.mktemp(suffix=".zip")
        urllib.request.urlretrieve(_ERNIE_URL, tmp_zip)

        tmp_dir = tempfile.mkdtemp(prefix="ernie_extract_")
        try:
            with zipfile.ZipFile(tmp_zip) as z:
                msh_names = [n for n in z.namelist() if n.endswith("ernie.msh")]
                if not msh_names:
                    raise FileNotFoundError("ernie.msh not found in downloaded zip")
                z.extract(msh_names[0], tmp_dir)

                # Also extract T1 MRI if present in the full dataset
                t1_names = [n for n in z.namelist() if n.endswith("ernie_T1.nii.gz")]
                if full and t1_names:
                    z.extract(t1_names[0], tmp_dir)
                    t1_src = os.path.join(tmp_dir, t1_names[0])
                    t1_cached = os.path.join(_TEST_CACHE, "ernie_T1.nii.gz")
                    shutil.copy2(t1_src, t1_cached)
                    print(f"Cached T1 MRI → {t1_cached}")

            msh_path = os.path.join(tmp_dir, msh_names[0])

            print(f"Converting {msh_path} with meshio …")
            m = meshio.read(msh_path)

            # SimNIBS .msh: points are in mm; tetrahedra carry physical group tags
            nodes_m = m.points * 1e-3                               # mm → metres
            tets    = m.cells_dict["tetra"].astype(_np.int32)       # 0-based
            tags    = m.cell_data_dict["gmsh:physical"]["tetra"].astype(_np.int32)
            sigma   = _np.array(
                [_CONDUCTIVITY_MAP.get(int(t), 0.275) for t in tags],
                dtype=_np.float64,
            )

            _np.savez_compressed(
                cached,
                nodes=nodes_m, elements=tets, conductivity=sigma, tag1=tags,
            )
            print(f"Saved ernie_data.npz → {cached}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            try:
                os.remove(tmp_zip)
            except Exception:
                pass

        return cached

    @staticmethod
    def findErnieT1():
        """Return cached ernie_T1.nii.gz path if it exists, else None."""
        path = os.path.join(_TEST_CACHE, "ernie_T1.nii.gz")
        return path if os.path.isfile(path) else None

    # ------------------------------------------------------------------
    # Mesh loading into Slicer scene
    # ------------------------------------------------------------------

    def loadMeshToScene(self, npzPath):
        """Load a TMSWarp .npz mesh into the Slicer scene as a model node.

        Creates a vtkUnstructuredGrid with tetrahedral cells and three
        cell-data scalar arrays:
          - "Enorm"        — E-field magnitude (initially zero)
          - "conductivity" — tissue conductivity in S/m
          - "tag1"         — tissue type label (1-6), if present in the .npz

        Returns the vtkMRMLModelNode.  Sets self._meshNode.
        """
        log.info(f"loadMeshToScene: {npzPath}")
        data = numpy.load(npzPath)
        nodes_m      = data["nodes"]
        elements     = data["elements"]
        conductivity = data["conductivity"]
        tag1         = data["tag1"] if "tag1" in data else None
        nCells       = elements.shape[0]

        nodes_mm = nodes_m * 1000.0  # metres → mm for VTK / Slicer

        # --- Build temporary VTK unstructured grid for surface extraction ---
        meshGrid = vtk.vtkUnstructuredGrid()

        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(len(nodes_mm))
        vtk.util.numpy_support.vtk_to_numpy(pts.GetData())[:] = nodes_mm
        meshGrid.SetPoints(pts)

        offsets = numpy.arange(0, nCells * 4 + 1, 4, dtype=numpy.int64)
        connectivity = numpy.ascontiguousarray(
            elements.ravel(), dtype=numpy.int64
        )
        cells = vtk.vtkCellArray()
        cells.SetData(
            vtk.util.numpy_support.numpy_to_vtk(offsets, deep=True),
            vtk.util.numpy_support.numpy_to_vtk(connectivity, deep=True),
        )
        meshGrid.SetCells(vtk.VTK_TETRA, cells)

        # Attach cell data to the UG so it propagates through the filter
        eArr = vtk.vtkDoubleArray()
        eArr.SetName("Enorm")
        eArr.SetNumberOfValues(nCells)
        eArr.FillComponent(0, 0.0)
        meshGrid.GetCellData().AddArray(eArr)

        condArr = vtk.vtkDoubleArray()
        condArr.SetName("conductivity")
        condArr.SetNumberOfValues(nCells)
        vtk.util.numpy_support.vtk_to_numpy(condArr)[:] = conductivity
        meshGrid.GetCellData().AddArray(condArr)

        if tag1 is not None:
            tagArr = vtk.vtkIntArray()
            tagArr.SetName("tag1")
            tagArr.SetNumberOfValues(nCells)
            vtk.util.numpy_support.vtk_to_numpy(tagArr)[:] = tag1
            meshGrid.GetCellData().AddArray(tagArr)

        # --- Extract surface polydata (one-time cost ~8 s for 4.4 M tets) ---
        log.info("loadMeshToScene: extracting surface ...")
        surfaceFilter = vtk.vtkGeometryFilter()
        surfaceFilter.SetInputData(meshGrid)
        surfaceFilter.PassThroughCellIdsOn()
        surfaceFilter.Update()
        surface = surfaceFilter.GetOutput()

        origIds = surface.GetCellData().GetArray("vtkOriginalCellIds")
        self._surfaceCellMap = vtk.util.numpy_support.vtk_to_numpy(origIds).copy()
        # Remove the bookkeeping array — not needed for display
        surface.GetCellData().RemoveArray("vtkOriginalCellIds")

        nSurface = surface.GetNumberOfCells()
        log.info(f"loadMeshToScene: {nCells} tets → {nSurface} surface triangles")

        # --- Create or reuse Slicer model node (hidden until E-field ready) ---
        existing = slicer.mrmlScene.GetFirstNodeByName("TMS E-field")
        if existing and existing.IsA("vtkMRMLModelNode"):
            modelNode = existing
            modelNode.SetAndObservePolyData(surface)
        else:
            modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
            modelNode.SetName("TMS E-field")
            modelNode.SetAndObservePolyData(surface)
            modelNode.CreateDefaultDisplayNodes()

        # Configure display but keep hidden — visibility enabled when E-field arrives
        dn = modelNode.GetDisplayNode()
        dn.SetVisibility(False)
        dn.SetAndObserveColorNodeID("vtkMRMLColorTableNodeFileViridis.txt")
        dn.SetActiveScalar("Enorm", vtk.vtkAssignAttribute.CELL_DATA)
        dn.SetScalarVisibility(True)
        dn.SetAutoScalarRange(True)

        self._meshNode = modelNode
        return modelNode

    # ------------------------------------------------------------------
    # Service lifecycle
    # ------------------------------------------------------------------

    def startService(self, meshPath, solver, port):
        """Launch TMSService.py as a QProcess subprocess.

        Kills zombie processes on the port and cleans orphaned shared
        memory before launching.
        """
        log.info(f"startService: mesh={meshPath}, solver={solver}, port={port}")
        if self._process is not None:
            self.stopService()
        if not os.path.isfile(_SERVICE_PATH):
            raise FileNotFoundError(
                f"TMSService.py not found at {_SERVICE_PATH}"
            )

        self._port = port

        # Zombie cleanup
        killed = _kill_processes_on_port(port)
        if killed:
            log.info(f"Killed {len(killed)} zombie(s) on port {port}: {killed}")
            time.sleep(1.0)  # OS needs time to release the port after SIGKILL

        # Orphaned shared-memory cleanup
        _cleanup_orphaned_shm(self._shmName)

        # Create QProcess and wire signals (SlicerParallelProcessing pattern)
        self._process = qt.QProcess()

        self._onStateChangedSlot = self._onProcessStateChanged
        self._process.connect(
            'stateChanged(QProcess::ProcessState)', self._onStateChangedSlot
        )

        self._onReadyReadOutSlot = self._onReadyReadStdout
        self._process.connect(
            'readyReadStandardOutput()', self._onReadyReadOutSlot
        )

        self._onReadyReadErrSlot = self._onReadyReadStderr
        self._process.connect(
            'readyReadStandardError()', self._onReadyReadErrSlot
        )

        self._onFinishedSlot = lambda exitCode, exitStatus: (
            self._onProcessFinished(exitCode, exitStatus)
        )
        self._process.connect(
            'finished(int,QProcess::ExitStatus)', self._onFinishedSlot
        )

        # Launch
        self._process.start(sys.executable, [
            _SERVICE_PATH,
            "--solver", solver,
            "--port",   str(port),
        ])

        if not self._process.waitForStarted(5000):
            error = _QProcessErrorNames.get(self._process.error(), "Unknown")
            raise RuntimeError(f"TMSService QProcess failed to start: {error}")
        log.info(f"QProcess started, PID={self._process.processId()}")

    # ------------------------------------------------------------------
    # QProcess signal handlers
    # ------------------------------------------------------------------

    def _onProcessStateChanged(self, newState):
        name = _QProcessStateNames.get(newState, f"Unknown({newState})")
        log.info(f"TMSService state: {name}")
        if self._process and self._process.error() != qt.QProcess.UnknownError:
            err = _QProcessErrorNames.get(self._process.error(), "?")
            log.warning(f"TMSService error: {err}")

    def _onReadyReadStdout(self):
        if self._process is None:
            return
        data = self._process.readAllStandardOutput()
        if data:
            needs_update = False
            for line in data.data().decode("utf-8", errors="replace").rstrip("\n").split("\n"):
                log.info(f"[TMSService] {line}")
                if (line.startswith("E_UPDATED") or line.startswith("STREAMING_READY")) \
                        and self._sharedEnorm is not None:
                    needs_update = True
            if needs_update:
                self._updateMeshColors()

    def _onReadyReadStderr(self):
        if self._process is None:
            return
        data = self._process.readAllStandardError()
        if data:
            for line in data.data().decode("utf-8", errors="replace").rstrip("\n").split("\n"):
                log.warning(f"[TMSService:stderr] {line}")

    def _onProcessFinished(self, exitCode, exitStatus):
        statusName = "NormalExit" if exitStatus == qt.QProcess.NormalExit else "CrashExit"
        log.info(
            f"TMSService finished: exitCode={exitCode}, status={statusName}"
        )
        self._onReadyReadStdout()
        self._onReadyReadStderr()
        self._disconnectProcessSignals()

    def _disconnectProcessSignals(self):
        """Safely disconnect all QProcess signal connections."""
        if self._process is None:
            return
        try:
            if self._onStateChangedSlot is not None:
                self._process.disconnect(
                    'stateChanged(QProcess::ProcessState)', self._onStateChangedSlot
                )
            if self._onReadyReadOutSlot is not None:
                self._process.disconnect(
                    'readyReadStandardOutput()', self._onReadyReadOutSlot
                )
            if self._onReadyReadErrSlot is not None:
                self._process.disconnect(
                    'readyReadStandardError()', self._onReadyReadErrSlot
                )
            if self._onFinishedSlot is not None:
                self._process.disconnect(
                    'finished(int,QProcess::ExitStatus)', self._onFinishedSlot
                )
        except Exception:
            pass
        self._onStateChangedSlot = None
        self._onReadyReadOutSlot = None
        self._onReadyReadErrSlot = None
        self._onFinishedSlot     = None

    def isServiceRunning(self):
        """Return True if the TMSService QProcess is alive."""
        return (
            self._process is not None
            and self._process.state() != qt.QProcess.NotRunning
        )

    # ------------------------------------------------------------------

    def connectToService(self, port, max_attempts=30):
        """Connect to running TMSService via RPyC; retry for up to 30 s.

        Fails fast if the QProcess exits before the connection succeeds.
        """
        log.info(f"connectToService: port={port}, max_attempts={max_attempts}")
        import rpyc
        for _ in range(max_attempts):
            if not self.isServiceRunning():
                if self._process is not None:
                    self._onReadyReadStdout()
                    self._onReadyReadStderr()
                raise RuntimeError(
                    f"TMSService subprocess died before accepting connections "
                    f"on port {port}. Check the Slicer log for details."
                )
            try:
                self._tms = rpyc.connect(
                    "localhost", port,
                    config={
                        "allow_public_attrs": True,
                        "allow_pickle":       True,
                        "sync_request_timeout": None,
                    },
                )
                log.info("RPyC connection established")
                return
            except ConnectionRefusedError:
                slicer.app.processEvents()
                time.sleep(1)
        raise RuntimeError(
            f"Could not connect to TMSService on port {port} after {max_attempts} s. "
            "Check that TMSService.py started without errors."
        )

    def initializeFEM(self, meshPath, solver):
        """Call initialize_system() on the service (may take several minutes)."""
        if self._tms is None:
            raise RuntimeError("Not connected to TMSService — call connectToService() first.")
        log.info(f"initializeFEM: mesh={meshPath}, solver={solver}")
        self._tms.root.initialize_system(meshPath or None, solver)
        log.info("initializeFEM: complete")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def setupVisualization(self):
        """Set up shared-memory E-field buffer and configure live display.

        If self._meshNode already has a valid mesh (from loadMeshToScene),
        reuses it.  Otherwise builds the grid from TMSService data.
        """
        log.info("setupVisualization: start")
        if self._tms is None:
            raise RuntimeError("Not connected to TMSService.")

        n_elements = int(self._tms.root.n_elements)
        log.info(f"setupVisualization: n_elements={n_elements}")

        # Reuse existing mesh node if loadMeshToScene() was called earlier;
        # otherwise build from service data (backward compat / direct start).
        if (self._meshNode is None
                or self._meshNode.GetMesh() is None
                or self._meshNode.GetMesh().GetNumberOfCells() == 0):
            log.info("setupVisualization: building grid from service data (no prior mesh)")
            nodes_mm = numpy.array(self._tms.root.nodes_mm)
            elements = numpy.array(self._tms.root.elements)

            meshGrid = vtk.vtkUnstructuredGrid()
            pts = vtk.vtkPoints()
            pts.SetNumberOfPoints(len(nodes_mm))
            vtk.util.numpy_support.vtk_to_numpy(pts.GetData())[:] = nodes_mm
            meshGrid.SetPoints(pts)

            nCells = elements.shape[0]
            offsets = numpy.arange(0, nCells * 4 + 1, 4, dtype=numpy.int64)
            connectivity = elements.ravel().astype(numpy.int64)
            cells = vtk.vtkCellArray()
            cells.SetData(
                vtk.util.numpy_support.numpy_to_vtk(offsets, deep=True),
                vtk.util.numpy_support.numpy_to_vtk(connectivity, deep=True),
            )
            meshGrid.SetCells(vtk.VTK_TETRA, cells)

            eArr = vtk.vtkDoubleArray()
            eArr.SetName("Enorm")
            eArr.SetNumberOfValues(nCells)
            meshGrid.GetCellData().AddArray(eArr)

            # Extract surface polydata
            log.info("setupVisualization: extracting surface ...")
            surfaceFilter = vtk.vtkGeometryFilter()
            surfaceFilter.SetInputData(meshGrid)
            surfaceFilter.PassThroughCellIdsOn()
            surfaceFilter.Update()
            surface = surfaceFilter.GetOutput()

            origIds = surface.GetCellData().GetArray("vtkOriginalCellIds")
            self._surfaceCellMap = vtk.util.numpy_support.vtk_to_numpy(origIds).copy()
            surface.GetCellData().RemoveArray("vtkOriginalCellIds")

            log.info(f"setupVisualization: {nCells} tets → "
                     f"{surface.GetNumberOfCells()} surface triangles")

            self._meshNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
            self._meshNode.SetName("TMS E-field")
            self._meshNode.SetAndObservePolyData(surface)
            self._meshNode.CreateDefaultDisplayNodes()

        log.info(f"setupVisualization: mesh node "
                 f"({self._meshNode.GetMesh().GetNumberOfCells()} cells)"
                 if self._meshNode and self._meshNode.GetMesh()
                    and self._meshNode.GetMesh().GetNumberOfCells() > 0
                 else "setupVisualization: mesh node ready")

        # Shared memory for scalar Enorm streaming (avoids RPyC serialisation)
        _fix_spawn_executable()
        self._cleanupSharedMemory()
        _cleanup_orphaned_shm(self._shmName)
        enorm_nbytes = n_elements * 8  # float64 scalar per element
        log.info(f"setupVisualization: creating shared memory '{self._shmName}', "
                 f"size={enorm_nbytes} bytes ({n_elements} elements)")
        self._shm = multiprocessing.shared_memory.SharedMemory(
            create=True, size=enorm_nbytes, name=self._shmName
        )
        self._sharedEnorm = numpy.ndarray(
            (n_elements,), dtype=numpy.float64, buffer=self._shm.buf
        )
        log.info("setupVisualization: shared memory created OK")

        # Ensure display is configured for Enorm (visibility deferred to first update)
        dn = self._meshNode.GetDisplayNode()
        dn.SetAndObserveColorNodeID("vtkMRMLColorTableNodeFileViridis.txt")
        dn.SetActiveScalar("Enorm", vtk.vtkAssignAttribute.CELL_DATA)
        dn.SetScalarVisibility(True)

        # Start streaming solve loop (non-blocking — runs in service process)
        import rpyc
        log.info("setupVisualization: starting streaming solve loop")
        self._streamingResult = rpyc.async_(
            self._tms.root.start_streaming
        )(self._shmName)

        # Create default probe transform if none exists
        probe = slicer.mrmlScene.GetFirstNodeByName("TMS Probe")
        if probe is None:
            probe = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLinearTransformNode", "TMS Probe"
            )
            probe.CreateDefaultDisplayNodes()
            probe.GetDisplayNode().SetEditorVisibility(True)
        self.setProbeTransformNode(probe)

    def setProbeTransformNode(self, node):
        """Attach / detach the E-field update observer to a transform node."""
        if self._probeNode is not None and self._probeObsTag is not None:
            self._probeNode.RemoveObserver(self._probeObsTag)
            self._probeObsTag = None

        self._probeNode = node
        if node is None or self._process is None:
            return

        self._probeObsTag = node.AddObserver(
            slicer.vtkMRMLTransformNode.TransformModifiedEvent,
            self._onProbeTransformModified,
        )
        node.TransformModified()   # trigger initial update

    def switchSolver(self, solver):
        """Switch solver backend without reloading the mesh."""
        if self._tms is None:
            raise RuntimeError("Not connected to TMSService.")
        self._tms.root.set_solver(solver)

    def stopService(self):
        """Disconnect from the service and terminate the QProcess subprocess.

        Uses graceful terminate() → waitForFinished() → kill() escalation.
        """
        log.info("stopService: begin teardown")
        if self._probeNode is not None and self._probeObsTag is not None:
            self._probeNode.RemoveObserver(self._probeObsTag)
            self._probeObsTag = None

        if self._tms is not None:
            try:
                self._tms.close()
            except Exception:
                pass
            self._tms = None

        if self._process is not None:
            # Signal the streaming solve loop to exit gracefully
            if self._process.state() != qt.QProcess.NotRunning:
                try:
                    self._process.write(b"STOP\n")
                    self._process.waitForBytesWritten(1000)
                except Exception:
                    pass

            self._disconnectProcessSignals()

            if self._process.state() != qt.QProcess.NotRunning:
                self._process.terminate()
                if not self._process.waitForFinished(3000):
                    self._process.kill()
                    self._process.waitForFinished(3000)

            # PythonSlicer is a wrapper that spawns python-real as a child.
            # QProcess.kill() only kills the wrapper, orphaning python-real.
            # Clean up orphaned children by port.
            if self._port is not None:
                _kill_processes_on_port(self._port)

            # Drain final output
            try:
                self._onReadyReadStdout()
                self._onReadyReadStderr()
            except Exception:
                pass

            self._process = None

        self._cleanupSharedMemory()
        self._port = None
        log.info("stopService: teardown complete")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _onProbeTransformModified(self, node, event):
        if self._process is None or self._meshNode is None:
            return
        try:
            node.GetMatrixTransformToParent(self._probeMatrix)
            mat = slicer.util.arrayFromVTKMatrix(self._probeMatrix)
            pos = mat[:3, 3]
            vals = " ".join(f"{v:.6f}" for v in mat.ravel())
            self._process.write(f"PROBE {vals}\n".encode("utf-8"))
            log.debug(f"probe moved: pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")
        except Exception as exc:
            log.error(f"updateEField error: {exc}", exc_info=True)

    def _updateMeshColors(self):
        eArray = slicer.util.arrayFromModelCellData(self._meshNode, "Enorm")
        if self._surfaceCellMap is not None:
            eArray[:] = self._sharedEnorm[self._surfaceCellMap]
        else:
            eArray[:] = self._sharedEnorm
        slicer.util.arrayFromModelCellDataModified(self._meshNode, "Enorm")
        dn = self._meshNode.GetDisplayNode()
        if not dn.GetVisibility():
            dn.SetVisibility(True)
        dn.Modified()

    def _cleanupSharedMemory(self):
        self._sharedEnorm = None
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            try:
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None


# ============================================================
# Self-test
# ============================================================

class SlicerTMSTest(ScriptedLoadableModuleTest):
    """Self-test: fully automatic — installs dependencies, generates test mesh,
    starts TMSService, runs FEM, verifies E-field.  No manual steps required.
    """

    def runTest(self):
        self.setUp()
        self.test_1_dependencies()
        self.test_2_prepare_mesh()
        self.test_3_service_and_fem()

    def setUp(self):
        slicer.mrmlScene.Clear()
        os.makedirs(_TEST_CACHE, exist_ok=True)

    # ------------------------------------------------------------------

    def test_1_dependencies(self):
        self.delayDisplay("Step 1 — installing / verifying dependencies …")
        SlicerTMSLogic().ensureDependencies()
        import rpyc    # noqa: F401
        import tmswarp # noqa: F401
        self.delayDisplay("  Dependencies OK ✓")

    def test_2_prepare_mesh(self):
        """Generate sphere3 test mesh if not already cached."""
        self.delayDisplay("Step 2 — preparing test mesh …")

        # Prefer the local checkout if it already has the file
        local = os.path.join(_TMSWARP_ROOT, "sphere3_data.npz")
        cached = os.path.join(_TEST_CACHE, "sphere3_data.npz")

        if os.path.isfile(local):
            self._sphere3Path = local
            self.delayDisplay(f"  Using local sphere3_data.npz ({os.path.getsize(local)//1024} KB) ✓")
            return

        if os.path.isfile(cached):
            self._sphere3Path = cached
            self.delayDisplay(f"  Using cached sphere3_data.npz ({os.path.getsize(cached)//1024} KB) ✓")
            return

        self.delayDisplay("  Generating sphere3 mesh (takes ~5 s) …")
        from tmswarp.conductor import make_sphere_mesh
        mesh = make_sphere_mesh(radius=0.09, n_shells=5, n_surface=200, conductivity=0.33)
        numpy.savez(cached,
                    nodes=mesh.nodes,
                    elements=mesh.elements,
                    conductivity=mesh.conductivity)
        self._sphere3Path = cached
        self.delayDisplay(
            f"  Generated sphere3: {mesh.nodes.shape[0]:,} nodes, "
            f"{mesh.elements.shape[0]:,} elements ✓"
        )

    def test_3_service_and_fem(self):
        self.delayDisplay("Step 3 — starting TMSService and running FEM …")
        logic = SlicerTMSLogic()
        port  = DEFAULT_PORT + 1    # avoid colliding with a running session

        try:
            logic.startService(self._sphere3Path, "numpy", port)
            self.delayDisplay("  Service started, connecting …")
            logic.connectToService(port)
            self.delayDisplay("  Connected.  Initializing FEM …")
            logic.initializeFEM(self._sphere3Path, "numpy")

            E    = numpy.array(logic._tms.root.E)
            Emag = numpy.linalg.norm(E, axis=1)
            self.assertEqual(E.ndim,    2, f"E.ndim={E.ndim}, expected 2")
            self.assertEqual(E.shape[1],3, f"E.shape[1]={E.shape[1]}, expected 3")
            self.assertGreater(Emag.max(), 0, "E-field magnitude is zero")
            self.delayDisplay(
                f"  FEM OK: {E.shape[0]:,} elements, |E|_max = {Emag.max():.3f} V/m ✓"
            )

            # Ernie smoke test — only if the .npz is already present locally
            ernie = os.path.join(_TMSWARP_ROOT, "ernie_data.npz")
            if os.path.isfile(ernie):
                self.delayDisplay(
                    f"  ernie_data.npz found ({os.path.getsize(ernie)//1024//1024} MB) — "
                    "running FEM (numpy pre-factorization may take several minutes) …"
                )
                logic.initializeFEM(ernie, "numpy")
                E_e    = numpy.array(logic._tms.root.E)
                Emag_e = numpy.linalg.norm(E_e, axis=1)
                self.assertGreater(Emag_e.max(), 0, "Ernie E-field is zero")
                self.delayDisplay(
                    f"  Ernie FEM OK: {E_e.shape[0]:,} elements, "
                    f"|E|_max = {Emag_e.max():.3f} V/m ✓"
                )
            else:
                self.delayDisplay(
                    "  ernie_data.npz not present — skipping ernie test.\n"
                    "  (Generate it with TMSWarp/scripts/fetch_ernie.py using SimNIBS Python.)"
                )

        finally:
            logic.stopService()

        self.delayDisplay("All tests passed ✓")
