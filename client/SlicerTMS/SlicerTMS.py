"""SlicerTMS: Interactive TMS E-field visualization using TMSWarp FEM.

Connects to TMSService.py (RPyC) running as a PythonSlicer subprocess.
Move the 'TMS Probe' linear transform to update the E-field in real time.
"""

import multiprocessing.shared_memory
import os
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
_SHARED_E_NAME = "tmsSharedE"
_TEST_CACHE    = os.path.join(os.path.expanduser("~"), ".cache", "SlicerTMS")


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

        # One-click ernie setup
        ernieBtn = qt.QPushButton("Setup Ernie Mesh + Probe  (downloads ~394 MB if needed)")
        ernieBtn.setToolTip(
            "Find or download ernie_data.npz, set the mesh path, "
            "create a TMS Probe transform, and select the best available solver."
        )
        ernieBtn.clicked.connect(self._setupErnieMesh)
        svcForm.addRow(ernieBtn)

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
            self.logic.stopService()

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

    def _setupErnieMesh(self):
        """Download ernie_data.npz if needed, set mesh path, create probe, pick best solver."""
        self._setStatus("Setting up ernie mesh (may download ~394 MB) …")
        try:
            path = self.logic.setupErnieMesh()
            self.meshPathEdit.currentPath = path
            self._setStatus(f"Ernie mesh ready: {os.path.basename(path)}")
        except Exception as exc:
            self._setStatus(f"Ernie setup failed: {exc}")
            slicer.util.errorDisplay(str(exc))
            return

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
            "Ernie ready. Click 'Start Service' to begin "
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
            # Select the auto-created probe in the combo
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
        self._process      = None   # subprocess handle for TMSService
        self._tms          = None   # RPyC connection
        self._shm          = None   # shared memory block
        self._sharedE      = None   # numpy view into shared memory
        self._meshNode     = None   # vtkMRMLModelNode for E-field display
        self._probeNode    = None   # vtkMRMLLinearTransformNode
        self._probeObsTag  = None   # observer tag
        self._probeMatrix  = vtk.vtkMatrix4x4()

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
    def setupErnieMesh():
        """Return path to ernie_data.npz, downloading and converting if needed.

        Search order:
          1. _TMSWARP_ROOT/ernie_data.npz  (local TMSWarp checkout)
          2. _TEST_CACHE/ernie_data.npz    (previously cached download)
          3. Download zip + convert via meshio (~394 MB, one-time)
        """
        _ERNIE_URL = (
            "https://github.com/simnibs/example-dataset/releases/"
            "download/v4.0-lowres/ernie_lowres_V2.zip"
        )
        _CONDUCTIVITY_MAP = {
            1: 0.126,   # white matter
            2: 0.275,   # gray matter
            3: 1.654,   # CSF
            4: 0.010,   # skull
            5: 0.465,   # scalp
            6: 0.500,   # eye balls
        }

        # 1. Local checkout
        local = os.path.join(_TMSWARP_ROOT, "ernie_data.npz")
        if os.path.isfile(local):
            return local

        # 2. Cache
        os.makedirs(_TEST_CACHE, exist_ok=True)
        cached = os.path.join(_TEST_CACHE, "ernie_data.npz")
        if os.path.isfile(cached):
            return cached

        # 3. Download + convert
        # Try SimNIBS Python first (has mesh_io, avoids meshio dependency)
        fetch_script = os.path.join(_TMSWARP_ROOT, "scripts", "fetch_ernie.py")
        simnibs_python = os.path.join(
            os.path.expanduser("~"), "Applications",
            "SimNIBS-4.5", "simnibs_env", "bin", "python",
        )
        if os.path.isfile(simnibs_python) and os.path.isfile(fetch_script):
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

    # ------------------------------------------------------------------
    # Service lifecycle
    # ------------------------------------------------------------------

    def startService(self, meshPath, solver, port):
        """Launch TMSService.py as a PythonSlicer subprocess."""
        if self._process is not None:
            self.stopService()
        if not os.path.isfile(_SERVICE_PATH):
            raise FileNotFoundError(
                f"TMSService.py not found at {_SERVICE_PATH}"
            )
        cmdList = [
            sys.executable, _SERVICE_PATH,
            "--solver", solver,
            "--port",   str(port),
        ]
        self._process = slicer.util.launchConsoleProcess(
            cmdList, useStartupEnvironment=False
        )

    def connectToService(self, port, max_attempts=30):
        """Connect to running TMSService via RPyC; retry for up to 30 s."""
        import rpyc
        for _ in range(max_attempts):
            try:
                self._tms = rpyc.connect(
                    "localhost", port,
                    config={
                        "allow_public_attrs": True,
                        "allow_pickle":       True,
                        "sync_request_timeout": None,
                    },
                )
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
        self._tms.root.initialize_system(meshPath or None, solver)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def setupVisualization(self):
        """Build VTK mesh model node and shared-memory E-field buffer."""
        if self._tms is None:
            raise RuntimeError("Not connected to TMSService.")

        nodes_mm = numpy.array(self._tms.root.nodes_mm)   # (N, 3) mm
        elements = numpy.array(self._tms.root.elements)   # (M, 4) 0-based
        E_ref    = numpy.array(self._tms.root.E)           # (M, 3) initial E

        # Build VTK unstructured tetrahedral grid
        meshGrid = vtk.vtkUnstructuredGrid()
        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(len(nodes_mm))
        vtk.util.numpy_support.vtk_to_numpy(pts.GetData())[:] = nodes_mm
        meshGrid.SetPoints(pts)

        offsets = numpy.arange(0, elements.shape[0] * 4 + 1, 4, dtype=numpy.int64)
        cells   = vtk.vtkCellArray()
        cells.SetData(
            vtk.util.numpy_support.numpy_to_vtk(offsets, deep=True),
            vtk.util.numpy_support.numpy_to_vtk(elements.ravel(), deep=True),
        )
        meshGrid.SetCells(vtk.VTK_TETRA, cells)

        eArr = vtk.vtkDoubleArray()
        eArr.SetNumberOfValues(elements.shape[0])
        eArr.SetName("Enorm")
        meshGrid.GetCellData().AddArray(eArr)

        # Create Slicer model node
        self._meshNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        self._meshNode.SetName("TMS E-field")
        self._meshNode.SetAndObserveMesh(meshGrid)
        self._meshNode.CreateDefaultDisplayNodes()
        dn = self._meshNode.GetDisplayNode()
        dn.SetAndObserveColorNodeID("vtkMRMLColorTableNodeFileViridis.txt")
        dn.SetScalarVisibility(True)
        dn.SetActiveScalar("Enorm", vtk.vtkAssignAttribute.CELL_DATA)

        # Shared memory for fast E-field streaming (avoids RPyC serialisation)
        self._cleanupSharedMemory()
        self._shm = multiprocessing.shared_memory.SharedMemory(
            create=True, size=E_ref.nbytes, name=_SHARED_E_NAME
        )
        self._sharedE = numpy.ndarray(
            E_ref.shape, dtype=E_ref.dtype, buffer=self._shm.buf
        )

        # Paint initial E-field
        self._tms.root.copy_E_to_share(_SHARED_E_NAME)
        self._updateMeshColors()

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
        if node is None or self._tms is None:
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
        """Disconnect from the service and kill the subprocess."""
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
            try:
                self._process.kill()
            except Exception:
                pass
            self._process = None

        self._cleanupSharedMemory()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _onProbeTransformModified(self, node, event):
        if self._tms is None or self._meshNode is None:
            return
        try:
            node.GetMatrixTransformToParent(self._probeMatrix)
            mat = slicer.util.arrayFromVTKMatrix(self._probeMatrix)
            self._tms.root.update_E_field(mat.tolist())
            self._tms.root.copy_E_to_share(_SHARED_E_NAME)
            self._updateMeshColors()
        except Exception as exc:
            print(f"[SlicerTMS] updateEField error: {exc}")

    def _updateMeshColors(self):
        eArray = slicer.util.arrayFromModelCellData(self._meshNode, "Enorm")
        eArray[:] = numpy.linalg.norm(self._sharedE, axis=1)
        slicer.util.arrayFromModelCellDataModified(self._meshNode, "Enorm")
        self._meshNode.GetDisplayNode().Modified()

    def _cleanupSharedMemory(self):
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None
        self._sharedE = None


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
