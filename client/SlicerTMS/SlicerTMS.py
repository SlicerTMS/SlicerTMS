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

        # Solver combo — detect GPU and set as default if available
        self.solverCombo = qt.QComboBox()
        self.solverCombo.addItem("NumPy  —  pre-factorized LU,  CPU",    "numpy")
        self.solverCombo.addItem("Warp   —  CG iterative,       CPU",    "warp_cpu")
        self.solverCombo.addItem("Warp   —  CG iterative,       GPU / CUDA", "warp_gpu")
        defaultSolverIdx = 2 if self.logic.hasCudaGpu() else 0
        self.solverCombo.setCurrentIndex(defaultSolverIdx)
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
        self.liveSolverCombo.addItem("NumPy  —  pre-factorized LU,  CPU",    "numpy")
        self.liveSolverCombo.addItem("Warp   —  CG iterative,       CPU",    "warp_cpu")
        self.liveSolverCombo.addItem("Warp   —  CG iterative,       GPU / CUDA", "warp_gpu")
        self.liveSolverCombo.setCurrentIndex(defaultSolverIdx)
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

        # Auto-detect mesh on startup
        self._autoDetectMesh()

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

    def _installDependencies(self):
        self._setStatus("Installing dependencies …")
        try:
            self.logic.ensureDependencies()
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
    """Self-test: check dependencies, ernie availability, and FEM service.

    The FEM service test uses sphere3_data.npz (small synthetic mesh, fast)
    so the test completes in under a minute.  Ernie availability is checked
    but not required for the test to pass.
    """

    def runTest(self):
        self.setUp()
        self.test_1_dependencies()
        self.test_2_mesh_availability()
        self.test_3_service_and_fem()

    def setUp(self):
        slicer.mrmlScene.Clear()

    # ------------------------------------------------------------------

    def test_1_dependencies(self):
        self.delayDisplay("Step 1 — checking / installing dependencies …")
        logic = SlicerTMSLogic()
        logic.ensureDependencies()
        import rpyc  # noqa: F401 — must not raise
        self.delayDisplay("  rpyc available ✓")

    def test_2_mesh_availability(self):
        self.delayDisplay("Step 2 — checking mesh files …")
        sphere3 = os.path.join(_TMSWARP_ROOT, "sphere3_data.npz")
        ernie   = os.path.join(_TMSWARP_ROOT, "ernie_data.npz")

        if not os.path.isfile(sphere3):
            raise Exception(
                f"sphere3_data.npz not found at {sphere3}.\n"
                "Run TMSWarp tests first:  cd TMSWarp && pixi run pytest"
            )
        self.delayDisplay(f"  sphere3_data.npz ✓  ({os.path.getsize(sphere3)//1024} KB)")

        if os.path.isfile(ernie):
            self.delayDisplay(
                f"  ernie_data.npz  ✓  ({os.path.getsize(ernie)//1024//1024} MB)"
            )
        else:
            self.delayDisplay(
                "  ernie_data.npz  — not found (optional).\n"
                "  To generate: run TMSWarp/scripts/fetch_ernie.py with SimNIBS Python.\n"
                "  Download URL: https://github.com/simnibs/example-dataset/"
                "releases/download/v4.0-lowres/ernie_lowres_V2.zip"
            )

    def test_3_service_and_fem(self):
        self.delayDisplay("Step 3 — starting TMSService with sphere3 mesh …")
        logic = SlicerTMSLogic()
        sphere3 = os.path.join(_TMSWARP_ROOT, "sphere3_data.npz")
        port    = DEFAULT_PORT + 1   # avoid colliding with a running session

        try:
            logic.ensureDependencies()
            logic.startService(sphere3, "numpy", port)
            self.delayDisplay("  Service process started, connecting …")
            logic.connectToService(port)
            self.delayDisplay("  Connected.  Initializing FEM …")
            logic.initializeFEM(sphere3, "numpy")

            E = numpy.array(logic._tms.root.E)
            self.assertIsNotNone(E, "E-field is None")
            self.assertEqual(E.ndim, 2, f"E.ndim expected 2, got {E.ndim}")
            self.assertEqual(E.shape[1], 3, f"E.shape[1] expected 3, got {E.shape[1]}")
            Emag = numpy.linalg.norm(E, axis=1)
            self.assertGreater(Emag.max(), 0, "E-field magnitude is zero")

            self.delayDisplay(
                f"  FEM result: {E.shape[0]} elements, "
                f"|E|_max = {Emag.max():.3f} V/m  ✓"
            )

            # Quick ernie smoke test if the mesh is present
            ernie = os.path.join(_TMSWARP_ROOT, "ernie_data.npz")
            if os.path.isfile(ernie):
                self.delayDisplay(
                    "  ernie_data.npz found — verifying service can load it "
                    "(this may take several minutes) …"
                )
                logic.initializeFEM(ernie, "numpy")
                E_ernie = numpy.array(logic._tms.root.E)
                self.assertGreater(
                    numpy.linalg.norm(E_ernie, axis=1).max(), 0,
                    "Ernie E-field magnitude is zero"
                )
                self.delayDisplay(
                    f"  Ernie FEM result: {E_ernie.shape[0]} elements, "
                    f"|E|_max = {numpy.linalg.norm(E_ernie, axis=1).max():.3f} V/m  ✓"
                )

        finally:
            logic.stopService()

        self.delayDisplay("All tests passed ✓")
