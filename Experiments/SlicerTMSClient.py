"""Slicer client for the TMSWarp interactive E-field service.

Connects to TMSService.py and visualizes the E-field on the head mesh
as the coil (probe transform) is moved interactively.

Usage
-----
Run this script in the Slicer Python console (Script Editor):

    exec(open("/path/to/SlicerTMS/Experiments/SlicerTMSClient.py").read())

What it does
------------
1. Starts TMSService.py using the TMSWarp pixi Python
2. Connects via RPyC on port 18892
3. Loads the mesh from the service and builds a VTK tetrahedral grid
4. Adds a "TMS Probe" linear transform node — move it to update the coil
5. Displays |E| as a coloured cell scalar (Viridis)

Switching solvers interactively
--------------------------------
From the Slicer Python console after this script has run:

    tms.root.set_solver("numpy")      # pre-factorized scipy — seconds/update
    tms.root.set_solver("warp_cpu")   # Warp CG on CPU       — ~2 min/update
    tms.root.set_solver("warp_gpu")   # Warp CG on GPU       — fast if CUDA
"""

import multiprocessing.shared_memory
import numpy
import time

try:
    import rpyc
except ModuleNotFoundError:
    import pip
    pip.main("install rpyc".split())
    import rpyc

# ---------------------------------------------------------------------------
# Configuration — edit these paths for your machine
# ---------------------------------------------------------------------------

# Python executable inside the TMSWarp pixi environment
tms_python = (
    "/Users/pieper/slicer/latest/SlicerTMS/TMSWarp"
    "/.pixi/envs/default/bin/python"
)

# Path to TMSService.py
tms_service_path = (
    "/Users/pieper/slicer/latest/SlicerTMS/Experiments/TMSService.py"
)

# Mesh file: None = auto-detect ernie_data.npz (or sphere3_data.npz fallback)
mesh_path = None

# Starting solver backend
solver = "numpy"   # "numpy" | "warp_cpu" | "warp_gpu"

port = 18892

# ---------------------------------------------------------------------------
# (Re)start the service
# ---------------------------------------------------------------------------

slicer.mrmlScene.Clear()

try:
    process.kill()
    time.sleep(0.5)
except NameError:
    pass

cmdList = [tms_python, tms_service_path,
           "--solver", solver, "--port", str(port)]
process = slicer.util.launchConsoleProcess(cmdList, useStartupEnvironment=False)

print("Waiting for TMSService to start ...")
for attempt in range(30):
    try:
        tms = rpyc.connect(
            "localhost", port,
            config={
                "allow_public_attrs": True,
                "allow_pickle": True,
                "sync_request_timeout": None,   # no timeout — solves can be slow
            },
        )
        print(f"Connected after {attempt+1} attempt(s).")
        break
    except ConnectionRefusedError:
        slicer.app.processEvents()
        time.sleep(1)
else:
    raise RuntimeError("Could not connect to TMSService after 30 s. "
                       "Check that the path to tms_python is correct.")

# ---------------------------------------------------------------------------
# Initialize FEM system
# ---------------------------------------------------------------------------

print(f"Initializing FEM system (solver={solver}) ...")
print("  [ernie mesh: ~3-5 min for initial LU factorization (numpy)]")
slicer.app.processEvents()
tms.root.initialize_system(mesh_path, solver)
print("Initialization complete.")
slicer.app.processEvents()

# ---------------------------------------------------------------------------
# Build VTK unstructured mesh in Slicer
# ---------------------------------------------------------------------------

nodeCoords     = numpy.array(tms.root.nodes_mm)   # (N, 3) float64  mm
elementIndices = numpy.array(tms.root.elements)   # (M, 4) int32  0-based

meshGrid   = vtk.vtkUnstructuredGrid()
gridPoints = vtk.vtkPoints()
gridPoints.SetNumberOfPoints(len(nodeCoords))
vtk.util.numpy_support.vtk_to_numpy(gridPoints.GetData())[:] = nodeCoords
meshGrid.SetPoints(gridPoints)

offsetsArray     = numpy.arange(0, elementIndices.shape[0]*4+1, 4,
                                dtype=numpy.int64)
gridCellVTKArray = vtk.vtkCellArray()
gridCellVTKArray.SetData(
    vtk.util.numpy_support.numpy_to_vtk(offsetsArray, deep=True),
    vtk.util.numpy_support.numpy_to_vtk(elementIndices.ravel(), deep=True),
)
meshGrid.SetCells(vtk.VTK_TETRA, gridCellVTKArray)

eVTKArray = vtk.vtkDoubleArray()
eVTKArray.SetNumberOfValues(elementIndices.shape[0])
eVTKArray.SetName("Enorm")
meshGrid.GetCellData().AddArray(eVTKArray)

meshNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
meshNode.SetAndObserveMesh(meshGrid)
meshNode.CreateDefaultDisplayNodes()
meshNode.GetDisplayNode().SetAndObserveColorNodeID(
    "vtkMRMLColorTableNodeFileViridis.txt"
)
meshNode.GetDisplayNode().SetScalarVisibility(True)
meshNode.GetDisplayNode().SetActiveScalar(
    "Enorm", vtk.vtkAssignAttribute.CELL_DATA
)

# ---------------------------------------------------------------------------
# Shared memory for fast E-field transfer (avoids slow RPyC serialisation)
# ---------------------------------------------------------------------------

try:
    sharedMemoryForE.close()
    sharedMemoryForE.unlink()
except (NameError, FileNotFoundError):
    pass

E_ref            = numpy.array(tms.root.E)    # (M, 3)  reference shape+dtype
sharedEName      = "tmsSharedE"
sharedMemoryForE = multiprocessing.shared_memory.SharedMemory(
    create=True, size=E_ref.nbytes, name=sharedEName
)
sharedE = numpy.ndarray(E_ref.shape, dtype=E_ref.dtype,
                        buffer=sharedMemoryForE.buf)

# Paint initial E-field
tms.root.copy_E_to_share(sharedEName)
eArray = slicer.util.arrayFromModelCellData(meshNode, "Enorm")
eArray[:] = numpy.linalg.norm(sharedE, axis=1)
slicer.util.arrayFromModelCellDataModified(meshNode, "Enorm")
meshNode.GetDisplayNode().Modified()

# ---------------------------------------------------------------------------
# Probe transform observer — drives coil position
# ---------------------------------------------------------------------------

probeMatrix = vtk.vtkMatrix4x4()

def updateEField(transformNode, event):
    """Called whenever the TMS Probe transform is moved."""
    try:
        probeNode.GetMatrixTransformToParent(probeMatrix)
        matArray = slicer.util.arrayFromVTKMatrix(probeMatrix)
        tms.root.update_E_field(matArray.tolist())
        tms.root.copy_E_to_share(sharedEName)
        eArray = slicer.util.arrayFromModelCellData(meshNode, "Enorm")
        eArray[:] = numpy.linalg.norm(sharedE, axis=1)
        slicer.util.arrayFromModelCellDataModified(meshNode, "Enorm")
        meshNode.GetDisplayNode().Modified()
    except Exception as exc:
        print(f"updateEField error: {exc}")

probeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
probeNode.SetName("TMS Probe")
probeNode.CreateDefaultDisplayNodes()
probeNode.GetDisplayNode().SetEditorVisibility(True)
probeNode.AddObserver(
    slicer.vtkMRMLTransformNode.TransformModifiedEvent, updateEField
)
probeNode.TransformModified()

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

print("""
TMSWarp interactive session ready.

Move the 'TMS Probe' transform in Slicer to update the E-field.
The coil position comes from the transform's translation (mm);
the coil normal comes from the transform's Z-column.

Switch solver from the Python console:
    tms.root.set_solver("numpy")      # pre-factorized scipy (seconds/update)
    tms.root.set_solver("warp_cpu")   # Warp CG on CPU  (~2 min/update)
    tms.root.set_solver("warp_gpu")   # Warp CG on GPU  (fast if CUDA)

Compare two solvers on the same coil position:
    tms.root.set_solver("numpy")
    E_numpy = numpy.array(tms.root.E)
    tms.root.set_solver("warp_gpu")
    E_gpu   = numpy.array(tms.root.E)
    print("RDM numpy vs GPU:", numpy.linalg.norm(
        E_gpu/numpy.linalg.norm(E_gpu) - E_numpy/numpy.linalg.norm(E_numpy)
    ))
""")
