
import numpy
import time

try:
	import rpyc
except ModuleNotFoundError:
	import pip
	pip.main("install rpyc")
	import rpyc

simnibs_pythonPath = "/Users/pieper/Applications/SimNIBS-4.5/bin/simnibs_python"
simnibs_pythonPath = "/media/share/tms-work/SimNIBS/install/bin/simnibs_python"

SimNIBSServicePath = "/Users/pieper/slicer/latest/SlicerTMS/Experiments/SimNIBSService.py"
SimNIBSServicePath = "/media/share/tms-work/SlicerTMS/Experiments/SimNIBSService.py"



try:
    process
except NameError:
    pass
else:
    process.kill()

cmdList = [simnibs_pythonPath, SimNIBSServicePath]
process = slicer.util.launchConsoleProcess(cmdList, useStartupEnvironment=True)
time.sleep(1)

port = 18891

for attempt in range(10):
	try:
		simnibs = rpyc.connect("localhost", port,
			config = {
				"allow_public_attrs": True,
				"allow_pickle": True,
				"sync_request_timeout": None
			}
		)
		break
	except ConnectionRefusedError:
		time.sleep(1)


print("initialize_system")
slicer.app.processEvents()
simnibs.root.initialize_system()


print("Simulation setup complete")
slicer.app.processEvents()

nodeCoords = numpy.array(simnibs.root.mesh.nodes.node_coord)
elementIndices = numpy.array(simnibs.root.mesh.elm.node_number_list) - 1

meshGrid = vtk.vtkUnstructuredGrid()

gridPoints = vtk.vtkPoints()
gridPoints.SetNumberOfPoints(len(nodeCoords))
gridPointsArray = vtk.util.numpy_support.vtk_to_numpy(gridPoints.GetData())
gridPointsArray[:] = nodeCoords
meshGrid.SetPoints(gridPoints)

offsetsArray = numpy.arange(0, elementIndices.shape[0] * 4 + 1, 4, dtype=numpy.int64)
gridCellVTKArray = vtk.vtkCellArray()
offsetsVTKArray = vtk.util.numpy_support.numpy_to_vtk(offsetsArray, deep=True)
tetraIndexVTKArray = vtk.util.numpy_support.numpy_to_vtk(elementIndices.ravel(), deep=True)
gridCellVTKArray.SetData(offsetsVTKArray, tetraIndexVTKArray)
meshGrid.SetCells(vtk.VTK_TETRA, gridCellVTKArray)

eVTKArray = vtk.vtkDoubleArray()
eVTKArray.SetNumberOfValues(elementIndices.shape[0])
eVTKArray.SetName("Enorm")
meshGrid.GetCellData().AddArray(eVTKArray)

meshNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
meshNode.SetAndObserveMesh(meshGrid)
meshNode.CreateDefaultDisplayNodes()


# direct E
eArray = slicer.util.arrayFromModelCellData(meshNode, "Enorm")
eArray[:] = numpy.linalg.norm(numpy.array(simnibs.root.E)[:-1], axis=1)
slicer.util.arrayFromModelCellDataModified(meshNode, "Enorm")

probeMatrix = vtk.vtkMatrix4x4()
def updateEField(transformNode, event):
    probeNode.GetMatrixTransformToParent(probeMatrix)
    probeMatrixArray = slicer.util.arrayFromVTKMatrix(probeMatrix)
    simnibs.root.update_E_field(probeMatrixArray.tolist())
    eArray = slicer.util.arrayFromModelCellData(meshNode, "Enorm")
    eArray[:] = numpy.linalg.norm(numpy.array(simnibs.root.E)[:-1], axis=1)
    slicer.util.arrayFromModelCellDataModified(meshNode, "Enorm")
    meshNode.GetDisplayNode().Modified()

needed = """ ?
eArray = slicer.util.arrayFromModelCellData(meshNode, "Enorm")
probeMatrixArray = slicer.util.arrayFromVTKMatrix(probeMatrix)
E = numpy.frombuffer(update_field(probeMatrixArray.tolist()))
E = E.reshape((eArray.shape[0]+1,3))
eArray[:] = numpy.linalg.norm(numpy.array(simnibs_eval("E"))[:-1], axis=1)
slicer.util.arrayFromModelCellDataModified(meshNode, "Enorm")
"""

# this should be working, but if we call this the scalars don't show and can't be turned on manually
#meshNode.GetDisplayNode().SetAndObserveColorNodeID('vtkMRMLColorTableNodeFileViridis.txt')
#meshNode.GetDisplayNode().SetActiveScalarName("Enorm")
#meshNode.GetDisplayNode().SetScalarVisibility(True)

probeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
probeNode.SetName("Probe")
probeNode.CreateDefaultDisplayNodes()
probeNode.GetDisplayNode().SetEditorVisibility(True)

probeNode.AddObserver(slicer.vtkMRMLTransformNode.TransformModifiedEvent, updateEField)
