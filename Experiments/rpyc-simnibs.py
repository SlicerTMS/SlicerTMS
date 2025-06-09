"""
/Users/pieper/Applications/SimNIBS-4.5/bin/simnibs_python -m pip install rpyc
"""

onlinefemPath = "/Users/pieper/slicer/latest/SlicerTMS/Experiments/onlinefem.py"
onlinefemPath = "/media/share/tms-work/SlicerTMS/Experiments/onlinefem.py"

simnibs_pythonPath = "/Users/pieper/Applications/SimNIBS-4.5/bin/simnibs_python"
simnibs_pythonPath = "/media/share/tms-work/SimNIBS/install/bin/simnibs_python"

rpyc_path = "/Users/pieper/Applications/SimNIBS-4.5/simnibs_env/bin/rpyc_classic"
rpyc_path = "/media/share/tms-work/SimNIBS/install/simnibs_env/bin/rpyc_classic"

need_to_fix_pipe = """
try:
    process
except NameError:
    pass
else:
    process.kill()

process = slicer.util.launchConsoleProcess([rpyc_path], useStartupEnvironment=True)

line = process.stdout.readline().strip()
port = int(line.split(":")[-1])
"""

port = 18812


import numpy
import rpyc

simnibs = rpyc.classic.connect("localhost", port)

script = open(onlinefemPath).read()
simnibs.execute(script)
ofem = simnibs.eval("globals()")["ofem"]

print("Simulation setup complete")
slicer.app.processEvents()

m = simnibs.eval("m")
nodeCoords = numpy.array(m.nodes.node_coord)
elementIndices = numpy.array(m.elm.node_number_list) - 1

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

eArray = slicer.util.arrayFromModelCellData(meshNode, "Enorm")
eArray[:] = numpy.linalg.norm(numpy.array(simnibs.eval("E"))[:-1], axis=1)
slicer.util.arrayFromModelCellDataModified(meshNode, "Enorm")

# this should be working, but if we call this the scalars don't show and can't be turned on manually
#meshNode.GetDisplayNode().SetAndObserveColorNodeID('vtkMRMLColorTableNodeFileViridis.txt')
#meshNode.GetDisplayNode().SetActiveScalarName("Enorm")
#meshNode.GetDisplayNode().SetScalarVisibility(True)

probeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
probeNode.SetName("Probe")
probeNode.CreateDefaultDisplayNodes()
probeNode.GetDisplayNode().SetEditorVisibility(True)

probeMatrix = vtk.vtkMatrix4x4()
def updateEField(transformNode, event):
    probeNode.GetMatrixTransformToParent(probeMatrix)
    probeMatrixArray = slicer.util.arrayFromVTKMatrix(probeMatrix)
    E = numpy.frombuffer(update_field(probeMatrixArray.tolist()))
    eArray = slicer.util.arrayFromModelCellData(meshNode, "Enorm")
    E = E.reshape((eArray.shape[0]+1,3))
    eArray[:] = numpy.linalg.norm(numpy.array(E)[:-1], axis=1)
    slicer.util.arrayFromModelCellDataModified(meshNode, "Enorm")
    meshNode.GetDisplayNode().Modified()

probeNode.AddObserver(slicer.vtkMRMLTransformNode.TransformModifiedEvent, updateEField)

update_field = simnibs.eval("globals()")["update_field"]
def animate(iterations=300):
    for i in range(iterations):
        coilPosition = numpy.identity(4)
        coilPosition[0,3] = 2*i

        #ofem = simnibs.eval("globals()")["ofem"]
        #E = ofem.update_field(matsimnibs=coilPosition.data, didt=1e6)[0][0]


        E = numpy.frombuffer(update_field(coilPosition.tolist()))

        eArray = slicer.util.arrayFromModelCellData(meshNode, "Enorm")
        E = E.reshape((eArray.shape[0]+1,3))
        eArray[:] = numpy.linalg.norm(numpy.array(E)[:-1], axis=1)
        slicer.util.arrayFromModelCellDataModified(meshNode, "Enorm")
        meshNode.GetDisplayNode().Modified()
        slicer.app.processEvents()
        if i % 10 == 0:
            print(i)
            slicer.app.processEvents()
