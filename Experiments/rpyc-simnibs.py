"""
/Users/pieper/Applications/SimNIBS-4.5/bin/simnibs_python -m pip install rpyc
"""

simnibs_python_path = "/Users/pieper/Applications/SimNIBS-4.5/bin/simnibs_python"

rpyc_path = "/Users/pieper/Applications/SimNIBS-4.5/simnibs_env/bin/rpyc_classic"

try:
    process
except NameError:
    pass
else:
    process.kill()

process = slicer.util.launchConsoleProcess([rpyc_path], useStartupEnvironment=True)

line = process.stdout.readline().strip()
port = int(line.split(":")[-1])


import numpy
import rpyc

simnibs = rpyc.classic.connect("localhost", port)

script = open("/Users/pieper/slicer/latest/SlicerTMS/Experiments/onlinefem.py").read()
simnibs.execute(script)

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


def animate():
    for i in range(300):
        coilPosition = numpy.identity(4)
        coilPosition[0,3] = 2*i

        #ofem = simnibs.eval("globals()")["ofem"]
        #E = ofem.update_field(matsimnibs=coilPosition.data, didt=1e6)[0][0]

        update_field = simnibs.eval("globals()")["update_field"]

        E = update_field(coilPosition.tolist())

        eArray = slicer.util.arrayFromModelCellData(meshNode, "Enorm")
        eArray[:] = numpy.linalg.norm(numpy.array(E)[:-1], axis=1)
        slicer.util.arrayFromModelCellDataModified(meshNode, "Enorm")
        meshNode.GetDisplayNode().Modified()
        slicer.app.processEvents()
