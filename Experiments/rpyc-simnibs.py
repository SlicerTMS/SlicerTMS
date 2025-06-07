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

offsetsArray = numpy.arange(0, countsArray.shape[0] * 4, 4, dtype=numpy.int64)
gridCellVTKArray = vtk.vtkCellArray()
offsetsVTKArray = vtk.util.numpy_support.numpy_to_vtk(offsetsArray, deep=True)
tetraIndexVTKArray = vtk.util.numpy_support.numpy_to_vtk(elementIndices.ravel(), deep=True)
gridCellVTKArray.SetData(offsetsVTKArray, tetraIndexVTKArray)
meshGrid.SetCells(vtk.VTK_TETRA, gridCellVTKArray)

meshNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
meshNode.SetAndObserveMesh(meshGrid)
meshNode.CreateDefaultDisplayNodes()
