"""
/Users/pieper/Applications/SimNIBS-4.5/bin/simnibs_python -m pip install rpyc
"""

simnibs_python_path = "/Users/pieper/Applications/SimNIBS-4.5/bin/simnibs_python"

rpyc_path = "/Users/pieper/Applications/SimNIBS-4.5/simnibs_env/bin/rpyc_classic"

process = slicer.util.launchConsoleProcess([rpyc_path], useStartupEnvironment=True)

line = process.stdout.readline().strip()
port = int(line.split(":")[-1])


import rpyc

simnibs = rpyc.classic.connect("localhost", port)

script = open("/Users/pieper/slicer/latest/SlicerTMS/Experiments/onlinefem.py").read()
simnibs.execute(script)

print("Simulation setup complete")
slicer.app.processEvents()

m = simnibs.eval("m")
nodeCoords = m.nodes.node_coord
elementIndices = m.elm.node_number_list

meshGrid = vtk.vtkUnstructuredGrid()



meshNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
meshNode.CreateDefaultDisplayNodes()
