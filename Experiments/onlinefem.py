
"""
conda activate simnibs
python

exec(open("/media/share/tms-work/SlicerTMS/Experiments/onlinefem.py").read())

"""


import os
import pytest
import numpy as np

from simnibs.optimization.tes_flex_optimization.electrode_layout import ElectrodeArrayPair
from simnibs.simulation import analytical_solutions, onlinefem
from simnibs.simulation import tms_coil
import sys
from simnibs.mesh_tools import mesh_io
from simnibs import SIMNIBSDIR
from simnibs.simulation.onlinefem import OnlineFEM, FemTargetPointCloud


fn = os.path.join(SIMNIBSDIR, '_internal_resources', 'testing_files', 'sphere3.msh')
sphere3_msh = mesh_io.read_msh(fn)

m = sphere3_msh.crop_mesh(elm_type=4)
dipole_pos = np.array([0., 0., 300])
dipole_moment = np.array([1., 0., 0.])
didt = 1e6
r = (m.nodes.node_coord - dipole_pos) * 1e-3
dAdt = 1e-7 * didt * np.cross(dipole_moment, r) / (np.linalg.norm(r, axis=1)[:, None] ** 3)
dAdt = mesh_io.NodeData(dAdt, mesh=m)
dAdt.field_name = 'dAdt'
dAdt.mesh = m
pos = m.elements_baricenters().value
E_analytical = analytical_solutions.tms_E_field(dipole_pos * 1e-3,
                                                dipole_moment, didt,
                                                pos * 1e-3)
cond = mesh_io.ElementData(np.ones(m.elm.nr))
cond.mesh = m
stimulator = tms_coil.tms_stimulator.TmsStimulator("Example Stimulator", "Example Stimulator Brand", 100e6)
dipole_elm = tms_coil.tms_coil_element.DipoleElements(stimulator=stimulator, points=dipole_pos[None], values=dipole_moment[None])
coil = tms_coil.tms_coil.TmsCoil([dipole_elm])


solver_type = "pardiso"

fill=False
nearest=True
useElements=True


# create the ROI
center_points = m.elements_baricenters().value
out_point = np.array((0, 0, 100))[None]
center_points = np.concatenate((center_points, out_point), axis=0)
point_cloud = onlinefem.FemTargetPointCloud(m, center_points, nearest_neighbor=nearest, fill_nearest=fill)
            
#prepare and setup OnlineFEM
ofem = onlinefem.OnlineFEM(m, 'TMS', roi=[point_cloud], coil=coil, useElements=useElements, solver_options=solver_type, cond=cond)
    
#calculate vector E-field
ofem.dataType = [1]
#Solve the FEM
E = ofem.update_field(matsimnibs=np.identity(4), didt=1e6)[0][0]        
    
tests = """
assert rdm(E[:-1,:], E_analytical) < .2
assert np.abs(mag(E[:-1,:], E_analytical)) < np.log(1.1)
    
if fill:
    #find nearest point to the extra one outside
    nearest_idx = np.argmin(np.sqrt(((
        m.elements_baricenters().value - out_point)**2).sum(axis=1)))
    if nearest:
        assert np.all(E[-1, :] == E[nearest_idx])
    else:
        assert ofem.roi[0].sF[-1, :].sum() == 1
        assert ofem.roi[0].sF[-1, nearest_idx] == 1
else:
    assert np.all(E[-1, :] == 0)
"""
