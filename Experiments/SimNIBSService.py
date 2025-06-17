import numpy as np
import multiprocessing.shared_memory
import os
import pytest

try:
	import rpyc
except ModuleNotFoundError:
	import pip
	pip.main("install rpyc")
	import rpyc

from simnibs.optimization.tes_flex_optimization.electrode_layout import ElectrodeArrayPair
from simnibs.simulation import analytical_solutions, onlinefem
from simnibs.simulation import tms_coil
import sys
from simnibs.mesh_tools import mesh_io
from simnibs import SIMNIBSDIR
from simnibs.simulation.onlinefem import OnlineFEM, FemTargetPointCloud


class SimNIBSService(rpyc.SlaveService):
	def __init__(self):
		self.sharedE = None

	def initialize_system(self, mesh_path=None, solver="mumps", cpus=None):
		if not mesh_path:
			mesh_path = os.path.join(SIMNIBSDIR, '_internal_resources', 'testing_files', 'sphere3.msh')
			#fn = '/media/share/tms-work/SimNIBS/data/m2m_MNI152/MNI152.msh'
		self.mesh = mesh_io.read_msh(mesh_path)

		if not cpus:
			cpus = os.cpu_count()

		self.mesh = self.mesh.crop_mesh(elm_type=4)
		dipole_pos = np.array([0., 0., 300])
		dipole_moment = np.array([1., 0., 0.])
		didt = 1e6
		r = (self.mesh.nodes.node_coord - dipole_pos) * 1e-3
		dAdt = 1e-7 * didt * np.cross(dipole_moment, r) / (np.linalg.norm(r, axis=1)[:, None] ** 3)
		dAdt = mesh_io.NodeData(dAdt, mesh=self.mesh)
		dAdt.field_name = 'dAdt'
		dAdt.mesh = self.mesh
		pos = self.mesh.elements_baricenters().value

		cond = mesh_io.ElementData(np.ones(self.mesh.elm.nr))
		cond.mesh = self.mesh
		stimulator = tms_coil.tms_stimulator.TmsStimulator("Example Stimulator", "Example Stimulator Brand", 100e6)
		dipole_elm = tms_coil.tms_coil_element.DipoleElements(stimulator=stimulator, points=dipole_pos[None], values=dipole_moment[None])
		coil = tms_coil.tms_coil.TmsCoil([dipole_elm])

		fill=False
		nearest=True
		useElements=True

		# create the ROI
		center_points = self.mesh.elements_baricenters().value
		out_point = np.array((0, 0, 100))[None]
		center_points = np.concatenate((center_points, out_point), axis=0)
		print("Creating point cloud...")
		point_cloud = onlinefem.FemTargetPointCloud(self.mesh, center_points, nearest_neighbor=nearest, fill_nearest=fill)
		 
		print("prepare and setup OnlineFEM")
		self.ofem = onlinefem.OnlineFEM(
				self.mesh,
				'TMS',
				roi=[point_cloud],
				coil=coil,
				useElements=useElements,
				solver_options=solver,
				cond=cond,
				cpus=cpus)

		#calculate vector E-field
		self.ofem.dataType = [1]
		print("Solve the FEM")
		self.E = self.ofem.update_field(matsimnibs=np.identity(4), didt=1e6)[0][0]

	def copy_E_to_share(self, share_name):
		if self.sharedE is None:
			self.shared_memory_for_E = multiprocessing.shared_memory.SharedMemory(name=share_name)
			self.sharedE = np.ndarray(self.E.shape, self.E.dtype, buffer=self.shared_memory_for_E.buf)
		self.sharedE[:] = self.E

	def update_E_field(self, probe_matrix_list):
		mat = np.array(probe_matrix_list)
		self.E = self.ofem.update_field(matsimnibs=mat, didt=1e6)[0][0]


port = 18891
server = rpyc.utils.server.ThreadedServer(SimNIBSService,
	port=port,
	protocol_config={
		'allow_public_attrs': True,
		'allow_all_attrs': True,
		'allow_pickle': True,
	}
)
server.start()
