"""TMSWarp FEM service for interactive TMS E-field visualization.

RPyC server that exposes TMSWarp FEM solvers so a Slicer client can
update the E-field interactively as the coil position changes.

Three solver backends
---------------------
  "numpy"    — scipy spsolve with pre-factorized reduced K (fast re-solves)
  "warp_cpu" — Warp.fem CG on CPU  (~112 s per solve on Apple Silicon)
  "warp_gpu" — Warp.fem CG on GPU  (requires CUDA; potentially < 1 s)

Usage
-----
    # From the TMSWarp pixi environment:
    cd /path/to/SlicerTMS/TMSWarp
    pixi run python ../Experiments/TMSService.py [--solver numpy] [--port 18892]

    # Or with explicit python (use the pixi env python):
    /path/to/TMSWarp/.pixi/envs/default/bin/python \\
        /path/to/SlicerTMS/Experiments/TMSService.py

Notes
-----
The "numpy" backend pre-factorizes the stiffness matrix on startup
(one-time cost: ~2-5 min for ernie, instant for sphere3).  Each
subsequent update_E_field call then costs only the sparse triangular
solve + RHS assembly: typically a few seconds.
"""

import argparse
import multiprocessing.shared_memory
import os
import select
import sys

import numpy as np

import rpyc

# ---------------------------------------------------------------------------
# Locate TMSWarp src/ so we can import tmswarp without a pip install
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_TMSWARP_ROOT = os.path.abspath(os.path.join(_HERE, "..", "TMSWarp"))
_TMSWARP_SRC = os.path.join(_TMSWARP_ROOT, "src")
if os.path.isdir(_TMSWARP_SRC) and _TMSWARP_SRC not in sys.path:
    sys.path.insert(0, _TMSWARP_SRC)

DEFAULT_PORT = 18892
DIDT = 1e6  # A/s — standard TMS pulse


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class TMSService(rpyc.SlaveService):
    """RPyC service wrapping TMSWarp FEM backends."""

    SOLVERS = ("numpy", "warp_cpu", "warp_gpu")

    def __init__(self):
        self.mesh = None
        self.nodes_mm = None      # (N, 3) float64 mm — for VTK
        self.elements = None      # (M, 4) int32  0-based — for VTK
        self.E = None             # (M, 3) float64  current E-field (V/m)
        self._solver = "numpy"
        self._G = None            # gradient operator (N_elem, 4, 3)
        self._K = None            # assembled stiffness (sparse, unpinned)
        self._K_factor = None     # callable: b_reduced → phi_reduced
        self._dAdt = None         # last dAdt at nodes (N, 3)
        self._sharedE = None      # numpy view into shared memory
        self._shm = None          # shared memory block
        self._warp_ctx = None     # WarpFEMContext for streaming CG
        self._converged = True    # streaming solve state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize_system(self, mesh_path=None, solver="numpy"):
        """Load mesh, assemble stiffness, run initial solve.

        Parameters
        ----------
        mesh_path : str or None
            Path to a .npz mesh file in TMSWarp format (ernie_data.npz or
            sphere3_data.npz).  None → auto-detect from TMSWarp/.
        solver : str
            Starting backend: "numpy", "warp_cpu", or "warp_gpu".
        """
        from tmswarp.conductor import TetMesh
        from tmswarp.solver import gradient_operator, assemble_stiffness

        self._solver = solver
        self._K_factor = None

        mesh_path = self._resolve_mesh_path(mesh_path)
        print(f"Loading mesh: {mesh_path}")
        data = np.load(mesh_path)
        self.mesh = TetMesh(
            nodes=data["nodes"].astype(np.float64),
            elements=data["elements"].astype(np.int32),
            conductivity=data["conductivity"].astype(np.float64),
        )
        self.nodes_mm = self.mesh.nodes * 1000.0   # m → mm  (for VTK)
        self.elements = self.mesh.elements          # 0-based int32
        print(
            f"  {len(self.mesh.nodes):,} nodes  "
            f"{len(self.mesh.elements):,} elements"
        )

        print("Assembling stiffness matrix ...")
        self._G = gradient_operator(self.mesh)
        self._K = assemble_stiffness(self.mesh, self._G)
        print("  Done.")

        if solver == "numpy":
            self._prefactorize_K()
        elif solver in ("warp_cpu", "warp_gpu"):
            self._warmup_warp(solver)

        # Initial solve — coil 100 mm above head centre (positive z)
        init_mat = np.eye(4, dtype=np.float64)
        init_mat[2, 3] = 100.0
        self.update_E_field(init_mat.tolist())
        print("Initialization complete.")

    def set_solver(self, solver):
        """Switch solver backend without reloading the mesh.

        Parameters
        ----------
        solver : str
            "numpy", "warp_cpu", or "warp_gpu".
        """
        if solver not in self.SOLVERS:
            raise ValueError(
                f"Unknown solver {solver!r}. Options: {self.SOLVERS}"
            )
        if self.mesh is None:
            raise RuntimeError("Call initialize_system() first.")

        if solver == "numpy" and self._K_factor is None:
            self._prefactorize_K()
        elif solver in ("warp_cpu", "warp_gpu"):
            self._warmup_warp(solver)

        old = self._solver
        self._solver = solver
        print(f"Solver: {old} → {solver}")

        # Re-solve immediately with last dAdt so E stays current
        if self._dAdt is not None:
            self._solve_and_update(self._dAdt)

    def update_E_field(self, probe_matrix_list):
        """Recompute E-field for a new coil position/orientation.

        Parameters
        ----------
        probe_matrix_list : list[list[float]]
            4×4 coil-to-world transform (Slicer RAS, mm).
            Column 3 = coil centre in mm.
            Column 2 = coil normal (pointing away from head).
        """
        from tmswarp.coil import magnetic_dipole_dadt

        mat = np.array(probe_matrix_list, dtype=np.float64)
        dipole_pos_m = mat[:3, 3] * 1e-3        # mm → m
        dipole_moment = mat[:3, 2]               # coil normal
        norm = np.linalg.norm(dipole_moment)
        if norm > 1e-12:
            dipole_moment = dipole_moment / norm

        dAdt = magnetic_dipole_dadt(
            dipole_pos_m, dipole_moment, DIDT, self.mesh.nodes
        )
        self._dAdt = dAdt
        self._solve_and_update(dAdt)

    def copy_E_to_share(self, share_name):
        """Copy current E-field into a pre-created shared-memory block."""
        if self._sharedE is None:
            self._shm = multiprocessing.shared_memory.SharedMemory(
                name=share_name
            )
            self._sharedE = np.ndarray(
                self.E.shape, dtype=self.E.dtype,
                buffer=self._shm.buf
            )
        self._sharedE[:] = self.E

    # ------------------------------------------------------------------
    # Streaming solve loop (event-driven via stdin/stdout)
    # ------------------------------------------------------------------

    def start_streaming(self, share_name):
        """Enter the streaming solve loop.

        Blocks until STOP is received on stdin.  Probe positions arrive
        via stdin as ``PROBE <16 floats>`` lines; E-field updates are
        written to shared memory and signalled via stdout
        ``E_UPDATED iter=N residual=R converged=0|1`` lines.

        Called via ``rpyc.async_()`` so the client isn't blocked.
        """
        # Set up shared memory
        self.copy_E_to_share(share_name)
        print(f"STREAMING_READY solver={self._solver}")
        sys.stdout.flush()
        self._solve_loop()

    def _solve_loop(self):
        from tmswarp.coil import magnetic_dipole_dadt
        from tmswarp.fields import compute_efield_at_elements

        # Create WarpFEMContext if using warp solver
        if self._solver in ("warp_cpu", "warp_gpu") and self._warp_ctx is None:
            from tmswarp.solver_warp import WarpFEMContext
            device = "cpu" if self._solver == "warp_cpu" else None
            self._warp_ctx = WarpFEMContext(
                self.mesh, device=device, tol=1e-4
            )

        self._converged = True  # start idle, waiting for first PROBE

        while True:
            if self._converged or self._dAdt is None:
                # Nothing to compute — block on stdin until new input
                line = sys.stdin.readline().strip()
                if not line or line == "STOP":
                    break
                latest = self._drain_stdin_keep_latest(line)
                if latest == "STOP":
                    break
                self._apply_probe(latest, magnetic_dipole_dadt)
                if self._solver == "numpy":
                    self._solve_numpy_and_emit(compute_efield_at_elements)
                    continue

            # Mid-solve: check for new input without blocking
            latest = self._drain_stdin_nonblocking()
            if latest is not None:
                if latest == "STOP":
                    break
                self._apply_probe(latest, magnetic_dipole_dadt)
                if self._solver == "numpy":
                    self._solve_numpy_and_emit(compute_efield_at_elements)
                    continue

            # Warp CG: run one chunk of iterations
            if self._solver in ("warp_cpu", "warp_gpu") and not self._converged:
                err, iters, converged = self._warp_ctx.step(n_iters=50)
                phi = self._warp_ctx.get_phi()
                self.E = compute_efield_at_elements(
                    self.mesh, phi, self._dAdt, self._G
                )
                if self._sharedE is not None:
                    self._sharedE[:] = self.E
                self._converged = converged
                print(
                    f"E_UPDATED iter={iters} "
                    f"residual={err:.6e} "
                    f"converged={int(converged)}"
                )
                sys.stdout.flush()

        print("Streaming stopped.")
        sys.stdout.flush()

    def _drain_stdin_nonblocking(self):
        """Read ALL available stdin lines, return the last one (or None)."""
        latest = None
        while select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip()
            if not line:
                break
            latest = line
        return latest

    def _drain_stdin_keep_latest(self, first_line):
        """After reading first_line (blocking), drain queued lines, return latest."""
        latest = first_line
        while select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip()
            if not line:
                break
            latest = line
        return latest

    def _apply_probe(self, line, magnetic_dipole_dadt):
        """Parse a PROBE line, compute dAdt, set new RHS, reset convergence."""
        if not line.startswith("PROBE"):
            return
        floats = [float(x) for x in line.split()[1:]]
        mat = np.array(floats, dtype=np.float64).reshape(4, 4)

        dipole_pos_m = mat[:3, 3] * 1e-3
        dipole_moment = mat[:3, 2]
        norm = np.linalg.norm(dipole_moment)
        if norm > 1e-12:
            dipole_moment = dipole_moment / norm

        dAdt = magnetic_dipole_dadt(
            dipole_pos_m, dipole_moment, DIDT, self.mesh.nodes
        )
        self._dAdt = dAdt
        self._converged = False

        if self._solver in ("warp_cpu", "warp_gpu") and self._warp_ctx is not None:
            self._warp_ctx.set_rhs(dAdt)

    def _solve_numpy_and_emit(self, compute_efield_at_elements):
        """Full numpy backsubstitution, write to shared memory, emit notification."""
        from tmswarp.solver import assemble_rhs_tms

        b = assemble_rhs_tms(self.mesh, self._dAdt, self._G)
        phi_reduced = self._K_factor(b[1:])
        phi = np.zeros(len(self.mesh.nodes), dtype=np.float64)
        phi[1:] = phi_reduced

        self.E = compute_efield_at_elements(
            self.mesh, phi, self._dAdt, self._G
        )
        if self._sharedE is not None:
            self._sharedE[:] = self.E
        self._converged = True

        mag = np.linalg.norm(self.E, axis=1)
        print(
            f"E_UPDATED iter=1 "
            f"residual=0.000000e+00 "
            f"converged=1"
        )
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_mesh_path(self, mesh_path):
        if mesh_path and os.path.isfile(mesh_path):
            return mesh_path
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "SlicerTMS")
        search_dirs = [_TMSWARP_ROOT, cache_dir]
        for name in ("ernie_data.npz", "sphere3_data.npz"):
            for d in search_dirs:
                candidate = os.path.join(d, name)
                if os.path.isfile(candidate):
                    return candidate
        raise FileNotFoundError(
            "No mesh file found. Run TMSWarp/scripts/fetch_ernie.py first, "
            "or pass an explicit mesh_path to initialize_system()."
        )

    def _prefactorize_K(self):
        """LU-factorize the reduced stiffness matrix for fast re-solves.

        Matches solve_fem()'s row/column-elimination pin: node 0 is removed
        from the system.  The returned callable accepts b[1:] and returns
        phi[1:].
        """
        from scipy.sparse.linalg import factorized

        print(
            "Pre-factorizing stiffness matrix (one-time cost; "
            "~2-5 min for ernie, instant for sphere3) ..."
        )
        K_csr = self._K.tocsr()
        # Remove pin node (index 0) — same as solve_fem()
        K_reduced = K_csr[1:, 1:].tocsc()
        self._K_factor = factorized(K_reduced)
        print("  Done — subsequent solves will be seconds instead of minutes.")

    def _warmup_warp(self, solver):
        """JIT-compile Warp kernels on a tiny mesh so first real solve is fast."""
        from tmswarp.conductor import make_sphere_mesh
        from tmswarp.coil import magnetic_dipole_dadt
        from tmswarp.solver_warp import solve_fem_warp

        device = "cpu" if solver == "warp_cpu" else None
        print(f"Warming up Warp kernels ({solver}) ...")
        small = make_sphere_mesh(
            radius=0.05, n_shells=3, n_surface=50, conductivity=1.0
        )
        dAdt_s = magnetic_dipole_dadt(
            np.array([0.0, 0.0, 0.1]),
            np.array([1.0, 0.0, 0.0]),
            DIDT, small.nodes
        )
        solve_fem_warp(small, dAdt_s, device=device, quiet=True)
        print("  Done.")

    def _solve_and_update(self, dAdt):
        """Run the selected backend, compute E-field, store in self.E."""
        from tmswarp.solver import assemble_rhs_tms
        from tmswarp.fields import compute_efield_at_elements

        b = assemble_rhs_tms(self.mesh, dAdt, self._G)

        if self._solver == "numpy":
            # Fast path: apply pre-factorized K⁻¹ to reduced RHS
            phi_reduced = self._K_factor(b[1:])
            phi = np.zeros(len(self.mesh.nodes), dtype=np.float64)
            phi[1:] = phi_reduced

        elif self._solver in ("warp_cpu", "warp_gpu"):
            from tmswarp.solver_warp import solve_fem_warp
            device = "cpu" if self._solver == "warp_cpu" else None
            try:
                phi = solve_fem_warp(self.mesh, dAdt, device=device, quiet=True)
            except RuntimeError as exc:
                print(f"  WARNING: {exc}")
                if self.E is None:
                    raise
                print("  Keeping previous E-field.")
                return
        else:
            raise ValueError(f"Unknown solver: {self._solver!r}")

        self.E = compute_efield_at_elements(self.mesh, phi, dAdt, self._G)
        mag = np.linalg.norm(self.E, axis=1)
        print(
            f"  [{self._solver}]  |E| mean={mag.mean():.3f}  "
            f"max={mag.max():.3f} V/m"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TMSWarp RPyC FEM service"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Port to listen on (default {DEFAULT_PORT})")
    parser.add_argument(
        "--solver", default="numpy", choices=TMSService.SOLVERS,
        help="Default solver backend (default: numpy)"
    )
    args = parser.parse_args()

    print(
        f"Starting TMSService on port {args.port} "
        f"(default solver: {args.solver}) ..."
    )
    server = rpyc.utils.server.ThreadedServer(
        TMSService,
        port=args.port,
        protocol_config={
            "allow_public_attrs": True,
            "allow_all_attrs": True,
            "allow_pickle": True,
        },
    )
    server.start()
