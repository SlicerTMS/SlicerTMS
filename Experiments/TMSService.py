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
        self._sharedE = None      # numpy view into shared memory (vector E)
        self._sharedEnorm = None  # numpy view into shared memory (scalar |E|)
        self._shmEnorm = None     # shared memory block for streaming Enorm
        self._shm = None          # shared memory block (vector E, legacy)
        self._warp_ctx = None     # WarpFEMContext for streaming CG
        self._converged = True    # streaming solve state
        self._tag1 = None         # (M,) int tissue tags, if available
        self._barycenters = None  # (M, 3) element barycenters (meters)
        self._elem_weights = None # (M,) sigma*vol per element
        self._last_probe_mat = None  # last 4x4 probe matrix (for optimization init)

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
        self._tag1 = data["tag1"].astype(np.int32) if "tag1" in data else None
        print(
            f"  {len(self.mesh.nodes):,} nodes  "
            f"{len(self.mesh.elements):,} elements"
        )

        if solver in ("warp_cpu", "warp_gpu"):
            # Warp builds its own stiffness matrix on GPU — skip numpy assembly
            self._G = None
            self._K = None
            self._warmup_warp(solver)
        else:
            from tmswarp.solver import gradient_operator, assemble_stiffness
            print("Assembling stiffness matrix ...")
            self._G = gradient_operator(self.mesh)
            self._K = assemble_stiffness(self.mesh, self._G)
            print("  Done.")
            self._prefactorize_K()

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
            # Assemble numpy stiffness if not yet done (e.g. started with warp)
            if self._K is None:
                from tmswarp.solver import gradient_operator, assemble_stiffness
                print("Assembling stiffness matrix ...")
                self._G = gradient_operator(self.mesh)
                self._K = assemble_stiffness(self.mesh, self._G)
                print("  Done.")
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

    @property
    def n_elements(self):
        """Number of mesh elements (for shared memory sizing)."""
        return len(self.mesh.elements) if self.mesh is not None else 0

    # ------------------------------------------------------------------
    # Streaming solve loop (event-driven via stdin/stdout)
    # ------------------------------------------------------------------

    def start_streaming(self, share_name):
        """Enter the streaming solve loop.

        Blocks until STOP is received on stdin.  Probe positions arrive
        via stdin as ``PROBE <16 floats>`` lines; |E| (Enorm) updates are
        written to shared memory and signalled via stdout
        ``E_UPDATED iter=N residual=R converged=0|1`` lines.

        Called via ``rpyc.async_()`` so the client isn't blocked.

        The shared memory block *share_name* must be sized for
        ``(n_elements,) float64`` — scalar Enorm per element.
        """
        # Attach to shared memory for scalar Enorm
        self._shmEnorm = multiprocessing.shared_memory.SharedMemory(
            name=share_name
        )
        n_elem = len(self.mesh.elements)
        self._sharedEnorm = np.ndarray(
            (n_elem,), dtype=np.float64, buffer=self._shmEnorm.buf
        )

        # Publish the initial E-field so the mesh shows colors immediately
        if self.E is not None:
            self._sharedEnorm[:] = np.linalg.norm(self.E, axis=1)

        print(f"STREAMING_READY solver={self._solver}")
        sys.stdout.flush()
        self._solve_loop()

    def _handle_command(self, line, magnetic_dipole_dadt, compute_efield_at_elements):
        """Dispatch a stdin command. Returns 'stop' to exit, 'continue' otherwise."""
        if line == "STOP" or not line:
            return "stop"
        if line.startswith("OPTIMIZE"):
            parts = line.split()
            if len(parts) >= 4:
                target_mm = np.array([float(parts[1]), float(parts[2]),
                                      float(parts[3])])
                pending = self._optimize_coil(target_mm)
                if pending is not None:
                    # Optimization was interrupted — process the pending cmd
                    return self._handle_command(
                        pending, magnetic_dipole_dadt, compute_efield_at_elements
                    )
            return "continue"
        if line.startswith("PROBE"):
            self._apply_probe(line, magnetic_dipole_dadt)
            if self._solver == "numpy":
                self._solve_numpy_and_emit(compute_efield_at_elements)
            return "continue"
        return "continue"

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
                result = self._handle_command(
                    latest, magnetic_dipole_dadt, compute_efield_at_elements
                )
                if result == "stop":
                    break
                continue

            # Mid-solve: check for new input without blocking
            latest = self._drain_stdin_nonblocking()
            if latest is not None:
                if latest == "STOP":
                    break
                result = self._handle_command(
                    latest, magnetic_dipole_dadt, compute_efield_at_elements
                )
                if result == "stop":
                    break
                continue

            # Warp CG: run one chunk of iterations
            if self._solver in ("warp_cpu", "warp_gpu") and not self._converged:
                err, iters, converged = self._warp_ctx.step(n_iters=50)
                enorm = self._warp_ctx.compute_enorm()
                if self._sharedEnorm is not None:
                    self._sharedEnorm[:] = enorm
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
        self._last_probe_mat = mat

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
        if self._sharedEnorm is not None:
            self._sharedEnorm[:] = np.linalg.norm(self.E, axis=1)
        self._converged = True

        mag = np.linalg.norm(self.E, axis=1)
        print(
            f"E_UPDATED iter=1 "
            f"residual=0.000000e+00 "
            f"converged=1"
        )
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Coil position optimization (adjoint-based)
    # ------------------------------------------------------------------

    def _find_nearest_element(self, target_m):
        """Find the brain element nearest to a target point (in meters).

        Restricts to grey matter (tag=2) if tag1 is available.
        """
        from tmswarp.conductor import element_barycenters

        if self._barycenters is None:
            self._barycenters = element_barycenters(self.mesh)

        bary = self._barycenters
        if self._tag1 is not None:
            brain_mask = self._tag1 == 2
            brain_idx = np.nonzero(brain_mask)[0]
            if len(brain_idx) > 0:
                dists = np.linalg.norm(bary[brain_idx] - target_m, axis=1)
                return int(brain_idx[np.argmin(dists)])

        # Fallback: all elements
        dists = np.linalg.norm(bary - target_m, axis=1)
        return int(np.argmin(dists))

    def _ensure_elem_weights(self):
        """Precompute sigma * vol per element (cached)."""
        if self._elem_weights is None:
            from tmswarp.conductor import element_volumes
            vols = element_volumes(self.mesh)
            self._elem_weights = vols * self.mesh.conductivity

    def _compute_objective_and_gradient(self, params, target_elem):
        """Compute -|E[target]| and its analytic gradient w.r.t. coil params.

        Parameters
        ----------
        params : (6,) array
            [px, py, pz, mx, my, mz] — position (mm), moment (unnormalized).
        target_elem : int
            Element index to maximize |E| at.

        Returns
        -------
        loss : float
            -|E[target]|
        gradient : (6,) array
            d(loss)/d(params)
        """
        from tmswarp.coil import magnetic_dipole_dadt
        from tmswarp.solver import assemble_rhs_tms

        self._ensure_elem_weights()
        w = self._elem_weights

        pos_mm = params[:3]
        moment_raw = params[3:6]
        moment_norm = np.linalg.norm(moment_raw)
        if moment_norm < 1e-12:
            return 0.0, np.zeros(6)
        moment = moment_raw / moment_norm
        pos_m = pos_mm * 1e-3

        nodes = self.mesh.nodes
        elems = self.mesh.elements
        G = self._G
        t = target_elem
        t_nodes = elems[t]

        # === FORWARD ===
        dAdt = magnetic_dipole_dadt(pos_m, moment, DIDT, nodes)
        b = assemble_rhs_tms(self.mesh, dAdt, G)
        phi = np.zeros(len(nodes), dtype=np.float64)
        phi[1:] = self._K_factor(b[1:])

        # E at target element
        phi_t = phi[t_nodes]                          # (4,)
        G_t = G[t]                                    # (4, 3)
        grad_phi_t = phi_t @ G_t                      # (3,)
        dAdt_bary_t = dAdt[t_nodes].mean(axis=0)      # (3,)
        E_t = -grad_phi_t - dAdt_bary_t               # (3,)
        Enorm_t = np.linalg.norm(E_t)

        loss = -Enorm_t
        if Enorm_t < 1e-15:
            return loss, np.zeros(6)

        # === ADJOINT ===
        # Step 1: dL/dE
        dL_dE = -E_t / Enorm_t                        # (3,)

        # Step 2: dL/dphi (sparse — only 4 entries at target nodes)
        dL_dphi = np.zeros(len(nodes), dtype=np.float64)
        for i in range(4):
            dL_dphi[t_nodes[i]] += dL_dE @ (-G_t[i])

        # Step 3: adjoint solve (reuse pre-factored K!)
        lam = np.zeros(len(nodes), dtype=np.float64)
        lam[1:] = self._K_factor(dL_dphi[1:])

        # Step 4: dL/d(dAdt_nodes) via b (adjoint contribution)
        # "Gradient of lambda" at each element
        lam_elem = lam[elems]                          # (n_elem, 4)
        lam_grad = np.einsum('ei,eid->ed', lam_elem, G)  # (n_elem, 3)

        # Scatter weighted contributions to nodes
        dL_dAdt = np.zeros_like(dAdt)                  # (N, 3)
        weighted_lam_grad = w[:, None] * lam_grad * (-0.25)  # (n_elem, 3)
        for i in range(4):
            np.add.at(dL_dAdt, elems[:, i], weighted_lam_grad)

        # Step 5: direct dL/d(dAdt) from E_t
        for j in t_nodes:
            dL_dAdt[j] += dL_dE * (-0.25)

        # Step 6: chain rule through Biot-Savart
        C = 1e-7 * DIDT  # mu0_4pi * didt
        r = nodes - pos_m                              # (N, 3)
        r_norm = np.linalg.norm(r, axis=1)[:, None]   # (N, 1)
        r_norm3 = r_norm ** 3
        r_norm5 = r_norm ** 5

        # Clamp to avoid division by zero near the dipole
        r_norm3 = np.maximum(r_norm3, 1e-30)
        r_norm5 = np.maximum(r_norm5, 1e-30)

        cross_mr = np.cross(moment, r)                 # (N, 3)

        # dL/d(pos_m) — two terms from d/dr[cross(m,r)/|r|^3]
        # Term 1: -C * cross(m, dL_dAdt) / |r|^3
        term1 = -C * np.cross(moment, dL_dAdt) / r_norm3
        # Term 2: +3C * cross(m,r) * (r . dL_dAdt) / |r|^5
        r_dot_dLdA = np.sum(r * dL_dAdt, axis=1)[:, None]
        term2 = 3 * C * cross_mr * r_dot_dLdA / r_norm5
        dL_dpos_m = (term1 + term2).sum(axis=0)
        dL_dpos_mm = dL_dpos_m * 1e-3  # chain: pos_m = pos_mm * 1e-3

        # dL/d(moment)
        cross_r_dLdA = np.cross(r, dL_dAdt)           # (N, 3)
        dL_dmoment = C * (cross_r_dLdA / r_norm3).sum(axis=0)

        # Chain through normalization: m = m_raw / |m_raw|
        dL_dmoment_raw = (
            dL_dmoment - moment * np.dot(moment, dL_dmoment)
        ) / moment_norm

        gradient = np.concatenate([dL_dpos_mm, dL_dmoment_raw])
        return loss, gradient

    def _params_to_matrix(self, params):
        """Convert optimization params [px,py,pz,mx,my,mz] to a 4x4 matrix."""
        pos_mm = params[:3]
        moment_raw = params[3:6]
        n = moment_raw / np.linalg.norm(moment_raw)

        # Build orthonormal frame with n as Z
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(n, ref)) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        x = np.cross(ref, n)
        x /= np.linalg.norm(x)
        y = np.cross(n, x)

        mat = np.eye(4, dtype=np.float64)
        mat[:3, 0] = x
        mat[:3, 1] = y
        mat[:3, 2] = n
        mat[:3, 3] = pos_mm
        return mat

    def _optimize_coil(self, target_mm):
        """Run Adam optimization to maximize |E| at the target point.

        Returns a pending stdin command (str) if interrupted, else None.
        """
        from tmswarp.coil import magnetic_dipole_dadt
        from tmswarp.solver import assemble_rhs_tms
        from tmswarp.fields import compute_efield_at_elements

        # Ensure numpy backend prerequisites
        if self._K_factor is None or self._G is None:
            print("OPTIMIZE_ERROR: numpy solver not initialized")
            sys.stdout.flush()
            return None

        target_m = np.array(target_mm) * 1e-3
        target_elem = self._find_nearest_element(target_m)
        print(f"Optimizing for element {target_elem} "
              f"(brain tag={self._tag1[target_elem] if self._tag1 is not None else '?'})")
        sys.stdout.flush()

        # Initialize from current coil state or default
        if self._dAdt is not None and self._last_probe_mat is not None:
            mat = self._last_probe_mat
            pos_mm = mat[:3, 3].copy()
            moment = mat[:3, 2].copy()
        else:
            pos_mm = np.array([0.0, 0.0, 100.0])
            moment = np.array([0.0, 0.0, 1.0])

        params = np.concatenate([pos_mm, moment])

        # Adam optimizer state
        lr = 2.0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_adam = np.zeros(6)
        v_adam = np.zeros(6)

        max_iters = 200
        prev_loss = None

        for iteration in range(max_iters):
            # Check for interruption
            cmd = self._drain_stdin_nonblocking()
            if cmd is not None:
                if cmd == "STOP":
                    return "STOP"
                return cmd  # OPTIMIZE or PROBE — caller handles

            loss, grad = self._compute_objective_and_gradient(params, target_elem)

            # Adam update
            t_adam = iteration + 1
            m_adam = beta1 * m_adam + (1 - beta1) * grad
            v_adam = beta2 * v_adam + (1 - beta2) * grad ** 2
            m_hat = m_adam / (1 - beta1 ** t_adam)
            v_hat = v_adam / (1 - beta2 ** t_adam)
            params = params - lr * m_hat / (np.sqrt(v_hat) + eps)

            # Renormalize moment direction to stay on unit sphere
            moment_norm = np.linalg.norm(params[3:6])
            if moment_norm > 1e-12:
                params[3:6] /= moment_norm

            # Compute full E-field for visualization every few iterations
            if iteration % 5 == 0 or iteration == max_iters - 1:
                pos_m = params[:3] * 1e-3
                moment_dir = params[3:6] / np.linalg.norm(params[3:6])
                dAdt = magnetic_dipole_dadt(pos_m, moment_dir, DIDT,
                                            self.mesh.nodes)
                b = assemble_rhs_tms(self.mesh, dAdt, self._G)
                phi = np.zeros(len(self.mesh.nodes), dtype=np.float64)
                phi[1:] = self._K_factor(b[1:])
                self.E = compute_efield_at_elements(
                    self.mesh, phi, dAdt, self._G
                )
                self._dAdt = dAdt
                if self._sharedEnorm is not None:
                    self._sharedEnorm[:] = np.linalg.norm(self.E, axis=1)

                # Emit current probe position
                mat = self._params_to_matrix(params)
                vals = " ".join(f"{v:.6f}" for v in mat.ravel())
                print(f"OPT_PROBE {vals}")
                print(f"E_UPDATED iter={iteration} loss={loss:.6e} converged=0")
                sys.stdout.flush()

            # Convergence check
            if prev_loss is not None and abs(loss - prev_loss) < 1e-8:
                break
            prev_loss = loss

        # Final full solve and emit
        pos_m = params[:3] * 1e-3
        moment_dir = params[3:6] / np.linalg.norm(params[3:6])
        dAdt = magnetic_dipole_dadt(pos_m, moment_dir, DIDT, self.mesh.nodes)
        b = assemble_rhs_tms(self.mesh, dAdt, self._G)
        phi = np.zeros(len(self.mesh.nodes), dtype=np.float64)
        phi[1:] = self._K_factor(b[1:])
        self.E = compute_efield_at_elements(self.mesh, phi, dAdt, self._G)
        self._dAdt = dAdt
        if self._sharedEnorm is not None:
            self._sharedEnorm[:] = np.linalg.norm(self.E, axis=1)

        mat = self._params_to_matrix(params)
        self._last_probe_mat = mat
        vals = " ".join(f"{v:.6f}" for v in mat.ravel())
        print(f"OPT_PROBE {vals}")
        print(f"OPTIMIZE_DONE iter={iteration} loss={loss:.6e}")
        sys.stdout.flush()
        return None

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
        from tmswarp.fields import compute_efield_at_elements

        if self._solver == "numpy":
            from tmswarp.solver import assemble_rhs_tms
            b = assemble_rhs_tms(self.mesh, dAdt, self._G)
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
