from utilities import np, sparse, Any, Tuple, Dict, Optional
from heat_mappers import  MIN_TEMP_VALUE




def _compute_explicit_laplacian(
    T: np.ndarray,
    dx: float,
    boundary_condition: str = "neumann",
) -> np.ndarray:
    
    T_bc = T.copy()

    if boundary_condition == "periodic":
        pad = np.pad(T_bc, 1, mode="wrap")
    elif boundary_condition == "neumann":
        pad = np.pad(T_bc, 1, mode="edge")
    else:
        pad = np.pad(T_bc, 1, mode="constant", constant_values=MIN_TEMP_VALUE)
    
    lap = (
        pad[:-2, 1:-1] + pad[2:, 1:-1] +
        pad[1:-1, :-2] + pad[1:-1, 2:] -
        4.0 * pad[1:-1, 1:-1]
    ) / dx**2

    return lap


def _build_implicit_laplacian(
    H: int,
    W: int,
    dx: float,
    boundary_condition: str = "neumann"
) -> sparse.csr_matrix:
    """
    2D Laplacian via Kron-sum for 'periodic' or 'neumann' homogeneous cases.
    For Dirichlet we rely on apply_boundary_conditions() in the solver.
    """
    def lap1d(n: int, bc: str) -> sparse.csr_matrix:
        if bc == "periodic":
            T = sparse.diags([1, -2, 1], [-1,0,1], (n,n), format="lil")
            T[0, -1] = T[-1, 0] = 1.0
        else:
            off  = np.ones(n-1)
            main = -2.0 * np.ones(n)
            if bc == "neumann":
                main[[0,-1]] = -1.0
            T = sparse.diags([off, main, off], [-1,0,1], (n,n), format="csr")
        return T.tocsr()

    T_w = lap1d(W, boundary_condition)
    T_h = lap1d(H, boundary_condition)
    I_w = sparse.eye(W, format="csr")
    I_h = sparse.eye(H, format="csr")

    return (sparse.kron(I_h, T_w) + sparse.kron(T_h, I_w)) / dx**2


def ftcs_solver(
    T_init: np.ndarray,
    alpha_map: np.ndarray,
    *,
    dx: float = 1e-4,
    max_iterations: int = 5000,
    convergence_tolerance: float = 1e-4,
    boundary_condition: str = "neumann",
    save_states_predicate: Any = lambda _: False,
) -> Tuple[Dict[str, np.ndarray], int, float]:
    """
    FTCS explicit solver.
    """
    alpha_max = alpha_map.max()
    dt = 0.4 * dx*dx / (4 * alpha_max)
    limit = dx*dx/(4 * alpha_max)
    if dt > limit:
        raise RuntimeError(f"FTCS unstable: dt={dt:.2e} > {limit:.2e}")
    alpha_dt = alpha_map * dt

    snapshots = {"temp_state_initial": T_init.copy()}

    T_curr = T_init.copy()
    T_prev = np.empty_like(T_curr)

    for step in range(1, max_iterations + 1):
        np.copyto(T_prev, T_curr)

        lap = _compute_explicit_laplacian(T_prev, dx=dx, boundary_condition=boundary_condition)
        T_curr = T_prev + alpha_dt * lap

        prev_norm = np.linalg.norm(T_prev)
        rel_delta = np.linalg.norm(T_curr - T_prev) / prev_norm if prev_norm > 0 else -1.0

        if rel_delta < convergence_tolerance:
            break

        if save_states_predicate(step):
            snapshots[f"temp_state_{step}"] = T_curr.copy()


    T_final = T_curr.copy()
    snapshots[f"temp_state_{step}"] = T_final
    snapshots["temp_state_final"] = T_final
    
    return snapshots, step, rel_delta


def crank_nicolson_solver(
    T_init: np.ndarray,
    alpha_map: np.ndarray,
    *,
    dx: float = 1e-4,
    max_iterations: int = 5000,
    convergence_tolerance: float = 1e-4,
    boundary_condition: str = "neumann",
    save_states_predicate: Any = lambda _: False,
) -> Tuple[Dict[str, np.ndarray], int, float]:
    """
    Implicit Crank–Nicolson solver with LU factorization.
    """
    H, W = T_init.shape
    N = H * W

    alpha_flat = alpha_map.ravel()
    dt = 0.4 * dx*dx / (4 * alpha_flat.mean())

    L = _build_implicit_laplacian(H, W, dx, boundary_condition=boundary_condition)
    A = sparse.eye(N) - 0.5*dt * sparse.diags(alpha_flat) @ L
    B = sparse.eye(N) + 0.5*dt * sparse.diags(alpha_flat) @ L
    lu = sparse.linalg.splu(A.tocsc())

    snapshots = {"temp_state_initial": T_init.copy()}

    T_prev = T_init.ravel().copy()
    T_curr = np.empty_like(T_prev)

    for step in range(1, max_iterations + 1):
        rhs = B @ T_prev
        T_curr = lu.solve(rhs)

        prev_norm = np.linalg.norm(T_prev)
        rel_delta = np.linalg.norm(T_curr - T_prev) / prev_norm if prev_norm > 0 else -1.0

        if rel_delta < convergence_tolerance:
            break

        if save_states_predicate(step):
            snapshots[f"temp_state_{step}"] = T_curr.reshape(H, W).copy()

        T_prev[:] = T_curr


    T_final = T_curr.reshape(H, W).copy()
    snapshots[f"temp_state_{step}"] = T_final
    snapshots["temp_state_final"] = T_final

    return snapshots, step, rel_delta


def backward_euler_solver(
    T_init: np.ndarray,
    alpha_map: np.ndarray,
    *,
    dx: float = 1e-4,
    max_iterations: int = 5000,
    convergence_tolerance: float = 1e-4,
    boundary_condition: str = "neumann",
    save_states_predicate: Any = lambda _: False,
) -> Tuple[Dict[str, np.ndarray], int, float]:
    """
    Implicit Backward Euler solver with LU factorization.
    """
    H, W = T_init.shape
    N = H * W

    alpha_flat = alpha_map.ravel()
    dt = 0.4 * dx*dx / (4 * alpha_flat.mean())

    L = _build_implicit_laplacian(H, W, dx, boundary_condition=boundary_condition)
    A = sparse.eye(N) - dt * sparse.diags(alpha_flat) @ L
    lu = sparse.linalg.splu(A.tocsc())

    snapshots = {"temp_state_initial": T_init.copy()}

    T_prev = T_init.ravel().copy()

    for step in range(1, max_iterations + 1):
        T_curr = lu.solve(T_prev)

        abs_change = np.linalg.norm(T_curr - T_prev)
        rel_delta = abs_change / np.linalg.norm(T_prev) if np.linalg.norm(T_prev) > 0 else -1.0

        if rel_delta < convergence_tolerance:
            break

        if save_states_predicate(step):
            snapshots[f"temp_state_{step}"] = T_curr.reshape(H, W).copy()

        T_prev[:] = T_curr

    T_final = T_curr.reshape(H, W).copy()
    snapshots[f"temp_state_{step}"] = T_final
    snapshots["temp_state_final"] = T_final

    return snapshots, step, rel_delta


_SOLVER_FUNCTIONS = {
    "ftcs": ftcs_solver,
    "crank_nicolson":  crank_nicolson_solver,
    "backward_euler":  backward_euler_solver,
}
DEFAULT_SOLVER_NAME = "crank_nicolson"
SOLVER_NAMES = list(_SOLVER_FUNCTIONS.keys())

def get_solver_function(solver_name: str) -> Any:
    if solver_name not in SOLVER_NAMES:
        raise ValueError(f"Solver name {solver_name} doesn't exist")
    return _SOLVER_FUNCTIONS.get(solver_name)