from utilities import np, ndimage, Tuple, Mapping, Dict, Optional


MATERIAL_DB: Dict[int, Dict[str, float]] = {
    0: dict(
        name="air",
        k=0.03,     
        rho=1.2,    
        cp=1000.0,
        q_lb=0.0,
        q_ub=0.0,
        t_lb=20.0,
        t_ub=20.0,
        prob=0.1
    ),
    1: dict(
        name="dielectric",
        k=0.25,     
        rho=2200.0, 
        cp=700.0,
        q_lb=0.0,
        q_ub=0.0,
        t_lb=20.0,
        t_ub=25.0,
        prob=0.2
    ),
    2: dict(
        name="copper",
        k=400.0,    
        rho=8960.0, 
        cp=385.0,
        q_lb=1e3,
        q_ub=5e3,
        t_lb=20.0,
        t_ub=80.0,
        prob=0.25
    ),
    3: dict(
        name="silicon",
        k=150.0, 
        rho=2330.0, 
        cp=705.0,
        q_lb=5e3,
        q_ub=2e4,
        t_lb=20.0,
        t_ub=125.0,
        prob=0.25
    ),
    4: dict(
        name="graphite",
        k=140.0,
        rho=2250.0, 
        cp=710.0,
        q_lb=0.0,
        q_ub=0.0,
        t_lb=20.0,
        t_ub=150.0,
        prob=0.1
    ),
    5: dict(
        name="ceramic",
        k=3.0, 
        rho=3800.0, 
        cp=750.0,
        q_lb=0.0,
        q_ub=0.0,
        t_lb=20.0,
        t_ub=100.0,
        prob=0.1
    )
}

for props in MATERIAL_DB.values():
    props["alpha"] = props["k"] / (props["rho"] * props["cp"])

MIN_TEMP_VALUE = min([MATERIAL_DB[mid]["t_lb"] for mid in MATERIAL_DB])
MAX_TEMP_VALUE = max([MATERIAL_DB[mid]["t_ub"] for mid in MATERIAL_DB])

MATERIAL_IDS = list(MATERIAL_DB.keys())
MATERIAL_NAMES = [MATERIAL_DB[mid]["name"] for mid in MATERIAL_IDS]
MATERIAL_PROBS = [MATERIAL_DB[mid]["prob"] for mid in MATERIAL_IDS]


PERLIN_OVERRIDE = {"air", "dielectric", "graphite", "ceramic"}
MATERIAL_PATTERNS = {'mosaic', 'perlin', 'mixed'}
BOUNDARY_CONDITIONS = {"dirichlet", "neumann", "periodic"}

def get_random_pattern(rng: np.random.Generator) -> str:
    return rng.choice(list(MATERIAL_PATTERNS))

def _voronoi_mosaic(
    rng: np.random.Generator,
    grid: int,
    n_grains: Optional[int] = None
) -> np.ndarray:
    # voronoi mosaic
    if n_grains is None:
        n_grains = max(8, grid * grid // 8)
    pts = rng.integers(0, grid, size=(n_grains, 2))
    y, x = np.mgrid[:grid, :grid]
    return ((y[..., None] - pts[:, 1])**2 + (x[..., None] - pts[:, 0])**2).argmin(axis=-1)



def _perlin_noise(
    rng: np.random.Generator, 
    grid: int,
    octaves_lo: int = 2,
    octaves_hi: Optional[int] = None,
) -> np.ndarray:
    # perlin noise
    if octaves_hi is None:
        octaves_hi = max(octaves_lo, int(np.log2(grid)) - 1)

    num_oct = int(rng.integers(octaves_lo, octaves_hi+1))
    noise = np.zeros((grid, grid))
    amp = 1.0
    for o in range(num_oct):
        step = 2**(num_oct - o)
        small = rng.random((max(grid // step, 1),)*2) * amp
        noise += ndimage.zoom(small, step, order=1)
        amp *= 0.5
    noise -= noise.min()
    noise /= np.ptp(noise)

    n = len(MATERIAL_DB)
    thresholds = np.linspace(1/n, 1, n-1)
    return np.digitize(noise, thresholds).astype(np.uint8)



def generate_material_map(
    rng: np.random.Generator,
    pattern: str,
    grid_size: int,
    perlin_override: set[str] | None = PERLIN_OVERRIDE
) -> np.ndarray:

    """
    pattern:
        "mosaic": pure fine grains
        "perlin": pure smooth clusters
        "mixed": smooth clusters selected materials and fine grains overlaid
    perlin_override: e.g. {"air", "dielectric", "graphite", "ceramic"} 
    """

    tot = sum(MATERIAL_PROBS)
    if tot <= 0:
        raise ValueError("MATERIAL_PROBS_BY_NAME gave zero total probability")

    p = [pi / tot for pi in MATERIAL_PROBS]

    # helper to pick per-region IDs under some pattern
    def pick_regions(region_id: np.ndarray) -> np.ndarray:
        pick = rng.choice(MATERIAL_IDS, size=region_id.max()+1, p=p)
        return pick[region_id]

    if pattern == "mosaic":
        region_id = _voronoi_mosaic(rng, grid_size)
        return pick_regions(region_id).astype(np.uint8)

    if pattern == "perlin":
        return _perlin_noise(rng, grid_size)

    if pattern == "mixed":
        # 2) coarse overlay: Perlin clusters
        coarse = pick_regions(_perlin_noise(rng, grid_size))

        # 1) base regions: fine-grained Voronoi
        base = pick_regions(_voronoi_mosaic(rng, grid_size))

        override_names = perlin_override or set()
        override_ids = list({mid for mid in MATERIAL_IDS if MATERIAL_DB[mid]["name"] in override_names})

        mask = np.isin(coarse, override_ids)
        mixed = base.copy()
        mixed[mask] = coarse[mask]

        return mixed.astype(np.uint8)
    
    raise ValueError(f"unknown style {pattern!r}")


def material_to_source(
    rng: np.random.Generator,
    material_map: np.ndarray,
) -> np.ndarray:
    """
    Returns a volumetric heat‐generation array (W/m³),
    sampling each cell’s power ∈ [q_lb, q_ub] from MATERIAL_DB.
    """
    H, W = material_map.shape
    heat = np.zeros((H, W), dtype=float)

    for mat_id, props in MATERIAL_DB.items():
        mask = (material_map == mat_id)
        q_lb = props["q_lb"]
        q_ub = props["q_ub"]
        if q_ub > q_lb:
            heat[mask] = rng.uniform(q_lb, q_ub, size=mask.sum())

    return heat

def apply_boundary_condition(
    T: np.ndarray,
    boundary_condition: str,
    fixed_value: float = MIN_TEMP_VALUE
) -> np.ndarray:
    H, W = T.shape
    T_bc = T.copy()

    if boundary_condition == "dirichlet":
        T_bc[0, :]  = fixed_value
        T_bc[-1, :] = fixed_value
        T_bc[:, 0]  = fixed_value
        T_bc[:, -1] = fixed_value

    elif boundary_condition == "periodic":
        T_bc[0, :]   = T_bc[-2, :]
        T_bc[-1, :]  = T_bc[1, :]
        T_bc[:, 0]   = T_bc[:, -2]
        T_bc[:, -1]  = T_bc[:, 1]

    elif boundary_condition == "neumann":
        T_bc[0, :]   = T_bc[1, :]
        T_bc[-1, :]  = T_bc[-2, :]
        T_bc[:, 0]   = T_bc[:, 1]
        T_bc[:, -1]  = T_bc[:, -2]

    else:
        raise ValueError(f"Unknown BC: {boundary_condition!r}")

    return T_bc


def generate_initial_temperature(
    rng: np.random.Generator,
    material_map: np.ndarray,
    source_map: np.ndarray,
    t_inject: float = 5.0,
    smooth_sigma: float = 1.0
) -> np.ndarray:
    """
    For each cell:
      - sample T_ambient ~ Uniform(t_lb, t_ub) for that material
      - compute ΔT = Q * t_inject / (rho * cp)
      - return T0 = T_ambient + ΔT
    """
    H, W = material_map.shape
    T0 = np.empty((H, W), float)

    for mid, props in MATERIAL_DB.items():
        mask = (material_map == mid)

        # 1) sample ambient from [t_lb, t_ub]
        lo, hi = props["t_lb"], props["t_ub"]
        if hi > lo:
            T_amb = rng.uniform(lo, hi, size=mask.sum())
        else:
            T_amb = np.full(mask.sum(), lo)

        # 2) compute ΔT = (Q·t_inject)/(ρ·cₚ)
        rho, cp = props["rho"], props["cp"]
        Cv = rho * cp          # J/(m³·K)
        ΔT = source_map[mask] * t_inject / Cv

        # 3) combine
        T0[mask] = T_amb + ΔT

    if smooth_sigma > 0:
        T0 = ndimage.gaussian_filter(T0, sigma=smooth_sigma, mode='reflect')

    return T0

def create_heat_maps(
    rng: np.random.Generator,
    pattern:str,
    grid_size: int,
    boundary_condition: str = "neumann",
    fixed_value: float = MIN_TEMP_VALUE,
    smooth_sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """Return (material_map, heat_mask)."""

    material_map = generate_material_map(rng, pattern, grid_size)
    source_map = material_to_source(rng, material_map)
    initial_raw = generate_initial_temperature(rng, material_map, source_map, smooth_sigma=smooth_sigma)

    initial_map = apply_boundary_condition(initial_raw, boundary_condition, fixed_value=fixed_value)

    alpha_map = np.zeros_like(material_map, dtype=float)
    for mid, props in MATERIAL_DB.items():
        alpha_map[material_map == mid] = props["alpha"]

    return initial_map, alpha_map, source_map, material_map