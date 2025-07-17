
from setup_logger import setup_logger
logger = setup_logger(__file__, log_stdout=True, log_stderr=True)
from utilities import Any, List, Tuple, Dict, Optional, np, makedirs, os_path, create_save_states_predicate
from heat_mappers import *
from heat_solvers import *


def run_heat_simulation(
    seed:int,
    *,
    grid_length: int,
    max_iterations: int,
    convergence_tolerance: float,
    boundary_condition: str,
    solver_name: str,
    pattern_name:str,
    save_states: Optional[Tuple[str, int]|Tuple[str]] = []
) -> Dict[str, Any]:

    """Run a single heat simulation given sim_config and a random seed

    Args:
        seed (Optional[int], optional): random seed. Defaults to None.
        grid_size (int): length of grid in pixels
        max_iterations (int, optional): maximum euler steps. Defaults to 1000.
        convergence_tolerance (float): convergence_tolerance for convergence
        save_states (Optional[Tuple[str, int] | Tuple[str]], optional): save states predicate config list. Defaults to [("first", 20), ("interval", 10)].

    Returns:
        Dict[str, Any]: 
            - 'meta': configuration and seed
            - 'image': contains u_init, v_init, optional v_frames, and u_final/v_final
    """
    rng = np.random.default_rng(seed)

    if pattern_name not in MATERIAL_PATTERNS:
        pattern_name = get_random_pattern(rng)

    initial_state, alpha_map, source_map, material_map = create_heat_maps(
        rng, 
        pattern_name, 
        grid_length,
        boundary_condition=boundary_condition
    )

    save_states_predicate = create_save_states_predicate(save_states)
    solver_fn = get_solver_function(solver_name)

    temp_states, total_iterations, relative_delta = solver_fn(
        initial_state,
        alpha_map,
        max_iterations=max_iterations,
        convergence_tolerance=convergence_tolerance,
        boundary_condition=boundary_condition,
        save_states_predicate=save_states_predicate,
    )

    meta: Dict[str, Any] = {
        'random_seed': seed,
        'grid_length': grid_length,
        "convergence_tolerance": convergence_tolerance,
        'max_iterations': max_iterations,
        'total_iterations': total_iterations,
        'relative_delta': relative_delta,
        'pattern_name': pattern_name,
        'boundary_condition': boundary_condition
    }

    images: Dict[str, Any] = {
        "material_map": material_map,
        "source_map": source_map,
        "diffusion_map": alpha_map,
        **temp_states
    }

    return {'image': images, 'meta': meta}

###############################################################################
# Generate batches of records
###############################################################################

def generate_heat_records(
    min_seed: int,
    max_seed: int,
    *,
    grid_length: int,
    max_iterations: int,
    convergence_tolerance: float,
    boundary_condition: str,
    solver_name: str,
    pattern_name: str,
    save_states: List
) -> List[Dict[str, Any]]:
    """Run a batch of several heat simulation given sim_config and a random seed

    Args:
        min_seed (int): minimum random seed (inclusive)
        max_seed (int): maximum random seed (inclusive)
        grid_size (int): length of grid in pixels
        max_iterations (int, optional): maximum euler steps. Defaults to 1000.
        patch_radius (int): radius in pixels per patch
        patch_prob (float): probability a patch is placed or not

    Returns:
        List[Dict[str, Any]]: List of simulation records
    """

    sim_config = {
        'grid_length': grid_length,
        'max_iterations': max_iterations,
        'convergence_tolerance': convergence_tolerance,
        "boundary_condition": boundary_condition,
        "solver_name": solver_name,
        "pattern_name": pattern_name,
        'save_states': save_states,
    }

    results: List[Dict[str, Any]] = []
    for seed in range(min_seed, max_seed + 1):
        result = run_heat_simulation(seed, **sim_config)
        results.append(result)

    return results



def main():
    output_folder = f"heat_example"
    makedirs(output_folder, exist_ok=True)

    for pattern in MATERIAL_PATTERNS:
        for sn in SOLVER_NAMES:
            for bc in BOUNDARY_CONDITIONS:
                
                for seed in range(1, 6):
                    print(f"Simulating pattern {pattern} seed #{seed}")
                    args = {
                        "seed": seed,
                        "grid_length": 32,
                        "convergence_tolerance": 1e-5,
                        "max_iterations": 5000,
                        "boundary_condition": bc,
                        "solver_name": sn,
                        "pattern_name": pattern,
                        "save_states": []
                    }

                    record = run_heat_simulation(**args)
                    
                    from visualize_dataset import save_record_images
                    save_record_images(
                        record,
                        file_path=os_path.join(output_folder, f"{pattern}_seed_{seed}.png"),
                        title="Initial vs Final Temperature",
                    )


if __name__ == "__main__":
    main()
