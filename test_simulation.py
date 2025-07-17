import pytest

from setup_logger import setup_logger, set_logger_level
logger = setup_logger(__file__, log_stdout=True, log_stderr=True)
from heat_mappers import *
from heat_solvers import *
from visualize_dataset import *
from utilities import *



def simulate_patterns(
    seed:int,
    grid_length: int,
    max_iterations: int,
    convergence_tolerance: float,
    boundary_condition:str,
    solver_name: str,
    save_states: Optional[Tuple[str, int]|Tuple[str]] = [("first", 20), ("interval", 100)],
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng(seed)
    save_states_predicate = create_save_states_predicate(save_states)

    pattern_records = []
    for pattern_name in MATERIAL_PATTERNS:
        solver_fn = get_solver_function(solver_name)

        initial_state, alpha_map, source_map, material_map = create_heat_maps(rng, pattern_name, grid_length)
        save_states_predicate = create_save_states_predicate(save_states)


        temp_states, total_iterations, relative_delta = solver_fn(
            initial_state, 
            alpha_map,
            max_iterations=max_iterations,
            convergence_tolerance=convergence_tolerance,
            boundary_condition=boundary_condition,
            save_states_predicate=save_states_predicate,
        )

        logger.debug(f"final state returned after {total_iterations}/{max_iterations} steps")
        meta: Dict[str, Any] = {
            'random_seed': seed,
            'grid_length': grid_length,
            "convergence_tolerance": convergence_tolerance,
            'max_iterations': max_iterations,
            'total_iterations': total_iterations,
            'pattern_name': pattern_name,
            'relative_delta': relative_delta
        }

        images: Dict[str, Any] = {
            "material_map": material_map,
            "source_map": source_map,
            "diffusion_map": alpha_map,
            **temp_states
        }

        pattern_records.append({'image': images, 'meta': meta})


    return pattern_records

def visualize_patterns(record, output_path):
    meta_data = record['meta']

    pattern = meta_data['pattern_name']
    seed = meta_data['random_seed']

    title = f"{pattern.title().replace('_',' ')} — Seed #{seed}"
    file_prefix = f"{DATATYPE_NAME}_{pattern}_{seed}"
    data_prefix = "temp"

    file_paths = []
    frames, steps = extract_record_frames(record, prefix=data_prefix)
    gif_path = os_path.join(output_path,  f"{file_prefix}_{data_prefix}_states.gif")
    unit_label = r"$^\circ\mathrm{C}$"
    gif_title = fr"T States ({unit_label}): {title}"

    save_record_frames(frames, 
                        steps, 
                        fps=15, 
                        delay=3, 
                        cmap="coolwarm", 
                        title=gif_title, 
                        file_path=gif_path)
    logger.info(f"Saved test {data_prefix} gif → {gif_path}")
    file_paths.append(str(gif_path))

    image_path = os_path.join(output_path,  f"{file_prefix}_input_output.png")

    save_record_images(
        record,
        cmap="turbo",
        title=title,
        file_path=image_path
    )
    file_paths.append(image_path)

    logger.info(f"Saved test image → {image_path}")
    return file_paths


@pytest.mark.parametrize("seed", list(range(1, 3)))
@pytest.mark.parametrize("samples", [None])
@pytest.mark.parametrize("boundary_condition", BOUNDARY_CONDITIONS)
@pytest.mark.parametrize("solver_name", SOLVER_NAMES)

def test_all_patterns(seed, samples, boundary_condition, solver_name):

    set_logger_level(10)

    sim_args = {
        'grid_length': 32,
        'max_iterations': 5000,
        'convergence_tolerance': 1e-4,
        'boundary_condition': boundary_condition,
        'solver_name': solver_name,
        'save_states': [("first", 20), ("interval", 100)]
    }

    output_folder = create_folder(os_path.join("test_results", f"{solver_name}_solver", f"{boundary_condition}_bc"))
    dataset_file = os_path.join(output_folder, f"{DATATYPE_NAME}_{boundary_condition}_{solver_name}_{seed}.hdf5")

    logger.info(f"Running Grey-Scott pattern test with seed = {seed}, samples = {samples}")
    
    result_records = simulate_patterns(seed, **sim_args)
    assert len(result_records) > 0, "No patterns were simulated"

    logger.info(f"Saving {len(result_records)} simulations to: {dataset_file}")
    save_to_hdf5(result_records, dataset_file)
    with h5py.File(dataset_file, "r") as h5f:
        logger.info(f"HDF5 contains {len(h5f.keys())} records")
        assert len(h5f.keys()) >= len(MATERIAL_PATTERNS), "Not all patterns were saved"

    sample_records = read_from_hdf5(dataset_file, sample_size=samples, flatten=False, random_seed=seed)

    logger.info(f"Visualizing {len(sample_records)} sampled records")
    for record in sample_records:
        file_paths = visualize_patterns(record, output_folder)
        for fp in file_paths:
            assert os_path.exists(fp), f"Cannot find file path '{fp}'"