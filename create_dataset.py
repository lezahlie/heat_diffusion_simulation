from setup_logger import setup_logger, set_logger_level
logger = setup_logger(__file__, log_stdout=True, log_stderr=True)
from arguments import process_args
from utilities import *
from heat_simulation import generate_heat_records

# processes function gets shape maps and saves to a file in chunks
def process_image_maps(data_file, 
                        seed_range, 
                        seed_step, 
                        shared_data, 
                        shared_lock,
                        **kwargs):

    remove_if_exists(data_file)

    for sr in split_seed_range(seed_range, seed_step):
        sim_results = generate_heat_records(min_seed=sr[0], max_seed=sr[1], **kwargs)
        # save the data chunk to the hdf5 file
        save_to_hdf5(sim_results, data_file, seed_step)

        # update the global min and max for all scalers and images
        local_stats = compute_local_stats(sim_results)
        update_shared_data(local_stats, shared_data, shared_lock)

# combines all results files into one big file
def gather_task_results(task_data_paths, final_file, seed_chunk):
    cp_pid = current_process().pid
    logger.info(f"PID[{cp_pid}]: Combining results files from each task into one file")
    remove_if_exists(final_file)
    combine_hdf5_files(task_data_paths, final_file, seed_chunk)
    logger.info(f"PID[{cp_pid}]: Saved combined shape maps to: {final_file}")


# creates and runs each process
def run_processes(task_data_paths, seed_range_per_task, seed_step, simulation_kwargs):
    # shared data is to track global min and max for normalizing data 
    # this is useful for normalizing now or in the dataloader later
    manager = Manager()
    shared_data = manager.dict() 
    shared_lock = Lock()

    procs_list = []
    for i, (task_file, seed_range) in enumerate(zip(task_data_paths, seed_range_per_task)):
        p_args = [task_file, seed_range, seed_step, shared_data, shared_lock]  # Pass shared dict to each process
        p = Process(target=process_image_maps, name=f"heat_simulation_p{i}", args=p_args, kwargs=simulation_kwargs)
        procs_list.append(p)
        p.start()
        logger.info(f"PID[{p.pid}]: child started running simulations for seeds {seed_range}")

    for p in procs_list:
        p.join()
        logger.info(f"PID[{p.pid}]: child joined parent")

    return dict(shared_data)



def create_dataset(args):
    
    if args.debug_on:
        set_logger_level(10)

    req_cores = getattr(args, 'num_tasks') 
    min_seed = getattr(args, 'min_seed') 
    max_seed = getattr(args, 'max_seed') 
    seed_step = getattr(args, 'seed_step') 

    grid_length = getattr(args, 'grid_length') 
    convergence_tolerance = getattr(args, 'convergence_tolerance')

    max_iterations = getattr(args, 'max_iterations') 

    boundary_condition = getattr(args, 'boundary_condition') 
    solver_name = getattr(args, 'solver_name') 
    pattern_name = getattr(args, 'pattern_name') 

    save_states = getattr(args, 'save_states', []) 

    total_seeds = (max_seed-min_seed+1)

    # creates output folder and data file prefix
    output_folder_path = os_path.join(args.output_path, args.output_folder)
    data_path = create_folder(output_folder_path)
    datafile_prefix = f"{DATATYPE_NAME}_{grid_length}x{grid_length}"

    arguments_file_path = f"arguments_{datafile_prefix}_{min_seed}-{max_seed}.json"
    save_to_json(os_path.join(output_folder_path, arguments_file_path), vars(args))

    # split up shapes between tasks(cores)
    seed_range_per_task = split_seed_range((min_seed, max_seed), total_seeds// req_cores)
    task_data_paths = [f"{data_path}/{datafile_prefix}_{seed_range[0]}-{seed_range[1]}.{DEFAULT_DATAFILE_EXT}" for seed_range in seed_range_per_task]

    # start process tasks
    logger.info(f"PID[{current_process().pid}]: parent process")
    
    simulation_kwargs = {
        "grid_length": grid_length, 
        "max_iterations": max_iterations,
        "convergence_tolerance": convergence_tolerance,
        "boundary_condition":boundary_condition,
        "solver_name": solver_name,
        "pattern_name": pattern_name,
        "save_states": save_states
    }

    global_stats = run_processes(task_data_paths, seed_range_per_task, seed_step, simulation_kwargs)

    # combine process results
    if req_cores > 1:
        final_file_path = f"{data_path}/{datafile_prefix}_{min_seed}-{max_seed}.{DEFAULT_DATAFILE_EXT}"
        gather_task_results(task_data_paths, final_file_path, seed_step)
    else:
        final_file_path = task_data_paths[0]

    global_stats_file_name = os_path.basename(final_file_path).split('.')[0]
    global_stats_file_path = f"global_statistics_{DEFAULT_DATAFILE_EXT}_{global_stats_file_name}.json"
    save_to_json(os_path.join(output_folder_path, global_stats_file_path), global_stats)


if __name__ == "__main__":
    try:
        args = process_args(__file__)
        create_dataset(args)
    except Exception as e:
        logger.error(e)
