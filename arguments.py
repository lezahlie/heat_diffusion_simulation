import argparse as ap
import utilities as util
from heat_solvers import SOLVER_NAMES
from heat_mappers import MATERIAL_PATTERNS, BOUNDARY_CONDITIONS, MIN_TEMP_VALUE

executable_groups = {
    "create_dataset.py": ["simulation", "batch", "output", "multiprocess"],
    "visualize_dataset.py": ["visualize"]
}


def parse_tuple(value):
    return tuple(map(int, value.strip("()").split(",")))

def parse_save_states(s: str | None):
    if not isinstance(s, str):
        return None

    s = s.strip().lower()

    # If no save states are required
    if s == 'none':
        return []

    conditions = []
    # Split the input by commas to handle multiple chained conditions
    for condition in s.split(','):
        condition = condition.strip()

        # Check for 'all'
        if 'all' == condition:
            conditions.append('all')

        # Check for 'interval-T'
        m = util.re.match(r'interval-(\d+)', condition)
        if m:
            t = int(m.group(1))
            if t < 1:
                raise ap.ArgumentTypeError("interval-<T> must be > 0")
            conditions.append(('interval', t))

        # Check for 'first-N'
        m = util.re.match(r'first-(\d+)', condition)
        if m:
            n = int(m.group(1))
            if n < 1:
                raise ap.ArgumentTypeError("first-<N> must be > 0")
            conditions.append(('first', n))

        # Check for 'base-B'
        m = util.re.match(r'base-(\d+)', condition)
        if m:
            b = int(m.group(1))
            if b < 2:
                raise ap.ArgumentTypeError("base-<B> must be > 1")
            conditions.append(('base', b))

    if not conditions:
        raise ap.ArgumentTypeError(
            f"invalid save-states '{s}'; choose none|all|first-<N>|interval-<N>|base-<N>"
        )
    return conditions


def add_multiprocess_group(parser, file_name):
    group = parser.add_argument_group('multi-process options')
    group.add_argument('--ntasks', dest="num_tasks", type=int, default=1, 
                    help="Number of parallel tasks (CPU cores). Max threads = 2 x NUM_TASKS if multithreading is on (default: 1)")
    if "simulation" in executable_groups[file_name]:
        group.add_argument('--seed-step', dest="seed_step", type=int, default=50, 
                        help="Number of seeds to be processed and written at a time | default: 100")


def check_multiprocess_args(args):
    if not (1 <= args.num_tasks < util.cpu_count()):
        raise ap.ArgumentError(None, f"NUM_TASKS must be a INT between [1, {util.cpu_count()} - 1]")


def add_batch_group(parser):
    group = parser.add_argument_group("batch processing options")
    group.add_argument('--min-seed', dest='min_seed', type=int, default=1,
                            help="Start seed for generating simulations | default: 1")
    group.add_argument('--max-seed', dest='max_seed', type=int, default=5,
                            help="End seed for generating simulations | default: 5")


def check_batch_args(args):
    if not (1 <= args.min_seed < args.max_seed):
        raise ap.ArgumentError(None, "min_seed must be a INT between [1, MAX_SEED]")
    
    if not (args.min_seed < args.max_seed):
        raise ap.ArgumentError(None, "max_seed must be a INT greater than MIN_SEED.")
    
    total_seeds = args.max_seed-args.min_seed+1
    if hasattr(args, 'seed_step') and not (0 < args.seed_step < total_seeds):
        raise ap.ArgumentError(None, f"seed_step must be a INT between [1, {total_seeds}]")


def add_simulation_group(parser):
    group = parser.add_argument_group("simulation behavior options")
    
    group.add_argument('--grid-length', dest='grid_length', type=int, default=32,
                            help="Length of one side of 2D grid | default: 32")
    
    group.add_argument('--max-iterations', dest='max_iterations', type=int, default=5000,
                            help="Maximum Euler integration steps | default: 5000")
    
    group.add_argument('--convergence-tolerance', dest='convergence_tolerance', type=float, default=1e-4,
                            help="Tolerance value for determining convergence | default: 1e-4")
    
    group.add_argument(
        "--solver-name",
        dest="solver_name",
        type=str,
        choices=SOLVER_NAMES,
        default="crank_nicolson",
        help=(
            "Which solver to use:\n"
            "   ftcs           - explicit; 1st-order time, 2nd-order space\n"
            "   crank-nicolson - implicit; 2nd-order time, unconditionally stable\n"
            "   backward-euler - implicit; 1st-order time, unconditionally stable\n"
            "(default: crank-nicolson)"
        )
    )

    group.add_argument(
        "--boundary-condition",
        dest="boundary_condition",
        type=str,
        choices=BOUNDARY_CONDITIONS,
        default="neumann",
        help=(
            "Which boundary condition to use:\n"
            f"  - dirichlet   fixed temperature ({MIN_TEMP_VALUE} C) at edges \n"
            "   - neumann     zero flux (insulated) at edges\n"
            "   - periodic    wrap-around at edges\n"
            "(default: neumann)"
        )
    )

    group.add_argument(
        "--pattern-name",
        dest="pattern_name",
        type=str,
        choices=list(MATERIAL_PATTERNS)+['none'],
        default=None,
        help=(
            "Which pattern for generating material maps:\n"
            "   mosaic  - fine Voronoi grains \n"
            "   perlin  - clustered Perlin-noise regions\n"
            "   mixed   - perlin layered with mosaic\n"
            "(default: random pattern determined by seed)"
        )
    )


def check_simulation_args(args):
    if not (4 < args.grid_length < 1025):
        raise ap.ArgumentError(None, "GRID_LENGTH must be a INT inclusively between [5, 1024]")

    if not (0 < args.max_iterations < 1e+9):
        raise ap.ArgumentError(None, "max_iterations must be a INT greater between [1, 1E9].")
    
    if not (0 < args.convergence_tolerance < 1.0):
        raise ap.ArgumentError(None, "convergence_tolerance must be a FLOAT greater between [0, 1.0].")


def add_visualize_group(parser):
    group = parser.add_argument_group('visualization options')

    group.add_argument('--data-file', dest='data_file', type=str, required=True,
        help="Path to the input HDF5 (with saved runs under top-level groups)")

    group.add_argument('--random-seed', dest='random_seed', type=int,
                            help="Random seed for selecting samples | default: None")
    
    group.add_argument('--num-samples', dest='num_samples', type=int, default=5,
                            help="Number simulation samples to visualize | default: 1")
    
    group.add_argument('--output-folder', dest='output_folder', type=str, required=True,
        help="Directory where visualizations will be written (created if needed)")

    group.add_argument('--gif-fps', dest='gif_fps', type=int, default=20,
        help="Frames per second for the output gif | default: 20")
    
    group.add_argument('--gif-delay', dest='gif_delay', type=int, default=2,
        help="Delay seconds between loops for the output gif | default: 2")

    group.add_argument('--gif-cmap', dest='gif_cmap', type=str, default='RdYlBu_r',
        help="Colormap for gif plots (perceptually uniform preferred)  | default: 'RdYlBu_r'")

    group.add_argument('--image-cmap', dest='image_cmap', type=str, default='turbo',
        help="Colormap for static image plots (diverging preferred) | default: 'turbo'")


def check_visualize_args(args):

    if not util.os_path.exists(args.data_file):
        raise FileNotFoundError(f"data_file '{args.data_file}' does not exist")
    
    if not (1 <= args.num_samples <= 1e9):
        raise ap.ArgumentError(None, f"NUM_SAMPLES must an INT between [1, 1e9]")
    
    if not (1 <= args.random_seed <= 1e9):
        raise ap.ArgumentError(None, f"RANDOM_SEED must an INT between [1, 1e9]")
    
    if util.os_path.exists(args.output_folder) and not util.os_path.isdir(args.output_folder):
        raise ap.ArgumentError(None, f"OUTPUT_FOLDER '{args.output_folder}' exists but is not a directory.")

    if not (1 <= args.gif_fps <= 120):
        raise ap.ArgumentError(None, f"GIF_FPS must be an INT between [1, 120].")
    
    if not (0 < args.gif_delay <= 60):
        raise ap.ArgumentError(None, f"GIF_DELAY must be an INT between [0, 60].")
    
    mpl_colormaps = ','.join(util.plt.colormaps())
    if not isinstance(args.gif_cmap, str) or args.gif_cmap.strip() == "":
        raise ap.ArgumentError(None, "GIF_CMAP must be a matplotlib colormap string (e.g. 'turbo', 'inferno', 'viridis', etc.).")
    
    elif args.gif_cmap not in util.plt.colormaps():
        raise ap.ArgumentError(None, f"Invalid matplotlib colormap '{args.gif_cmap}'. Colormaps: {mpl_colormaps}") 
    
    if not isinstance(args.image_cmap, str) or args.image_cmap.strip() == "":
        raise ap.ArgumentError(None, "IMAGE_CMAP must be a matplotlib colormap string (e.g. 'seismic', 'RdYlBu', 'Spectral', etc.).")
    
    elif args.image_cmap not in util.plt.colormaps():
        raise ap.ArgumentError(None, f"Invalid matplotlib colormap '{args.image_cmap}'. Colormaps: {mpl_colormaps}") 



def add_output_group(parser):
    group = parser.add_argument_group('output path options')
    group.add_argument('--output-path', dest='output_path', type=str, default=util.os_path.dirname(util.os_path.abspath(__file__)),
                        help="Path the the directory to create [--output-folder] and save to | default: current directory")
    
    group.add_argument('--output-folder', dest='output_folder', type=str,
                        help="Output folder name to create and save simulation data to | default: greyscott_dataset")

    group.add_argument('--save-states', dest='save_states',
        type=parse_save_states,
        default=None,
            help=(
                "When to save intermediate states:\n"
                "   all            - every iteration\n"
                "   interval-<T>   - every Nth iteration; e.g. interval-10\n"
                "   first-<N>      - first N states; e.g. first-10\n"
                "   base-<B>       - powers of B: 1, B, B², ...; base-2\n"
                "   all            - every iteration\n"
                "   none           - no intermediate states are saved\n"
                'Multiple options can be chained, e.g. "first-<T>,interval-<N>,base-<B>"'
            )
    )


def check_output_args(args, filename):
    if hasattr(args, 'output_path') and not util.os_path.exists(args.output_path):
        raise FileNotFoundError(f"OUTPUT_PATH '{args.output_path}' does not exist")
    
    if isinstance(args.output_folder, str) and not (0 < len(args.output_folder) < 129):
        raise ap.ArgumentError(None, f"OUTPUT_FOLDER '{args.output_folder}' must have a length between [1, 128]")
    
    if filename == 'create_dataset.py' and args.output_folder is None:
        raise ap.ArgumentError(None, f"OUTPUT_FOLDER '{args.output_folder}' is required for new dataset creation")



def check_args(parser, file_name):
    args = parser.parse_args()

    if "multiprocess" in executable_groups[file_name]:
        check_multiprocess_args(args)
    if "batch" in executable_groups[file_name]:
        check_batch_args(args)
    if "simulation" in executable_groups[file_name]:
        check_simulation_args(args)
    if "visualize" in executable_groups[file_name]:
        check_visualize_args(args)
    if "output" in executable_groups[file_name]:
        check_output_args(args, file_name)
    return args


def process_args(exe_file):
    file_name = util.os_path.basename(exe_file)
    parser = ap.ArgumentParser(description="Heat Diffusion Simulation", formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('-d', '--debug', dest='debug_on', action='store_true', 
                        help="Enables logging with debug level verbosity | default: false")

    if "multiprocess" in executable_groups[file_name]:
        add_multiprocess_group(parser, file_name)
    if "batch" in executable_groups[file_name]:
        add_batch_group(parser)
    if "simulation" in executable_groups[file_name]:
        add_simulation_group(parser)
    if "visualize" in executable_groups[file_name]:
        add_visualize_group(parser)
    if "output" in executable_groups[file_name]:
        add_output_group(parser)

    args = check_args(parser, file_name)

    return args