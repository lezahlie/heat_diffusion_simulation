from setup_logger import setup_logger, set_logger_level
logger = setup_logger(__file__, log_stdout=True, log_stderr=True)
from utilities import (DATATYPE_NAME, DEFAULT_DATAFILE_EXT,
                        List, Tuple, Optional, 
                        os_path, np, re, plt, mcolor, manimate, mtick,
                        read_from_hdf5, read_from_json, create_folder)
from heat_mappers import MATERIAL_IDS
from arguments import process_args

DEFAULT_FPS = 20
DEFAULT_DELAY = 3

def extract_record_frames(record, prefix:str) -> Tuple[np.ndarray, List[int]]:
    """
    Given an h5py Group record with a /image/ and /meta/ subgroup,
    extract all V state frames in chronological order.

    Returns:
        A NumPy array of shape (N, H, W).

    Raises:
        KeyError if no V-frame data is found.
    """
    

    image_dict = record.get("image", {})
    meta_dict = record.get("meta", {})

    pattern = re.compile(fr"{prefix}_state_(\d+)$")
    initial_image_key = f"{prefix}_state_initial"
    final_image_key = f"{prefix}_state_final"
    iteration_meta_key = "total_iterations"

    if initial_image_key not in image_dict:
        raise KeyError(f"Missing '{initial_image_key}' in image group")
    elif final_image_key not in image_dict:
        raise KeyError(f"Missing '{final_image_key}' in image group")
    elif iteration_meta_key not in meta_dict:
        raise KeyError(f"Missing '{iteration_meta_key}' in meta group")
    
    total_iterations = meta_dict[iteration_meta_key]

    # validate initial frame (step = 0)
    initial_state = image_dict[initial_image_key][()]
    if not isinstance(initial_state, np.ndarray) or initial_state.ndim != 2:
        raise ValueError(f"'{initial_image_key}' must be a 2D NumPy array")

    #validate final frame (step = final)
    final_state = image_dict[final_image_key][()]
    if not isinstance(final_state, np.ndarray) or final_state.ndim != 2:
        raise ValueError(f"'{final_image_key}' must be a 2D NumPy array")

    # get intermediate frames and iterations
    states: List[Tuple[int, np.ndarray]] = []
    for key in image_dict.keys():
        m = pattern.match(key)
        if m:
            step = int(m.group(1))
            arr = image_dict[key][()]
            if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                raise ValueError(f"'{key}' must be a 2D NumPy array")
            states.append((step, arr))

    # sort intermediate by iterations
    states.sort(key=lambda x: x[0])

    # build lists for frames and iterations
    steps: List[int] = []
    frames: List[np.ndarray] = []

    # initial (step = 0)
    steps.append(0)
    frames.append(initial_state)

    # intermediate
    for s, arr in states:
        steps.append(s)
        frames.append(arr)

    # final (step = total_iterations)
    steps.append(total_iterations)
    frames.append(final_state)

    # stack all frames into a single 3D array
    frames_stack = np.stack(frames, axis=0)   # shape: (num_frames, N, N)

    return frames_stack, steps


def save_record_frames(
    frames: np.ndarray,
    steps: list,
    *,
    file_path: str,
    fps: int = DEFAULT_FPS,
    delay: int = DEFAULT_DELAY,
    prefix: str = "temp",
    cmap: str = "turbo",
    title: str = "Heat Diffusion"
) -> None:
    """
    Create and save a GIF from a sequence of 2D frames.
    
    Args:
        frames:    np.ndarray of shape (N, H, W).
        steps:     List of length num_frames; steps[i] is the “step index” for frames[i].
        file_path:  Path or filename where the GIF will be written.
        fps:       Playback frames per second (default: 20).
        cmap:      Matplotlib colormap (default: "turbo").
        title:     Main title for the GIF (default: "Heat Diffusion").
        figsize:   Size of each frame in inches as (height, width), [default: (10, 10)].

    Raises:
        ValueError: If `frames` is not 3D.
    """


    if frames.ndim != 3:
        raise ValueError("frames must be a 3D array of shape (N, H, W)")

    N, H, W = frames.shape

    hold_count = int(round(delay * fps))
    frame_indices = list(range(N)) + [N - 1] * hold_count
    total_frames = len(frame_indices)

    min_val = np.nanmin(frames)
    max_val = np.nanmax(frames)
    norm = mcolor.Normalize(vmin=min_val, vmax=max_val, clip=True)
    cbar_ticks = np.linspace(min_val, max_val, 9)

    fig, ax = plt.subplots(figsize=(8.5, 8))
    img = ax.imshow(
        frames[0],
        cmap=cmap,
        norm=norm
    )

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.suptitle(title, fontsize=18)


    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04, ticks=cbar_ticks)
    cbar.set_label(r"$T\; ^\circ\mathrm{C}$", fontsize=14)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3g"))

    title_text = ax.set_xlabel(fr"$state$ = {steps[0]}", fontsize=16, labelpad=20)


    fig.tight_layout()
    
    def _render(i: int):
        idx = frame_indices[i]
        img.set_data(frames[idx])
        img.set_norm(norm)
        title_text.set_text(fr"$state$ = {steps[idx]}")
        return (img, title_text)

    fps2ms = 1000 / fps
    anim = manimate.FuncAnimation(
        fig,
        _render,
        frames= total_frames,
        interval= fps2ms,
        repeat=True,
        blit=True
    )

    writer = manimate.PillowWriter(fps=fps, metadata={"loop": 0})
    anim.save(file_path, writer=writer)
    plt.close(fig)


def save_record_images(
    record: dict,
    *,
    file_path: str,
    prefix: str = "temp",
    cmap: str = "turbo",
    title: str = "Heat Diffusion",
    statistics:Optional[dict]=None
) -> None:

    image_dict = record["image"]
    total_iterations = record["meta"]["total_iterations"]
    stats_dict = statistics.get('image', {}) if statistics else {}


    panels = [
        # (
        #     r"$Mask$: Heat Source $Q$",
        #     "source_map",
        #     r"$\mathrm{Q}\; (\mathrm{W}/\mathrm{m^3})$",
        #     "inferno"
        # ),
        (
            r"$X_1$: Thermal Diffusivity $\alpha$",
            "diffusion_map",
            r"$\mathrm{\alpha}\; (\mathrm{m^2}/\mathrm{s})$",
            "plasma"
        ),
        (
            r"$X_2$: Initial State $T_0$",
            f"{prefix}_state_initial",
            r"$T\; ^\circ\mathrm{C}$",
            cmap
        ),
        (
            fr"$Y$: Final State $T_{{{total_iterations}}}$",
            f"{prefix}_state_final",
            r"$T\; ^\circ\mathrm{C}$",
            cmap
        ),
    ]

    num_cols=len(panels)
    fig, axes = plt.subplots(1, num_cols, figsize=(5.5*num_cols, 5))

    # Each entry: (title, array, vmin, vmax, colormap)
    for ax, (title_i, key_i, label_i, cmap_i) in zip(axes, panels):
        if key_i not in image_dict:
            raise KeyError(f"image key '{key_i}' was not found in dataset")

        img_i = image_dict[key_i]

        stats_i = stats_dict.get(key_i, {})
        min_val = stats_i.get('min', img_i.min())
        max_val = stats_i.get('max', img_i.max())

        im = ax.imshow(img_i, cmap=cmap_i, vmin=min_val, vmax=max_val, aspect='equal')

        ticks = np.linspace(min_val, max_val, 6)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=ticks)

        cbar.set_label(label_i, fontsize=14)
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3g"))

        ax.set_title(title_i, fontsize=16)
        ax.axis("off")
 
    fig.suptitle(title, fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(file_path)
    plt.close(fig)


def visualize_samples(args):
    if args.debug_on:
        set_logger_level(10)

    # all the possible args

    data_file = getattr(args, 'data_file') 
    output_folder = getattr(args, 'output_folder') 
    

    random_seed = getattr(args, 'random_seed') 
    num_samples = getattr(args, 'num_samples') 

    fps = getattr(args, 'gif_fps')
    delay = getattr(args, 'gif_delay')
    gif_cmap = getattr(args, 'gif_cmap')
    image_cmap = getattr(args, 'image_cmap')

    output_folder = create_folder(output_folder)

    data_filename = os_path.basename(data_file)
    json_filename = f"global_statistics_{DEFAULT_DATAFILE_EXT}_{data_filename.replace(DEFAULT_DATAFILE_EXT, 'json')}"
    json_file = data_file.replace(data_filename, json_filename)

    global_statistics = read_from_json(json_file)
    sample_records = read_from_hdf5(data_file, sample_size=num_samples, flatten=False, random_seed=random_seed)

    data_prefix = "temp"


    for record in sample_records:

        meta_dict = record['meta']
        pattern = meta_dict['pattern_name']
        seed = meta_dict['random_seed']
        title = f"{pattern.title().replace('_',' ')} Pattern  — Seed #{seed}"
        file_prefix = f"{DATATYPE_NAME}_{pattern}_{seed}"

        image_path = os_path.join(output_folder,  f"{file_prefix}_compare.png")
        image_title = rf"Heat Diffusion ($X1, \;X2, \;Y$) — {title}"
        save_record_images(
            record, 
            file_path=image_path,
            title=image_title, 
            cmap=image_cmap, 
            statistics=global_statistics,
        )
        logger.info(f"Saved test images → {image_path}")

        frames, steps = extract_record_frames(record, prefix=data_prefix)
        gif_path = os_path.join(output_folder,  f"{file_prefix}_states.gif")
        gif_title = f"T States: {title}"

        save_record_frames(frames, 
                        steps, 
                        file_path=gif_path,
                        fps=fps, 
                        delay=delay, 
                        title=gif_title, 
                        cmap=gif_cmap)

        logger.info(f"Saved test {data_prefix} gif → {gif_path}")


if __name__ == "__main__":
    try:
        args = process_args(__file__)
        visualize_samples(args)
    except Exception as e:
        logger.error(e)