import logging
import wave
from decimal import ROUND_HALF_UP, Decimal, localcontext

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


def load_wav_file(
    input_file_path: str, num_channels_to_keep: int = 1
) -> tuple[np.array, int]:
    """
    Load WAV file returning an array of the data and the frame rate.

    Args:
        input_file_path (str): WAV input file
        num_channels_to_keep (int, optional): number of channels to keep. Defaults to 1.

    Raises:
        ValueError: number of channels must be 1 or 1

    Returns:
        tuple[np.array, int]: signal, frame_rate
    """
    if num_channels_to_keep not in (1, 2):
        raise ValueError(
            f"num_channels_to_keep ({num_channels_to_keep}) must be 1 or 2"
        )

    # read wave file
    with wave.open(input_file_path, "r") as mix_wave:
        raw_signal = mix_wave.readframes(-1)
        frame_rate = mix_wave.getframerate()
        num_channels = mix_wave.getnchannels()

    if num_channels not in (1, 2):
        raise ValueError(f"num_channels ({num_channels}) must be 1 or 2")

    # covert into numpy array
    signal = np.fromstring(raw_signal, "int16")
    if num_channels == 2:
        if num_channels_to_keep == 1:
            # keep left signal
            signal = signal[::2]
        else:
            # keep both and reshape array
            signal = signal.reshape([len(signal) / 2, 2])

    return signal, frame_rate


def convert_frames_to_seconds(time_frames: int, frame_rate: int = 44100) -> float:
    """
    Convert from number of frames to number of seconds

    Args:
        time_frames (int): number of frames
        frame_rate (int, optional): frame rate. Defaults to 44100.

    Returns:
        float: time_seconds
    """
    return time_frames / frame_rate


def convert_seconds_to_frames(time_seconds: float, frame_rate: int = 44100) -> int:
    """
    Convert from time in seconds to frame number

    Args:
        time_seconds (float): time
        frame_rate (int, optional): frame rate. Defaults to 44100.

    Returns:
        int: frame_number
    """
    # round the "normal" way.
    # NOTE: the built-in 'round' defaults to banker's rounding (i.e. round(0.5) = 0)
    with localcontext() as ctx:
        ctx.rounding = ROUND_HALF_UP
        frame_number = Decimal(time_seconds * frame_rate).to_integral_value()
        frame_number = int(frame_number)

    return frame_number


def get_start_frame(
    arr: np.array, start: tuple[None, int, float], unit: str, frame_rate: int
) -> int:
    """
    Provided a start number and its units, determine the start frame.  If start is None, then
    the start frame is zero.

    Args:
        arr (np.array): input audio array
        start (tuple[None, int, float]): start number.  If None, then start frame is zero.
        unit (str): "frame" or "second"
        frame_rate (int): frame rate (typical frame rate is 44100)

    Raises:
        ValueError: unit not recognized
        ValueError: start_frame out of bounds

    Returns:
        int: start_frame
    """
    if unit not in ("frame", "second"):
        return ValueError(f"unit '{unit}' not recognized")

    # calculate start_frame
    if start is None:
        start_frame = 0
    else:
        if unit == "frame":
            start_frame = start
        elif unit == "second":
            start_frame = convert_seconds_to_frames(start, frame_rate)

    # verify that start_frame is in bounds
    if not (0 <= start_frame < len(arr)):
        raise ValueError(f"start_frame ({start_frame}) is out of bounds")

    return start_frame


def get_end_frame(
    arr: np.array, end: tuple[None, int, float], unit: str, frame_rate: int
) -> int:
    """_summary_

    Args:
        arr (np.array): input audio array
        end (tuple[None, int, float]): end number.  If None, then end frame is last index.
        unit (str): "frame" or "second"
        frame_rate (int): frame rate (typical frame rate is 44100)

    Raises:
        ValueError: unit not recognized
        ValueError: end_frame out of bounds

    Returns:
        int: end_frame
    """
    if unit not in ("frame", "second"):
        return ValueError(f"unit '{unit}' not recognized")

    # calculate end_frame
    if end is None:
        end_frame = len(arr)
    else:
        if unit == "frame":
            end_frame = end
        elif unit == "second":
            end_frame = convert_seconds_to_frames(end, frame_rate)

    # verify that end_frame is in bounds
    if not (0 <= end_frame < len(arr)):
        raise ValueError(f"end_frame ({end_frame}) is out of bounds")

    return end_frame


def clip_wav_arr(
    arr: np.array,
    start: tuple[None, int, float] = None,
    end: tuple[None, int, float] = None,
    unit: str = "frame",
    frame_rate: int = 44100,
    channels=1,
) -> np.array:
    """
    Clip WAV array in range [start, end] (inclusive) in provided units.

    Args:
        arr (np.array): input audio array
        start (tuple[None, int, float]): start number.  If None, then start frame is zero.
        end (tuple[None, int, float]): end number.  If None, then end frame is last index.
        unit (str): "frame" or "second"
        frame_rate (int): frame rate (typical frame rate is 44100)
        channels (int, optional): number of audio channels (currently only implemente for 1
            channel). Defaults to 1.

    Raises:
        ValueError: arr cannot be empty
        NotImplementedError: _description_
        ValueError: start_frame > end_frame

    Returns:
        np.array: clipped_arr
    """
    # check arguments
    if len(arr) == 0:
        raise ValueError("arr is an empty array")
    if channels != 1:
        raise NotImplementedError("Only channels=1 has been implemented")
    if start is None and end is None:
        logging.warning("Both start and end are None, which results in no clipping")

    start_frame = get_start_frame(arr, start, unit, frame_rate)
    end_frame = get_end_frame(arr, end, unit, frame_rate)

    if start_frame > end_frame:
        raise ValueError(
            f"start_frame ({start_frame}) is greater than end_frame ({end_frame})"
        )

    return arr[start_frame : end_frame + 1]


def save_wave_arr(
    signal: np.array, output_file_path: str, frame_rate: int = 44100
) -> None:
    """
    Save numpy array to WAV file

    Args:
        signal (np.array): array to save
        output_file_path (str): path to which to save the file
        frame_rate (int, optional): frame rate for arr. Defaults to 44100.
    """
    wavfile.write(output_file_path, frame_rate, signal)


def normalize_wav_arr(arr: np.array, to_frac_max: float = 1.0) -> np.array:
    """
    Normalize data so that the largest |value| is at the max signal value.

    Args:
        arr (np.array): input array to normalize
        to_frac_max (float, optional): The largest |value| after normalization will be at
            to_frac_max * MAX.  MAX is the largest value allowed for a 16-bit integer (2**16 - 1).
            to_frac_max must be in range [0.0, 1.0].  Defaults to 1.0.

    Raises:
        ValueError: to_frac_max is out of range

    Returns:
        np.array: norm_arr
    """
    if len(arr.shape) != 1:
        raise NotImplementedError("have not implemented for dual channel")
    if not (0.0 <= to_frac_max <= 1.0):
        raise ValueError(f"to_frac_max ({to_frac_max}) must be in the range [0.0, 1.0]")
    min_val = arr.min()
    max_val = arr.max()
    largest_abs_val = max(abs(min_val), abs(max_val))

    new_largest_abs_val = (2**15 - 1) * to_frac_max
    factor = new_largest_abs_val / largest_abs_val
    norm_arr = arr * factor
    norm_arr = norm_arr.astype(np.int16)

    return norm_arr


def mix_wav_arrs(
    arr_list: np.array, frac_list: tuple[list[float], None] = None
) -> np.array:
    """
    Add a list of WAV arrays s.t. the fraction of the total amplitude at each point corresponds to
    fractions provided in frac_list.  If frac_list is not provided, they will be mixed in equal
    proportion.

    Args:
        arr_list (np.array): input list of arrays to combine
        frac_list (tuple[list[float], None], optional): Fraction of total amplitude to which each
            input array should be mixed. If None, then mix in equal proportion. Defaults to None.

    Raises:
        ValueError: input arrays < 2
        ValueError: frac_list values do not add to 1.0

    Returns:
        np.array: _description_
    """
    if len(arr_list) < 2:
        raise ValueError(
            f"Minimum number of arrays required to mix is 2.  {len(arr_list)} arrays provided"
        )
    if frac_list is None:
        frac_list = [1.0 / len(arr_list) for _ in range(len(arr_list))]
    else:
        if sum(frac_list) != 1.0:
            raise ValueError(
                f"the values in frac_list do not sum to 1.0. frac_list = {frac_list}"
            )

    normalized_arr_list = []
    for arr, frac in zip(arr_list, frac_list):
        norm_arr = normalize_wav_arr(arr, to_frac_max=frac)
        normalized_arr_list.append(norm_arr)

    mixed_arr = np.sum(normalized_arr_list, axis=0, dtype=np.int16)

    return mixed_arr


def plot_wave_arr(
    arr: np.array,
    frame_rate: int = 44100,
    title: tuple[str, None] = None,
    ax=None,
    color: str = "#3ABFE7",
    alpha: float = 1.0,
):
    """
    Plot WAV array

    Args:
        arr (np.array): array to plot
        frame_rate (int, optional): Defaults to 44100.
        title (tuple[str, None], optional): Defaults to None.
        ax (_type_, optional): matplotlib axis. Defaults to None.
        color (str, optional): color of plot. Defaults to "#3ABFE7".
        alpha (float, optional): opacity. Defaults to 1.0.

    Returns:
        _type_: matplotlib axis
    """

    time_points = np.linspace(0, len(arr) / frame_rate, num=len(arr))

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 2))

    ax.plot(time_points, arr, c=color, alpha=alpha)
    ax.set_ylim(-35000, 35000)
    if title is not None:
        ax.set_title(title)
    return ax


def compare_arrays(orig_arr1: np.array, orig_arr2: np.array) -> float:
    """
    Generate a comparison matrix with the root mean squared deviation (RMSD) between input arrays.

    Args:
        orig_arr1 (np.array):
        orig_arr2 (np.array):

    Raises:
        ValueError: arrays must be the same length

    Returns:
        float: rmsd
    """

    if len(orig_arr1) != len(orig_arr2):
        raise ValueError(
            f"arr1 and arr2 are not the same length ({len(orig_arr1)} vs {len(orig_arr2)})"
        )
    arr1 = orig_arr1.astype(np.float64)
    arr2 = orig_arr2.astype(np.float64)
    return np.sqrt(np.mean((arr1 - arr2) ** 2))


def build_comparison_matrix(arr_list: list[np.array]) -> np.array:
    """
    Build comparison matrix with RMSD between each pair of arrays

    Args:
        arr_list (list[np.array]): list of arrays to compare

    Returns:
        np.array: comparison_matrix
    """
    # TODO: update to not repeat unnecessary calculations
    N = len(arr_list)
    comparison_matrix = np.zeros((N, N))
    for i, arr1 in enumerate(arr_list):
        for j, arr2 in enumerate(arr_list):
            comparison_matrix[i, j] = compare_arrays(arr1, arr2)
    return comparison_matrix


def determine_scaling_factor(
    deconvoluted_arr: np.array, mixed_arr_list: list[np.array]
) -> float:
    """
    Determine the sign of the scaling factor required to best fit the mixed signal arrays. ICA
    cannot determine whether the proper scaling factor (including sign), but we can guess that
    the appropriate sign is the one that fits to the mixed audio signals best.

    Args:
        deconvoluted_arr (np.array): the normalized deconvoluted array to determine the sign for
        mixed_arr_list (list[np.array]): the list of normalized mixed signal arrays

    Returns:
        float: -1.0 or 1.0
    """

    min_comparison = None
    min_factor = None
    for mixed_arr in mixed_arr_list:
        for factor in (-1, 1):
            comparison = compare_arrays(deconvoluted_arr * factor, mixed_arr)
            if min_comparison is None or comparison < min_comparison:
                min_comparison = comparison
                min_factor = factor
    return min_factor
