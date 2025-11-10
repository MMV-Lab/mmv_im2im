import numpy as np
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from tqdm import tqdm
import os
from typing import Callable


def Im2Fourier(image: np.ndarray, mode: str = "complex") -> np.ndarray:
    """
    Computes the 2D Fourier Transform for a multi-channel image and returns the result
    in different formats based on the specified mode.

    Args:
        image (np.ndarray): The input image as a NumPy array of shape (C, Y, X).
        mode (str): The desired output format. Must be one of "complex", "real", or "freq".
                    - "complex": Returns a matrix of shape (2C, Y, X) with the real and
                                 imaginary parts concatenated along the channel axis.
                    - "real": Returns a matrix of shape (C, Y, X) with only the real
                              part of the Fourier Transform.
                    - "freq": Returns a matrix of shape (2C, Y, X) with the magnitude
                              (abs) and phase (angle) for each channel.

    Returns:
        np.ndarray: The transformed image matrix according to the specified mode.
                    Returns None if the mode is invalid.
    """

    allowed_modes = ["complex", "real", "freq"]
    if mode not in allowed_modes:
        raise ValueError(f"Invalid mode: '{mode}'. Must be one of {allowed_modes}.")

    if image.ndim != 3:
        raise ValueError("Input image must be a 3D array of shape (C, Y, X).")

    C, Y, X = image.shape

    transformed_channels = []

    for c in range(C):

        channel = image[c, :, :]

        f_transform = np.fft.fft2(channel)

        f_transform_shifted = np.fft.fftshift(f_transform)

        if mode == "complex":
            real_part = np.real(f_transform_shifted)
            imag_part = np.imag(f_transform_shifted)
            transformed_channels.append(real_part)
            transformed_channels.append(imag_part)

        elif mode == "real":
            real_part = np.real(f_transform_shifted)
            transformed_channels.append(real_part)

        elif mode == "freq":
            magnitude = np.abs(f_transform_shifted)
            phase = np.angle(f_transform_shifted)
            transformed_channels.append(magnitude)
            transformed_channels.append(phase)

    return np.stack(transformed_channels, axis=0)


def Fourier2Im(fourier_image: np.ndarray, mode: str = "complex") -> np.ndarray:
    """
    Computes the 2D Inverse Fourier Transform for a multi-channel image and
    returns the result. It automatically deduces the number of original channels
    based on the input shape and mode.

    Args:
        fourier_image (np.ndarray): The input Fourier-transformed image. Its shape
                                    depends on the specified mode.
        mode (str): The format of the input. Must be one of "complex", "real", or "freq".

    Returns:
        np.ndarray: The reconstructed image matrix of shape (C, Y, X).
    """
    allowed_modes = ["complex", "real", "freq"]
    if mode not in allowed_modes:
        raise ValueError(f"Invalid mode: '{mode}'. Must be one of {allowed_modes}.")

    if fourier_image.ndim != 3:
        raise ValueError("Input image must be a 3D array of shape (N, Y, X).")

    N, Y, X = fourier_image.shape

    # Deduce the number of original channels based on the mode
    if mode == "complex":
        if N % 2 != 0:
            raise ValueError(
                f"For mode 'complex', the number of channels ({N}) must be even."
            )
        C = N // 2
    elif mode == "freq":
        if N % 2 != 0:
            raise ValueError(
                f"For mode 'freq', the number of channels ({N}) must be even."
            )
        C = N // 2
    else:  # mode == "real"
        C = N

    reconstructed_channels = []

    if mode == "complex":
        for i in range(C):
            real_part = fourier_image[2 * i, :, :]
            imag_part = fourier_image[2 * i + 1, :, :]
            f_transform_shifted = real_part + 1j * imag_part
            f_transform = np.fft.ifftshift(f_transform_shifted)
            reconstructed_channel = np.fft.ifft2(f_transform)
            reconstructed_channels.append(np.real(reconstructed_channel))

    elif mode == "real":
        for i in range(C):
            f_transform_shifted = fourier_image[i, :, :]
            f_transform = np.fft.ifftshift(f_transform_shifted)
            reconstructed_channel = np.fft.ifft2(f_transform)
            reconstructed_channels.append(np.real(reconstructed_channel))

    elif mode == "freq":
        for i in range(C):
            magnitude = fourier_image[2 * i, :, :]
            phase = fourier_image[2 * i + 1, :, :]
            f_transform_shifted = magnitude * np.exp(1j * phase)
            f_transform = np.fft.ifftshift(f_transform_shifted)
            reconstructed_channel = np.fft.ifft2(f_transform)
            reconstructed_channels.append(np.real(reconstructed_channel))

    return np.stack(reconstructed_channels, axis=0)


def discretizer2n(matrix: np.ndarray, n: int = 3) -> np.ndarray:
    """
    Quantizes the values of a float matrix to a specified number of integer ranges.

    The thresholds are calculated by dividing the range of matrix values
    into n equal parts.

    Args:
        matrix (np.ndarray): The input matrix with float values.
        n (int): The number of integer levels to quantize to (e.g., 3 for 0, 1, 2).

    Returns:
        np.ndarray: A new matrix with the quantized integer values (from 0 to n-1).
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if not isinstance(n, int) or n < 2:
        raise ValueError("n_levels must be an integer greater than or equal to 2.")

    if matrix.size == 0:
        return np.array([], dtype=int)

    min_val = np.min(matrix)
    max_val = np.max(matrix)

    if min_val == max_val:
        return np.full_like(matrix, int(n / 2), dtype=int)

    data_range = max_val - min_val
    step = data_range / n

    thresholds = [min_val + i * step for i in range(1, n)]

    quantized_matrix = np.zeros_like(matrix, dtype=int)

    for i in range(n - 1):
        quantized_matrix[matrix >= thresholds[i]] = i + 1

    return quantized_matrix.astype(np.uint8)


def Vol2Fourier(volume: np.ndarray, mode: str = "complex") -> np.ndarray:
    """
    Applies the 2D Fourier Transform to each image (slice) of a volume and
    returns a new volume of transformations.

    Args:
        volume (np.ndarray): The input volume as a 4D NumPy array
                             with shape (Z, C, Y, X).
        mode (str): The desired output format. Must be one of "complex",
                    "real", or "freq".

    Returns:
        np.ndarray: The transformed volume with shape (Z, C', Y, X), where C'
                    is the number of channels of the transformation.
    """
    if volume.ndim != 4:
        raise ValueError("Input volume must be a 4D array with shape (Z, C, Y, X).")

    Z, C, Y, X = volume.shape
    transformed_slices = []

    for z in range(Z):
        image_slice = volume[z, :, :, :]
        transformed_slice = Im2Fourier(image_slice, mode=mode)
        transformed_slices.append(transformed_slice)

    return np.stack(transformed_slices, axis=0)


def Fourier2Vol(fourier_volume: np.ndarray, mode: str = "complex") -> np.ndarray:
    """
    Applies the 2D Inverse Fourier Transform to each slice of a transformed volume,
    reconstructing the original volume.

    Args:
        fourier_volume (np.ndarray): The input volume with the Fourier transform applied,
                                     with shape (Z, N, Y, X).
        mode (str): The format of the input. Must be one of "complex",
                    "real", or "freq".

    Returns:
        np.ndarray: The reconstructed volume with shape (Z, C, Y, X).
    """
    if fourier_volume.ndim != 4:
        raise ValueError("Input volume must be a 4D array with shape (Z, N, Y, X).")

    Z, _, Y, X = fourier_volume.shape
    reconstructed_slices = []

    for z in range(Z):
        fourier_slice = fourier_volume[z, :, :, :]
        reconstructed_slice = Fourier2Im(fourier_slice, mode=mode)
        reconstructed_slices.append(reconstructed_slice)

    return np.stack(reconstructed_slices, axis=0)


def VolDiscretizer(volume: np.ndarray, n: int = 3) -> np.ndarray:
    """
    Applies discretization to each image (slice) of a volume,
    quantizing float values to n integer ranges.

    Args:
        volume (np.ndarray): The input volume as a 4D array with
                             shape (Z, C, Y, X).
        n (int): The number of integer levels to quantize to
                 (e.g., 3 for 0, 1, 2).

    Returns:
        np.ndarray: The new volume with discretized values as a 4D integer array.
    """
    if volume.ndim != 4:
        raise ValueError("Input volume must be a 4D array with shape (Z, C, Y, X).")

    Z, C, Y, X = volume.shape
    discretized_slices = []

    for z in range(Z):
        image_slice = volume[z, :, :, :]
        discretized_channels = []
        for c in range(C):
            channel_slice = image_slice[c, :, :]
            discretized_channel = discretizer2n(channel_slice, n=n)
            discretized_channels.append(discretized_channel)

        discretized_slices.append(np.stack(discretized_channels, axis=0))

    return np.stack(discretized_slices, axis=0)


def FreqNorm(fourier_image: np.ndarray) -> np.ndarray:
    """
    Normalizes the magnitude and phase from the output of Im2Fourier with mode 'freq'.

    The magnitude is normalized using a logarithmic scale followed by a Min-Max
    normalization to the range [0, 1]. The phase is scaled to the range [0, 1].

    Args:
        fourier_image (np.ndarray): The input Fourier-transformed matrix with shape
                                     (2C, Y, X), where even channels are magnitude
                                     and odd channels are phase.

    Returns:
        np.ndarray: The output matrix with the same shape, but with normalized
                    magnitude and phase channels.

    Raises:
        ValueError: If the number of channels is not even.
    """
    if fourier_image.ndim != 3:
        raise ValueError("Input image must be a 3D array with shape (N, Y, X).")

    N, Y, X = fourier_image.shape
    if N % 2 != 0:
        raise ValueError(f"The number of channels ({N}) must be even for mode 'freq'.")

    C = N // 2
    normalized_channels = []

    for i in range(C):
        # Separate magnitude and phase for each channel
        magnitude = fourier_image[2 * i, :, :]
        phase = fourier_image[2 * i + 1, :, :]

        # Magnitude normalization: logarithm + Min-Max
        log_magnitude = np.log1p(magnitude)
        min_log_mag = np.min(log_magnitude)
        max_log_mag = np.max(log_magnitude)

        if max_log_mag == min_log_mag:
            norm_magnitude = np.zeros_like(log_magnitude)
        else:
            norm_magnitude = (log_magnitude - min_log_mag) / (max_log_mag - min_log_mag)

        # Phase normalization: Scale to the range [0, 1]
        # np.angle returns phase in the range [-pi, pi]
        norm_phase = (phase + np.pi) / (2 * np.pi)

        normalized_channels.append(norm_magnitude)
        normalized_channels.append(norm_phase)

    return np.stack(normalized_channels, axis=0)


def VolFreqNorm(fourier_volume: np.ndarray) -> np.ndarray:
    """
    Applies frequency-based normalization to each slice of a Fourier-transformed volume.

    This function normalizes the magnitude and phase for each 2D image (slice)
    within a 4D volume, using the same logic as FreqNorm.

    Args:
        fourier_volume (np.ndarray): The input volume with the Fourier transform applied,
                                     with shape (Z, 2C, Y, X).

    Returns:
        np.ndarray: The normalized volume with the same shape.

    Raises:
        ValueError: If the input volume is not a 4D array or if the number of
                    channels is not even.
    """
    if fourier_volume.ndim != 4:
        raise ValueError("Input volume must be a 4D array with shape (Z, N, Y, X).")

    Z, N, Y, X = fourier_volume.shape
    if N % 2 != 0:
        raise ValueError(f"The number of channels ({N}) must be even for mode 'freq'.")

    normalized_slices = []

    for z in range(Z):
        fourier_slice = fourier_volume[z, :, :, :]
        normalized_slice = FreqNorm(fourier_slice)
        normalized_slices.append(normalized_slice)

    return np.stack(normalized_slices, axis=0)


def transform_files(
    path_data: str,
    path_out: str,
    function: Callable[[np.ndarray], np.ndarray],
    mode: str = "complex",
    shape: str = "CYX",
) -> None:
    """
    Applies a function to all image files in a directory and saves the modified images.

    This function iterates through all files in the specified input directory,
    loads each image, applies a user-defined function to its data, and then
    saves the modified image to a new output directory.

    Args:
        path_data (str): The path to the directory containing the input image files.
        path_out (str): The path to the directory where the modified images will be saved.
        function (Callable[[np.ndarray], np.ndarray]): A function that takes a NumPy array
                                                        (the image data) and returns a
                                                        NumPy array (the transformed data).
        shape (str, optional): The dimension order of the image to be loaded.
                                Defaults to 'CYX'.

    Raises:
        TypeError: If path_data, path_out, or shape are not strings.
        TypeError: If the 'function' argument is not a callable object.
        ValueError: If a provided path does not exist.
    """

    if (
        not isinstance(path_data, str)
        or not isinstance(path_out, str)
        or not isinstance(shape, str)
    ):
        raise TypeError("path_data, path_out, and shape must be strings.")

    if not isinstance(function, Callable):
        raise TypeError("'function' must be a callable object.")

    if not os.path.isdir(path_data):
        raise ValueError(f"The input directory '{path_data}' does not exist.")

    if not os.path.exists(path_out):
        print(f"Creating output directory: {path_out}")
        os.makedirs(path_out)

    files = os.listdir(path_data)
    if not files:
        print(f"Warning: No files found in the directory: {path_data}")
        return

    for file in tqdm(files):
        try:
            im = BioImage(os.path.join(path_data, file)).get_image_data(shape, T=0)
            c_im = function(im, mode)
            OmeTiffWriter.save(c_im, os.path.join(path_out, file), dim_order=shape)
        except Exception as e:
            print(f"Error processing file '{file}': {e}")
