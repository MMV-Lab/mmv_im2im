import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import linregress
import math

class Slice_windows(nn.Module):
    """
    A PyTorch module to count windows with ones or calculate the average Shannon entropy
    of windows in an input tensor.

    Args:
        num_kernels (int): The number of kernel sizes to use. Must be between 1 and 10.
                           The kernel sizes will be 2^1, 2^2, ..., 2^num_kernels.
        mode (str): The operational mode of the module.
                    - "classic": Counts the number of windows that contain at least one '1'.
                    - "entropy": Calculates the average Shannon entropy for each window.
                                 H = -p0 * log2(p0) - p1 * log2(p1), where p0 and p1 are
                                 the probabilities of 0s and 1s in the window, respectively.
        to_binary (bool): If True, the input tensor will be converted to a binary tensor
                          where all values > 0 become 1, and 0 remains 0. Defaults to False.
    """
    def __init__(self, num_kernels: int, mode: str = "classic", to_binary: bool = False):
        super().__init__()
        if not (1 <= num_kernels <= 10):
            raise ValueError("num_kernels must be between 1 and 10 to avoid excessively large kernels.")
        self.num_kernels = num_kernels
        # Kernel sizes are powers of 2, from 2^1 to 2^num_kernels
        self.kernel_sizes = [2**i for i in range(1, num_kernels + 1)]

        if mode not in ["classic", "entropy"]:
            raise ValueError(f"Mode must be 'classic' or 'entropy', but got '{mode}'.")
        self.mode = mode
        self.to_binary = to_binary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input tensor based on the selected mode.

        Args:
            x (torch.Tensor): The input tensor, expected to be 4D (Batch, Channels, H, W).

        Returns:
            torch.Tensor: A 2D tensor (Batch, num_kernels) containing the results.
                          If mode is "classic", it contains the count of windows with ones.
                          If mode is "entropy", it contains the average entropy of windows.
        """
        if x.dim() != 4:
            raise ValueError(f"Input must be a 4D tensor (Batch, Channels, H, W), but got {x.dim()}D.")
        
        # Convert to binary if to_binary is True
        if self.to_binary:
            x = (x > 0).float() # Convert to float to maintain tensor type for subsequent operations
                                # Values > 0 become 1.0, 0 remains 0.0

        batch_size, channels, H, W = x.shape
        # Initialize results tensor to store output for each batch and kernel size
        results = torch.zeros(batch_size, self.num_kernels, device=x.device)

        # Iterate through each defined kernel size
        for i, kernel_size in enumerate(self.kernel_sizes):
            # If kernel size is larger than image dimensions, skip and set result to 0
            if kernel_size > H or kernel_size > W:
                results[:, i] = 0.0 # Use 0.0 for consistency with entropy which can be float
                continue
            
            # Calculate the total number of elements in a single window
            elements_in_window = kernel_size * kernel_size * channels

            # Process each item in the batch
            for b in range(batch_size):
                # Unfold the input tensor into overlapping windows.
                # x[b:b+1] is used to maintain the batch dimension for F.unfold
                unfolded_windows = F.unfold(x[b:b+1], 
                                            kernel_size=(kernel_size, kernel_size), 
                                            stride=(kernel_size, kernel_size))
                
                # Transpose and squeeze to get shape (num_windows, elements_in_window)
                # where num_windows is the total number of non-overlapping windows
                unfolded_windows = unfolded_windows.transpose(1, 2).squeeze(0)

                if self.mode == "classic":
                    # Classic mode: Count windows containing at least one '1'
                    # Sum elements in each window, if sum > 0, it has at least one '1'
                    windows_with_ones = (unfolded_windows.sum(dim=1) > 0).float()
                    current_count = torch.sum(windows_with_ones).item()
                    results[b, i] = current_count
                
                elif self.mode == "entropy":
                    # Entropy mode: Calculate Shannon entropy for each window
                    entropies = []
                    # Iterate over each window
                    for window in unfolded_windows:
                        # Count number of ones in the current window
                        num_ones = window.sum().item()
                        # Count number of zeros in the current window
                        num_zeros = elements_in_window - num_ones

                        # Calculate probabilities
                        p1 = num_ones / elements_in_window
                        p0 = num_zeros / elements_in_window

                        # Shannon entropy calculation
                        # H = -p0 * log2(p0) - p1 * log2(p1)
                        # Handle log2(0) case: if p is 0, p * log2(p) is considered 0
                        entropy = 0.0
                        if p0 > 0:
                            entropy -= p0 * math.log2(p0)
                        if p1 > 0:
                            entropy -= p1 * math.log2(p1)
                        entropies.append(entropy)
                    
                    # Calculate the average entropy for the current batch item and kernel size
                    if entropies: # Ensure there are entropies to average
                        results[b, i] = sum(entropies) / len(entropies)
                    else:
                        results[b, i] = 0.0 # No windows, so entropy is 0

        return results


class FractalDimension(nn.Module):
    def __init__(self, num_kernels: int, mode: str = "classic", to_binary: bool = False):
        """
        Initializes the layer to estimate a scaling exponent based on box characteristics.

        Args:
            num_kernels (int): The number of kernels to use for the underlying box-counting/entropy layer.
            mode (str): The operational mode for the underlying Slice_windows layer.
                        - "classic": The estimator calculates the traditional box-counting fractal dimension.
                        - "entropy": The estimator calculates a scaling exponent related to how
                                     average Shannon entropy changes with box size.
            to_binary (bool): If True, the input tensor will be converted to a binary tensor
                              where all values > 0 become 1, and 0 remains 0. This parameter
                              is passed directly to the Slice_windows layer. Defaults to False.
        """
        super().__init__()
        # Pass the mode and to_binary directly to the Slice_windows layer
        self.count_layer = Slice_windows(num_kernels=num_kernels, mode=mode, to_binary=to_binary)
        self.kernel_sizes = self.count_layer.kernel_sizes # Get kernel sizes from the sub-layer

        # Store the mode to clarify the output interpretation
        self.mode = mode 
        self.to_binary = to_binary # Store it for potential external inspection, though not strictly used internally in forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates a scaling exponent for each image in the batch.

        Args:
            x (torch.Tensor): The input image. If `to_binary` was set to True during initialization,
                              this input will be converted to binary internally by `Slice_windows`.
                              Expected format: (Batch_size, Channels, Height, Width)

        Returns:
            torch.Tensor: A tensor of size (Batch_size,) where each element
                          is the estimated scaling exponent for the corresponding image.
                          If mode is "classic", this is the box-counting fractal dimension.
                          If mode is "entropy", this is an entropy-based scaling exponent.
                          Returns 0.0 if regression cannot be performed (e.g., not enough valid points).
        """
        # Get the counts/entropies from the underlying layer based on the selected mode
        # The to_binary conversion happens inside self.count_layer if self.to_binary was True
        results_per_kernel = self.count_layer(x)
        
        batch_size = x.shape[0]
        scaling_exponents = torch.zeros(batch_size, device=x.device, dtype=torch.float32)

        # Calculate the scaling exponent for each image in the batch
        for b in range(batch_size):
            current_results = results_per_kernel[b].cpu().numpy()
            
            # Calculate inverse kernel sizes for the x-axis of the log-log plot
            inverse_kernel_sizes = np.array([1 / k for k in self.kernel_sizes])

            # Filter valid points for regression: only consider points where the result is greater than 0
            # This avoids issues with log(0) which would lead to -infinity
            valid_indices = current_results > 0
            
            if np.sum(valid_indices) < 2: # Need at least 2 points for a linear regression
                scaling_exponents[b] = 0.0 # Cannot estimate scaling exponent
                continue

            # Take the logarithm of the results and inverse kernel sizes
            if self.mode == 'classic':
                log_results = np.log(current_results[valid_indices])
                log_inverse_kernel_sizes = np.log(inverse_kernel_sizes[valid_indices])
            if self.mode == 'entropy':
                log_results = current_results[valid_indices]
                log_inverse_kernel_sizes = np.log(inverse_kernel_sizes[valid_indices])
                
            try:
                # Perform linear regression: the slope is the scaling exponent
                slope, intercept, r_value, p_value, std_err = linregress(log_inverse_kernel_sizes, log_results)
                scaling_exponents[b] = torch.tensor(slope, dtype=torch.float32, device=x.device)
            except ValueError:
                # This can happen if there are issues with the input to linregress (e.g., all same values)
                scaling_exponents[b] = 0.0
            except Exception as e:
                print(f"Error during linregress for batch {b}: {e}")
                scaling_exponents[b] = 0.0

        return scaling_exponents