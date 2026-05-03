import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import numpy as np
import os
import itertools
import pickle

def cartesian_product_transpose_pp(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty((la, *map(len, arrays)), dtype=dtype)
    idx = slice(None), *itertools.repeat(None, la)
    for i, a in enumerate(arrays):
        arr[i, ...] = a[idx[:la-i]]
    return arr.reshape(la, -1).T


class CustomWaveDataset(Dataset):
        def __init__(self, input_dim: int = 2, 
                     spectrum_size: int = 5, 
                     num_samples_train: int = 10000, 
                     num_samples_test: int = 500,
                     noise_std: float = 0.1,
                     transform = None,
                     root: str = './fft_data',
                     data_file_suffix: str = None,
                     original_coeffs_file_suffix: str = None,
                     rescale: float = 0.0,
                     ):
            
            self.input_dim = input_dim
            self.spectrum_size = spectrum_size
            self.num_samples_train = num_samples_train
            self.num_samples_test = num_samples_test
            self.noise_std = noise_std
            self.transform = transform
            self.root = root
            self.data_file = os.path.join(root, f"CustomWaveDataset_{input_dim}_{spectrum_size}_{num_samples_train}_{num_samples_test}_{noise_std}_{data_file_suffix}.npz")
            self.original_coeffs_file = os.path.join(root, f"original_fourier_coeff_{input_dim}_{spectrum_size}_{num_samples_train}_{num_samples_test}_{noise_std}_{original_coeffs_file_suffix}.pkl")
            self.num_samples = num_samples_train + num_samples_test
            
            if os.path.exists(self.original_coeffs_file):
                
                if os.path.exists(self.data_file):
                    data = np.load(self.data_file)
                    self.x = data['x']
                    self.y = data['y']
                    print(f"Data loaded from {self.data_file}")
                    print(f"max: {self.y.max()}, min: {self.y.min()}")
                    
                else:
                    with open(self.original_coeffs_file, 'rb') as file:
                        c = pickle.load(file)
                    
                    # Generate and save data
                    x1 = np.random.rand(self.num_samples_train, self.input_dim) * 2 * np.pi - rescale + 1e-5

                    # Create homogeneous grid from 0 to 2π in each dimension
                    grid_points_per_dim = int(np.ceil(self.num_samples_test**(1/self.input_dim)))
                    homogeneous_generators = [np.linspace(0, 2*np.pi, grid_points_per_dim, endpoint=True) - rescale + 1e-5
                                              for _ in range(self.input_dim)]
                    x2_grid = cartesian_product_transpose_pp(homogeneous_generators)

                    # Take only the required number of test samples
                    x2 = x2_grid[:self.num_samples_test]
                    
                    # Concatenate random and homogeneous samples
                    x = np.concatenate([x1, x2], axis=0)
                    
                    if input_dim < 2:
                        full_spectrum_generators = [np.linspace(0, self.spectrum_size-1, self.spectrum_size) for i in range(self.input_dim)]
                        full_spectrum = cartesian_product_transpose_pp(full_spectrum_generators)
                        omega_dot_x = np.einsum('ij,kj->ik', full_spectrum, x)
                        
                        c_real = c.real
                        c_imag = c.imag
                        c_0 = c_real[0]
                        cos_coeffs = c_real * 2
                        cos_coeffs[0] = c_0
                        y = (cos_coeffs @ np.cos(omega_dot_x) + (c_imag * -2) @ np.sin(omega_dot_x)).reshape(self.num_samples, 1)
                        if noise_std > 0:
                            y += np.random.rand(self.num_samples, 1) * noise_std
                        
                        self.x = x
                        self.y = y
                    
                    else:
                        c = c.flatten()
                        full_spectrum_generators = [np.linspace(-self.spectrum_size+1, self.spectrum_size-1, 2*self.spectrum_size-1) for i in range(self.input_dim)]
                        full_spectrum = cartesian_product_transpose_pp(full_spectrum_generators)
                        
                        y = np.zeros(self.num_samples)
                        chunk_size = min(100, self.num_samples)
                        for chunk_start in range(0, self.num_samples, chunk_size):
                            chunk_end = min(chunk_start + chunk_size, self.num_samples)
                            x_chunk = x[chunk_start:chunk_end]
                            omega_dot_x_chunk = np.einsum('ij,kj->ik', full_spectrum, x_chunk)
                            y[chunk_start:chunk_end] = (c @ np.exp(1j*omega_dot_x_chunk)).real
                        
                        y = y.reshape(-1, 1)
                        if noise_std > 0:
                            y += np.random.randn(self.num_samples, 1) * noise_std
                        
                        self.x = x
                        self.y = y
                    
                    np.savez(self.data_file, x=x, y=y)
                    print(f"Data generated and saved to {self.data_file}")
                    print(f"max: {y.max()}, min: {y.min()}")
                    
                    
            else:
                # Generate and save data
                x1 = np.random.rand(self.num_samples_train, self.input_dim) * 2 * np.pi - rescale + 1e-5

                # Create homogeneous grid from 0 to 2π in each dimension
                grid_points_per_dim = int(np.ceil(self.num_samples_test**(1/self.input_dim)))
                homogeneous_generators = [np.linspace(0, 2*np.pi, grid_points_per_dim, endpoint=True) - rescale + 1e-5
                                          for _ in range(self.input_dim)]
                x2_grid = cartesian_product_transpose_pp(homogeneous_generators)

                # Take only the required number of test samples
                x2 = x2_grid[:self.num_samples_test]
                
                # Concatenate random and homogeneous samples
                x = np.concatenate([x1, x2], axis=0)
                
                if input_dim < 2:
                    full_spectrum_generators = [np.linspace(0, self.spectrum_size-1, self.spectrum_size) for i in range(self.input_dim)]
                    full_spectrum = cartesian_product_transpose_pp(full_spectrum_generators)
                    omega_dot_x = np.einsum('ij,kj->ik', full_spectrum, x)
                
                    fourier_coeff_an = np.random.rand(len(full_spectrum))
                    fourier_coeff_bn = np.random.rand(len(full_spectrum))
                    
                    y = (fourier_coeff_bn @ np.sin(omega_dot_x) + fourier_coeff_an @ np.cos(omega_dot_x)).reshape(self.num_samples, 1)
                    if noise_std > 0:
                        y += np.random.rand(self.num_samples, 1) * noise_std
                    
                    c = np.concatenate((np.array([fourier_coeff_an[0]]), 0.5*(fourier_coeff_an[1:] - 1j*fourier_coeff_bn[1:])))
                    c_modulus_sum = np.sum(np.abs(c))
                    c = c / c_modulus_sum
                    
                    y /= c_modulus_sum
                    self.x = x
                    self.y = y
                
                else:
                    
                    full_spectrum_generators = [np.linspace(-self.spectrum_size+1, self.spectrum_size-1, 2*self.spectrum_size-1) for i in range(self.input_dim)]
                    full_spectrum = cartesian_product_transpose_pp(full_spectrum_generators)

                    freq_tuples = [tuple(f) for f in full_spectrum]
                    freq_to_idx = {f: i for i, f in enumerate(freq_tuples)}
                    
                    # Generate random Fourier coefficients with Hermitian symmetry
                    c = np.zeros(len(full_spectrum), dtype=complex)
                    for i, f in enumerate(freq_tuples[:len(freq_tuples)//2 + 1]):
                        neg_f = tuple(-np.array(f))
                        real = np.random.randn()
                        imag = np.random.randn()
                        c[i] = real + 1j * imag
                        if neg_f in freq_to_idx and neg_f != f:
                            c[freq_to_idx[neg_f]] = real - 1j * imag
                    c[len(freq_tuples)//2] = np.random.randn()
                    
                    c_modulus_sum = np.sum(np.abs(c))
                    c = c / c_modulus_sum
                    
                    # Generate y values in chunks to avoid memory issues
                    y = np.zeros(self.num_samples)
                    chunk_size = min(100, self.num_samples)
                    for chunk_start in range(0, self.num_samples, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, self.num_samples)
                        x_chunk = x[chunk_start:chunk_end]
                        omega_dot_x_chunk = np.einsum('ij,kj->ik', full_spectrum, x_chunk)
                        y[chunk_start:chunk_end] = (c @ np.exp(1j*omega_dot_x_chunk)).real
                    
                    y = y.reshape(-1, 1)
                    if noise_std > 0:
                        y += np.random.randn(self.num_samples, 1) * noise_std

                    self.x = x
                    self.y = y
                    # Reshape c to multi-dimensional array
                    c = c.reshape([spectrum_size*2-1]*input_dim)
                    
                with open(f'{root}/original_fourier_coeff_{input_dim}_{spectrum_size}_{num_samples_train}_{num_samples_test}_{noise_std}_{original_coeffs_file_suffix}.pkl', 'wb') as file:
                    pickle.dump(c, file, protocol=pickle.HIGHEST_PROTOCOL)
                    
                np.savez(self.data_file, x=x, y=y)
                print(f"Data generated and saved to {self.data_file}")
                print(f"max: {self.y.max()}, min: {self.y.min()}")
            
        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            x = self.x[idx]
            y = self.y[idx]
            if self.transform:
                x = self.transform(x)
                y = self.transform(y)
            return x, y



class CustomWaveDataset2(Dataset):
        def __init__(self, input_dim: int = 2, 
                     spectrum_size: int = 5, 
                     num_samples_train: int = 10000, 
                     num_samples_test: int = 500,
                     noise_std: float = 0.1,
                     transform = None,
                     root: str = './fft_data',
                     data_file_suffix: str = None,
                     original_coeffs_file_suffix: str = None,
                     rescale: float = 0.0,
                     ):
            
            self.input_dim = input_dim
            self.spectrum_size = spectrum_size
            self.num_samples_train = num_samples_train
            self.num_samples_test = num_samples_test
            self.noise_std = noise_std
            self.transform = transform
            self.root = root
            self.data_file = os.path.join(root, f"CustomWaveDataset_{input_dim}_{spectrum_size}_{num_samples_train}_{num_samples_test}_{noise_std}_{data_file_suffix}.npz")
            self.original_coeffs_file = os.path.join(root, f"original_fourier_coeff_{input_dim}_{spectrum_size}_{num_samples_train}_{num_samples_test}_{noise_std}_{original_coeffs_file_suffix}.pkl")
            self.num_samples = num_samples_train + num_samples_test
            
            if os.path.exists(self.original_coeffs_file):
                
                if os.path.exists(self.data_file):
                    data = np.load(self.data_file)
                    self.x = data['x']
                    self.y = data['y']
                    print(f"Data loaded from {self.data_file}")
                    print(f"max: {self.y.max()}, min: {self.y.min()}")
                    
                else:
                    with open(self.original_coeffs_file, 'rb') as file:
                        c = pickle.load(file)
                    
                    # Generate and save data
                    x1 = np.random.rand(self.num_samples_train, self.input_dim) * 2 * np.pi - rescale + 1e-5

                    # Create homogeneous grid from 0 to 2π in each dimension
                    grid_points_per_dim = int(np.ceil(self.num_samples_test**(1/self.input_dim)))
                    homogeneous_generators = [np.linspace(0, 2*np.pi, grid_points_per_dim, endpoint=True) - rescale + 1e-5
                                              for _ in range(self.input_dim)]
                    x2_grid = cartesian_product_transpose_pp(homogeneous_generators)

                    # Take only the required number of test samples
                    x2 = x2_grid[:self.num_samples_test]
                    
                    # Concatenate random and homogeneous samples
                    x = np.concatenate([x1, x2], axis=0)
                    
                    if input_dim < 2:
                        full_spectrum_generators = [np.linspace(0, self.spectrum_size-1, self.spectrum_size) for i in range(self.input_dim)]
                        full_spectrum = cartesian_product_transpose_pp(full_spectrum_generators)
                        omega_dot_x = np.einsum('ij,kj->ik', full_spectrum, x)
                        
                        c_real = c.real
                        c_imag = c.imag
                        c_0 = c_real[0]
                        cos_coeffs = c_real * 2
                        cos_coeffs[0] = c_0
                        y = (cos_coeffs @ np.cos(omega_dot_x) + (c_imag * -2) @ np.sin(omega_dot_x)).reshape(self.num_samples, 1)
                        if noise_std > 0:
                            y += np.random.rand(self.num_samples, 1) * noise_std
                        
                        self.x = x
                        self.y = y
                    
                    else:
                        c = c.flatten()
                        full_spectrum_generators = [np.linspace(-self.spectrum_size+1, self.spectrum_size-1, 2*self.spectrum_size-1) for i in range(self.input_dim)]
                        full_spectrum = cartesian_product_transpose_pp(full_spectrum_generators)
                        
                        y = np.zeros(self.num_samples)
                        chunk_size = min(100, self.num_samples)
                        for chunk_start in range(0, self.num_samples, chunk_size):
                            chunk_end = min(chunk_start + chunk_size, self.num_samples)
                            x_chunk = x[chunk_start:chunk_end]
                            omega_dot_x_chunk = np.einsum('ij,kj->ik', full_spectrum, x_chunk)
                            y[chunk_start:chunk_end] = (c @ np.exp(1j*omega_dot_x_chunk)).real
                        
                        y = y.reshape(-1, 1)
                        if noise_std > 0:
                            y += np.random.randn(self.num_samples, 1) * noise_std
                        
                        self.x = x
                        self.y = y
                    
                    np.savez(self.data_file, x=x, y=y)
                    print(f"Data generated and saved to {self.data_file}")
                    print(f"max: {y.max()}, min: {y.min()}")
                    
                    
            else:
                # Generate and save data
                x1 = np.random.rand(self.num_samples_train, self.input_dim) * 2 * np.pi - rescale + 1e-5

                # Create homogeneous grid from 0 to 2π in each dimension
                grid_points_per_dim = int(np.ceil(self.num_samples_test**(1/self.input_dim)))
                homogeneous_generators = [np.linspace(0, 2*np.pi, grid_points_per_dim, endpoint=True) - rescale + 1e-5
                                          for _ in range(self.input_dim)]
                x2_grid = cartesian_product_transpose_pp(homogeneous_generators)

                # Take only the required number of test samples
                x2 = x2_grid[:self.num_samples_test]
                
                # Concatenate random and homogeneous samples
                x = np.concatenate([x1, x2], axis=0)
                
                if input_dim < 2:
                    full_spectrum_generators = [np.linspace(0, self.spectrum_size-1, self.spectrum_size) for i in range(self.input_dim)]
                    full_spectrum = cartesian_product_transpose_pp(full_spectrum_generators)
                    omega_dot_x = np.einsum('ij,kj->ik', full_spectrum, x)
                
                    fourier_coeff_an = np.random.rand(len(full_spectrum))
                    fourier_coeff_bn = np.random.rand(len(full_spectrum))
                    
                    y = (fourier_coeff_bn @ np.sin(omega_dot_x) + fourier_coeff_an @ np.cos(omega_dot_x)).reshape(self.num_samples, 1)
                    if noise_std > 0:
                        y += np.random.rand(self.num_samples, 1) * noise_std
                    
                    c = np.concatenate((np.array([fourier_coeff_an[0]]), 0.5*(fourier_coeff_an[1:] - 1j*fourier_coeff_bn[1:])))
                    c_modulus_sum = np.sum(np.abs(c))
                    c[0] += 0.5*c_modulus_sum  # Adjust the DC component to ensure normalization
                    c_modulus_sum_updated = np.sum(np.abs(c))
                    c = c / c_modulus_sum_updated
                    
                    y += 0.5*c_modulus_sum
                    y /= c_modulus_sum_updated
                    self.x = x
                    self.y = y
                
                else:
                    full_spectrum_generators = [np.linspace(-self.spectrum_size+1, self.spectrum_size-1, 2*self.spectrum_size-1) for i in range(self.input_dim)]
                    full_spectrum = cartesian_product_transpose_pp(full_spectrum_generators)

                    freq_tuples = [tuple(f) for f in full_spectrum]
                    freq_to_idx = {f: i for i, f in enumerate(freq_tuples)}
                    
                    # Generate random Fourier coefficients with Hermitian symmetry
                    c = np.zeros(len(full_spectrum), dtype=complex)
                    for i, f in enumerate(freq_tuples[:len(freq_tuples)//2 + 1]):
                        neg_f = tuple(-np.array(f))
                        real = np.random.randn()
                        imag = np.random.randn()
                        c[i] = real + 1j * imag
                        if neg_f in freq_to_idx and neg_f != f:
                            c[freq_to_idx[neg_f]] = real - 1j * imag
                    c[len(freq_tuples)//2] = np.random.randn()
                    
                    c_modulus_sum = np.sum(np.abs(c))
                    c[len(freq_tuples)//2] += 0.5*c_modulus_sum # Adjust the DC component to ensure normalization
                    c_modulus_sum_updated = np.sum(np.abs(c))
                    c = c / c_modulus_sum_updated
                    
                    # Generate y values in chunks to avoid memory issues
                    y = np.zeros(self.num_samples)
                    chunk_size = min(100, self.num_samples)
                    for chunk_start in range(0, self.num_samples, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, self.num_samples)
                        x_chunk = x[chunk_start:chunk_end]
                        omega_dot_x_chunk = np.einsum('ij,kj->ik', full_spectrum, x_chunk)
                        y[chunk_start:chunk_end] = (c @ np.exp(1j*omega_dot_x_chunk)).real
                    
                    y = y.reshape(-1, 1)
                    if noise_std > 0:
                        y += np.random.randn(self.num_samples, 1) * noise_std

                    self.x = x
                    self.y = y
                    # Reshape c to multi-dimensional array
                    c = c.reshape([spectrum_size*2-1]*input_dim)
                    
                with open(f'{root}/original_fourier_coeff_{input_dim}_{spectrum_size}_{num_samples_train}_{num_samples_test}_{noise_std}_{original_coeffs_file_suffix}.pkl', 'wb') as file:
                    pickle.dump(c, file, protocol=pickle.HIGHEST_PROTOCOL)
                    
                np.savez(self.data_file, x=x, y=y)
                print(f"Data generated and saved to {self.data_file}")
                print(f"max: {self.y.max()}, min: {self.y.min()}")
            
        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            x = self.x[idx]
            y = self.y[idx]
            if self.transform:
                x = self.transform(x)
                y = self.transform(y)
            return x, y
        

class CustomWaveDataset3(Dataset):
        def __init__(self, input_dim: int = 2, 
                     spectrum_size: int = 5, 
                     num_samples_train: int = 10000, 
                     num_samples_test: int = 500,
                     noise_std: float = 0.1,
                     coeff_density: float = 0.2,
                     predefined_sparse_coeffs: dict = None,
                     transform = None,
                     root: str = './fft_data',
                     data_file_suffix: str = None,
                     original_coeffs_file_suffix: str = None,
                     rescale: float = 0.0,
                     ):
            
            self.input_dim = input_dim
            self.spectrum_size = spectrum_size
            self.num_samples_train = num_samples_train
            self.num_samples_test = num_samples_test
            self.noise_std = noise_std
            self.coeff_density = float(np.clip(coeff_density, 0.0, 1.0))
            self.transform = transform
            self.root = root
            self.data_file = os.path.join(root, f"CustomWaveDataset_{input_dim}_{spectrum_size}_{num_samples_train}_{num_samples_test}_{noise_std}_{coeff_density}_{data_file_suffix}.npz")
            if predefined_sparse_coeffs is None:
                self.original_coeffs_file = os.path.join(root, f"original_fourier_coeff_{input_dim}_{spectrum_size}_{num_samples_train}_{num_samples_test}_{noise_std}_{coeff_density}_{original_coeffs_file_suffix}.pkl")
            else:
                self.original_coeffs_file = os.path.join(root, f"original_fourier_coeff_predefined_{input_dim}_{spectrum_size}_{num_samples_train}_{num_samples_test}_{noise_std}_{original_coeffs_file_suffix}.pkl")
            self.num_samples = num_samples_train + num_samples_test

            def _normalize_with_coeff_bound(coeffs, y_values=None, one_sided=False):
                # Global bound from coefficients via triangle inequality.
                if one_sided:
                    scale = np.abs(coeffs[0]) + 2.0 * np.sum(np.abs(coeffs[1:]))
                else:
                    scale = np.sum(np.abs(coeffs))
                if scale < 1e-12:
                    scale = 1.0
                coeffs = coeffs / scale
                if y_values is None:
                    return coeffs
                return coeffs, y_values / scale

            def _apply_1d_sparsity(an_coeffs, bn_coeffs, density):
                if len(an_coeffs) <= 1 or density >= 1.0:
                    return an_coeffs, bn_coeffs
                mask = np.random.rand(len(an_coeffs) - 1) < density
                if not np.any(mask):
                    mask[np.random.randint(0, len(mask))] = True
                an_coeffs[1:] *= mask
                bn_coeffs[1:] *= mask
                return an_coeffs, bn_coeffs

            def _apply_hermitian_sparsity(coeffs, freq_tuples, freq_to_idx, density):
                if density >= 1.0:
                    return coeffs
                handled = set()
                active_pairs = 0
                # Pre-compute neg indices to avoid tuple(-np.array(f)) overhead per iteration
                neg_indices = {i: freq_to_idx[tuple(-np.array(f))] for i, f in enumerate(freq_tuples)}
                for i in range(len(freq_tuples)):
                    if i in handled:
                        continue
                    neg_i = neg_indices[i]
                    keep = np.random.rand() < density
                    if not keep:
                        coeffs[i] = 0.0
                        coeffs[neg_i] = 0.0
                    else:
                        active_pairs += 1
                    handled.add(i)
                    handled.add(neg_i)

                if active_pairs == 0:
                    center_idx = len(freq_tuples) // 2
                    coeffs[center_idx] = 1.0
                return coeffs
            
            if os.path.exists(self.original_coeffs_file):
                
                if os.path.exists(self.data_file):
                    data = np.load(self.data_file)
                    self.x = data['x']
                    self.y = data['y']
                    print(f"Data loaded from {self.data_file}")
                    print(f"max: {self.y.max()}, min: {self.y.min()}")
                    
                else:
                    with open(self.original_coeffs_file, 'rb') as file:
                        c = pickle.load(file)
                    
                    # Generate and save data
                    x1 = np.random.rand(self.num_samples_train, self.input_dim) * 2 * np.pi - rescale + 1e-5

                    # Create homogeneous grid from 0 to 2π in each dimension
                    grid_points_per_dim = int(np.ceil(self.num_samples_test**(1/self.input_dim)))
                    homogeneous_generators = [np.linspace(0, 2*np.pi, grid_points_per_dim, endpoint=True) - rescale + 1e-5
                                              for _ in range(self.input_dim)]
                    x2_grid = cartesian_product_transpose_pp(homogeneous_generators)

                    # Take only the required number of test samples
                    x2 = x2_grid[:self.num_samples_test]
                    
                    # Concatenate random and homogeneous samples
                    x = np.concatenate([x1, x2], axis=0)
                    
                    if input_dim < 2:
                        full_spectrum_generators = [np.linspace(0, self.spectrum_size-1, self.spectrum_size) for i in range(self.input_dim)]
                        full_spectrum = cartesian_product_transpose_pp(full_spectrum_generators)
                        omega_dot_x = np.einsum('ij,kj->ik', full_spectrum, x)
                        
                        c_real = c.real
                        c_imag = c.imag
                        c_0 = c_real[0]
                        cos_coeffs = c_real * 2
                        cos_coeffs[0] = c_0
                        y = (cos_coeffs @ np.cos(omega_dot_x) + (c_imag * -2) @ np.sin(omega_dot_x)).reshape(self.num_samples, 1)
                        if noise_std > 0:
                            y += np.random.rand(self.num_samples, 1) * noise_std
                        
                        self.x = x
                        self.y = y
                    
                    else:
                        c = c.flatten()
                        full_spectrum_generators = [np.linspace(-self.spectrum_size+1, self.spectrum_size-1, 2*self.spectrum_size-1) for i in range(self.input_dim)]
                        full_spectrum = cartesian_product_transpose_pp(full_spectrum_generators)
                        
                        y = np.zeros(self.num_samples)
                        chunk_size = min(100, self.num_samples)
                        for chunk_start in range(0, self.num_samples, chunk_size):
                            chunk_end = min(chunk_start + chunk_size, self.num_samples)
                            x_chunk = x[chunk_start:chunk_end]
                            omega_dot_x_chunk = np.einsum('ij,kj->ik', full_spectrum, x_chunk)
                            y[chunk_start:chunk_end] = (c @ np.exp(1j*omega_dot_x_chunk)).real
                        
                        y = y.reshape(-1, 1)
                        if noise_std > 0:
                            y += np.random.randn(self.num_samples, 1) * noise_std
                        
                        self.x = x
                        self.y = y
                    
                    np.savez(self.data_file, x=x, y=y)
                    print(f"Data generated and saved to {self.data_file}")
                    print(f"max: {y.max()}, min: {y.min()}")
                    
                    
            else:
                # Generate and save data
                x1 = np.random.rand(self.num_samples_train, self.input_dim) * 2 * np.pi - rescale + 1e-5

                # Create homogeneous grid from 0 to 2π in each dimension
                grid_points_per_dim = int(np.ceil(self.num_samples_test**(1/self.input_dim)))
                homogeneous_generators = [np.linspace(0, 2*np.pi, grid_points_per_dim, endpoint=True) - rescale + 1e-5
                                          for _ in range(self.input_dim)]
                x2_grid = cartesian_product_transpose_pp(homogeneous_generators)

                # Take only the required number of test samples
                x2 = x2_grid[:self.num_samples_test]
                
                # Concatenate random and homogeneous samples
                x = np.concatenate([x1, x2], axis=0)
                
                if input_dim < 2:
                    full_spectrum_generators = [np.linspace(0, self.spectrum_size-1, self.spectrum_size) for i in range(self.input_dim)]
                    full_spectrum = cartesian_product_transpose_pp(full_spectrum_generators)
                    omega_dot_x = np.einsum('ij,kj->ik', full_spectrum, x)
                    
                    if predefined_sparse_coeffs is not None:
                        fourier_coeff_an = np.zeros(len(full_spectrum))
                        fourier_coeff_bn = np.zeros(len(full_spectrum))
                        for f, value in predefined_sparse_coeffs.items():
                            if f == 0:
                                fourier_coeff_an[0] = value.real
                            else:
                                fourier_coeff_an[f] = 2*value.real
                                fourier_coeff_bn[f] = -2*value.imag
                                
                        y = (fourier_coeff_bn @ np.sin(omega_dot_x) + fourier_coeff_an @ np.cos(omega_dot_x)).reshape(self.num_samples, 1)
                        if noise_std > 0:
                            y += np.random.rand(self.num_samples, 1) * noise_std
                        
                        c = np.zeros(len(full_spectrum), dtype=complex)
                        for f, v in predefined_sparse_coeffs.items():
                            c[f] = v
                                
                    else:
                        fourier_coeff_an = np.random.rand(len(full_spectrum))
                        fourier_coeff_bn = np.random.rand(len(full_spectrum))
                        fourier_coeff_an, fourier_coeff_bn = _apply_1d_sparsity(
                            fourier_coeff_an,
                            fourier_coeff_bn,
                            self.coeff_density,
                        )
                        
                        y = (fourier_coeff_bn @ np.sin(omega_dot_x) + fourier_coeff_an @ np.cos(omega_dot_x)).reshape(self.num_samples, 1)
                        if noise_std > 0:
                            y += np.random.rand(self.num_samples, 1) * noise_std
                        
                        c = np.concatenate((np.array([fourier_coeff_an[0]]), 0.5*(fourier_coeff_an[1:] - 1j*fourier_coeff_bn[1:])))

                        c_modulus_sum = np.sum(np.abs(c))
                        c[0] += 0.1*c_modulus_sum  # Adjust the DC component to ensure normalization
                        
                        y += 0.1*c_modulus_sum
                    
                    c, y = _normalize_with_coeff_bound(c, y, one_sided=True)
                    self.x = x
                    self.y = y
                
                else:
                    full_spectrum_generators = [np.linspace(-self.spectrum_size+1, self.spectrum_size-1, 2*self.spectrum_size-1) for i in range(self.input_dim)]
                    full_spectrum = cartesian_product_transpose_pp(full_spectrum_generators)

                    freq_tuples = [tuple(f) for f in full_spectrum]
                    freq_to_idx = {f: i for i, f in enumerate(freq_tuples)}
                    
                    # Generate random Fourier coefficients with Hermitian symmetry
                    c = np.zeros(len(full_spectrum), dtype=complex)
                    
                    if predefined_sparse_coeffs is not None:
                        for f, value in predefined_sparse_coeffs.items():
                            if f in freq_to_idx:
                                idx = freq_to_idx[f]
                                c[idx] = value
                                neg_f = tuple(-np.array(f))
                                if neg_f in freq_to_idx and neg_f != f:
                                    c[freq_to_idx[neg_f]] = np.conj(value)
                        
                        c_modulus_sum = np.sum(np.abs(c))
                        c = _normalize_with_coeff_bound(c)
                                    
                    else:
                        for i, f in enumerate(freq_tuples[:len(freq_tuples)//2 + 1]):
                            neg_f = tuple(-np.array(f))
                            real = np.random.randn()
                            imag = np.random.randn()
                            c[i] = real + 1j * imag
                            if neg_f in freq_to_idx and neg_f != f:
                                c[freq_to_idx[neg_f]] = real - 1j * imag
                        c[len(freq_tuples)//2] = np.random.randn()
                    
                        c = _apply_hermitian_sparsity(c, freq_tuples, freq_to_idx, self.coeff_density)
                        c_modulus_sum = np.sum(np.abs(c))
                        c[len(freq_tuples)//2] += 0.1*c_modulus_sum # Adjust the DC component to ensure normalization
                        c = _normalize_with_coeff_bound(c)
                    
                    # Generate y values in chunks to avoid memory issues
                    y = np.zeros(self.num_samples)
                    chunk_size = min(100, self.num_samples)
                    for chunk_start in range(0, self.num_samples, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, self.num_samples)
                        x_chunk = x[chunk_start:chunk_end]
                        omega_dot_x_chunk = np.einsum('ij,kj->ik', full_spectrum, x_chunk)
                        y[chunk_start:chunk_end] = (c @ np.exp(1j*omega_dot_x_chunk)).real
                    
                    y = y.reshape(-1, 1)
                    if noise_std > 0:
                        y += np.random.randn(self.num_samples, 1) * noise_std

                    self.x = x
                    self.y = y
                    # Reshape c to multi-dimensional array
                    c = c.reshape([spectrum_size*2-1]*input_dim)
                    
                with open(self.original_coeffs_file, 'wb') as file:
                    pickle.dump(c, file, protocol=pickle.HIGHEST_PROTOCOL)
                    
                np.savez(self.data_file, x=x, y=y)
                print(f"Data generated and saved to {self.data_file}")
                print(f"max: {self.y.max()}, min: {self.y.min()}")
            
        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            x = self.x[idx]
            y = self.y[idx]
            if self.transform:
                x = self.transform(x)
                y = self.transform(y)
            return x, y



class FloatFreqWaveDataset(Dataset):
        def __init__(self, input_dim: int = 2, 
                     spectrum_size: int = 5, 
                     num_samples_train: int = 10000, 
                     num_samples_test: int = 500,
                     noise_std: float = 0.1,
                     coeff_density: float = 0.2,
                     predefined_sparse_coeffs: dict = None,
                     freq_density: int = 1,
                     transform = None,
                     root: str = './fft_data',
                     data_file_suffix: str = None,
                     original_coeffs_file_suffix: str = None,
                     rescale: float = 0.0,
                     ):
            
            self.input_dim = input_dim
            self.spectrum_size = spectrum_size
            self.num_samples_train = num_samples_train
            self.num_samples_test = num_samples_test
            self.noise_std = noise_std
            self.coeff_density = float(np.clip(coeff_density, 0.0, 1.0))
            self.transform = transform
            self.root = root
            self.data_file = os.path.join(root, f"CustomWaveDataset_{input_dim}_{spectrum_size}_{freq_density}_{num_samples_train}_{num_samples_test}_{noise_std}_{coeff_density}_{data_file_suffix}.npz")
            if predefined_sparse_coeffs is None:
                self.original_coeffs_file = os.path.join(root, f"original_fourier_coeff_{input_dim}_{spectrum_size}_{freq_density}_{num_samples_train}_{num_samples_test}_{noise_std}_{coeff_density}_{original_coeffs_file_suffix}.pkl")
            else:
                self.original_coeffs_file = os.path.join(root, f"original_fourier_coeff_predefined_{input_dim}_{spectrum_size}_{freq_density}_{num_samples_train}_{num_samples_test}_{noise_std}_{original_coeffs_file_suffix}.pkl")
            self.num_samples = num_samples_train + num_samples_test

            def _normalize_with_coeff_bound(coeffs, y_values=None, one_sided=False):
                # Global bound from coefficients via triangle inequality.
                if one_sided:
                    scale = np.abs(coeffs[0]) + 2.0 * np.sum(np.abs(coeffs[1:]))
                else:
                    scale = np.sum(np.abs(coeffs))
                if scale < 1e-12:
                    scale = 1.0
                coeffs = coeffs / scale
                if y_values is None:
                    return coeffs
                return coeffs, y_values / scale

            def _apply_1d_sparsity(an_coeffs, bn_coeffs, density):
                if len(an_coeffs) <= 1 or density >= 1.0:
                    return an_coeffs, bn_coeffs
                mask = np.random.rand(len(an_coeffs) - 1) < density
                if not np.any(mask):
                    mask[np.random.randint(0, len(mask))] = True
                an_coeffs[1:] *= mask
                bn_coeffs[1:] *= mask
                return an_coeffs, bn_coeffs

            def _apply_hermitian_sparsity(coeffs, freq_tuples, freq_to_idx, density):
                if density >= 1.0:
                    return coeffs
                handled = set()
                active_pairs = 0
                # Pre-compute neg indices to avoid tuple(-np.array(f)) overhead per iteration
                neg_indices = {i: freq_to_idx[tuple(-np.array(f))] for i, f in enumerate(freq_tuples)}
                for i in range(len(freq_tuples)):
                    if i in handled:
                        continue
                    neg_i = neg_indices[i]
                    keep = np.random.rand() < density
                    if not keep:
                        coeffs[i] = 0.0
                        coeffs[neg_i] = 0.0
                    else:
                        active_pairs += 1
                    handled.add(i)
                    handled.add(neg_i)

                if active_pairs == 0:
                    center_idx = len(freq_tuples) // 2
                    coeffs[center_idx] = 1.0
                return coeffs
            
            if os.path.exists(self.original_coeffs_file):
                
                if os.path.exists(self.data_file):
                    data = np.load(self.data_file)
                    self.x = data['x']
                    self.y = data['y']
                    print(f"Data loaded from {self.data_file}")
                    print(f"max: {self.y.max()}, min: {self.y.min()}")
                    
                else:
                    with open(self.original_coeffs_file, 'rb') as file:
                        c = pickle.load(file)
                    
                    # Generate and save data
                    x1 = np.random.rand(self.num_samples_train, self.input_dim) * 2 * np.pi - rescale + 1e-5

                    # Create homogeneous grid from 0 to 2π in each dimension
                    grid_points_per_dim = int(np.ceil(self.num_samples_test**(1/self.input_dim)))
                    homogeneous_generators = [np.linspace(0, 2*np.pi, grid_points_per_dim, endpoint=True) - rescale + 1e-5
                                              for _ in range(self.input_dim)]
                    x2_grid = cartesian_product_transpose_pp(homogeneous_generators)

                    # Take only the required number of test samples
                    x2 = x2_grid[:self.num_samples_test]
                    
                    # Concatenate random and homogeneous samples
                    x = np.concatenate([x1, x2], axis=0)
                    
                    if input_dim < 2:
                        full_spectrum_generators = [np.linspace(0, self.spectrum_size-1, (self.spectrum_size-1)*freq_density+1) for i in range(self.input_dim)]
                        full_spectrum = cartesian_product_transpose_pp(full_spectrum_generators)
                        omega_dot_x = np.einsum('ij,kj->ik', full_spectrum, x)
                        
                        c_real = c.real
                        c_imag = c.imag
                        c_0 = c_real[0]
                        cos_coeffs = c_real * 2
                        cos_coeffs[0] = c_0
                        y = (cos_coeffs @ np.cos(omega_dot_x) + (c_imag * -2) @ np.sin(omega_dot_x)).reshape(self.num_samples, 1)
                        if noise_std > 0:
                            y += np.random.rand(self.num_samples, 1) * noise_std
                        
                        self.x = x
                        self.y = y
                    
                    else:
                        c = c.flatten()
                        full_spectrum_generators = [np.linspace(-self.spectrum_size+1, self.spectrum_size-1, 2*(self.spectrum_size-1)*freq_density+1) for i in range(self.input_dim)]
                        full_spectrum = cartesian_product_transpose_pp(full_spectrum_generators)
                        
                        y = np.zeros(self.num_samples)
                        chunk_size = min(100, self.num_samples)
                        for chunk_start in range(0, self.num_samples, chunk_size):
                            chunk_end = min(chunk_start + chunk_size, self.num_samples)
                            x_chunk = x[chunk_start:chunk_end]
                            omega_dot_x_chunk = np.einsum('ij,kj->ik', full_spectrum, x_chunk)
                            y[chunk_start:chunk_end] = (c @ np.exp(1j*omega_dot_x_chunk)).real
                        
                        y = y.reshape(-1, 1)
                        if noise_std > 0:
                            y += np.random.randn(self.num_samples, 1) * noise_std
                        
                        self.x = x
                        self.y = y
                    
                    np.savez(self.data_file, x=x, y=y)
                    print(f"Data generated and saved to {self.data_file}")
                    print(f"max: {y.max()}, min: {y.min()}")
                    
                    
            else:
                # Generate and save data
                x1 = np.random.rand(self.num_samples_train, self.input_dim) * 2 * np.pi - rescale + 1e-5

                # Create homogeneous grid from 0 to 2π in each dimension
                grid_points_per_dim = int(np.ceil(self.num_samples_test**(1/self.input_dim)))
                homogeneous_generators = [np.linspace(0, 2*np.pi, grid_points_per_dim, endpoint=True) - rescale + 1e-5
                                          for _ in range(self.input_dim)]
                x2_grid = cartesian_product_transpose_pp(homogeneous_generators)

                # Take only the required number of test samples
                x2 = x2_grid[:self.num_samples_test]
                
                # Concatenate random and homogeneous samples
                x = np.concatenate([x1, x2], axis=0)
                
                if input_dim < 2:
                    full_spectrum_generators = [np.linspace(0, self.spectrum_size-1, (self.spectrum_size-1)*freq_density+1) for i in range(self.input_dim)]
                    full_spectrum = cartesian_product_transpose_pp(full_spectrum_generators)
                    omega_dot_x = np.einsum('ij,kj->ik', full_spectrum, x)
                    
                    if predefined_sparse_coeffs is not None:
                        fourier_coeff_an = np.zeros(len(full_spectrum))
                        fourier_coeff_bn = np.zeros(len(full_spectrum))
                        for f, value in predefined_sparse_coeffs.items():
                            if f == 0:
                                fourier_coeff_an[0] = value.real
                            else:
                                fourier_coeff_an[int(np.round(f*freq_density))] = 2*value.real
                                fourier_coeff_bn[int(np.round(f*freq_density))] = -2*value.imag
                                
                        y = (fourier_coeff_bn @ np.sin(omega_dot_x) + fourier_coeff_an @ np.cos(omega_dot_x)).reshape(self.num_samples, 1)
                        if noise_std > 0:
                            y += np.random.rand(self.num_samples, 1) * noise_std
                        
                        c = np.zeros(len(full_spectrum), dtype=complex)
                        for f, v in predefined_sparse_coeffs.items():
                            c[int(np.round(f*freq_density))] = v
                                
                    else:
                        fourier_coeff_an = np.random.rand(len(full_spectrum))
                        fourier_coeff_bn = np.random.rand(len(full_spectrum))
                        fourier_coeff_an, fourier_coeff_bn = _apply_1d_sparsity(
                            fourier_coeff_an,
                            fourier_coeff_bn,
                            self.coeff_density,
                        )
                        
                        y = (fourier_coeff_bn @ np.sin(omega_dot_x) + fourier_coeff_an @ np.cos(omega_dot_x)).reshape(self.num_samples, 1)
                        if noise_std > 0:
                            y += np.random.rand(self.num_samples, 1) * noise_std
                        
                        c = np.concatenate((np.array([fourier_coeff_an[0]]), 0.5*(fourier_coeff_an[1:] - 1j*fourier_coeff_bn[1:])))

                        c_modulus_sum = np.sum(np.abs(c))
                        c[0] += 0.1*c_modulus_sum  # Adjust the DC component to ensure normalization
                        
                        y += 0.1*c_modulus_sum
                    
                    c, y = _normalize_with_coeff_bound(c, y, one_sided=True)
                    self.x = x
                    self.y = y
                
                else:
                    full_spectrum_generators = [np.linspace(-self.spectrum_size+1, self.spectrum_size-1, 2*(self.spectrum_size-1)*freq_density+1) for i in range(self.input_dim)]
                    full_spectrum = cartesian_product_transpose_pp(full_spectrum_generators)

                    freq_tuples = [tuple(np.round(f,5)) for f in full_spectrum]
                    freq_to_idx = {f: i for i, f in enumerate(freq_tuples)}
                    
                    # Generate random Fourier coefficients with Hermitian symmetry
                    c = np.zeros(len(full_spectrum), dtype=complex)
                    
                    if predefined_sparse_coeffs is not None:
                        for f, value in predefined_sparse_coeffs.items():
                            f_rounded = tuple(np.round(f,5))
                            if f in freq_to_idx:
                                idx = freq_to_idx[f_rounded]
                                c[idx] = value
                                neg_f = tuple(-np.array(f_rounded))
                                if neg_f in freq_to_idx and neg_f != f_rounded:
                                    c[freq_to_idx[neg_f]] = np.conj(value)
                        
                        c_modulus_sum = np.sum(np.abs(c))
                        c = _normalize_with_coeff_bound(c)
                                    
                    else:
                        for i, f in enumerate(freq_tuples[:len(freq_tuples)//2 + 1]):
                            neg_f = tuple(-np.array(f))
                            real = np.random.randn()
                            imag = np.random.randn()
                            c[i] = real + 1j * imag
                            if neg_f in freq_to_idx and neg_f != f:
                                c[freq_to_idx[neg_f]] = real - 1j * imag
                        c[len(freq_tuples)//2] = np.random.randn()
                    
                        c = _apply_hermitian_sparsity(c, freq_tuples, freq_to_idx, self.coeff_density)
                        c_modulus_sum = np.sum(np.abs(c))
                        c[len(freq_tuples)//2] += 0.1*c_modulus_sum # Adjust the DC component to ensure normalization
                        c = _normalize_with_coeff_bound(c)
                    
                    # Generate y values in chunks to avoid memory issues
                    y = np.zeros(self.num_samples)
                    chunk_size = min(100, self.num_samples)
                    for chunk_start in range(0, self.num_samples, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, self.num_samples)
                        x_chunk = x[chunk_start:chunk_end]
                        omega_dot_x_chunk = np.einsum('ij,kj->ik', full_spectrum, x_chunk)
                        y[chunk_start:chunk_end] = (c @ np.exp(1j*omega_dot_x_chunk)).real
                    
                    y = y.reshape(-1, 1)
                    if noise_std > 0:
                        y += np.random.randn(self.num_samples, 1) * noise_std

                    self.x = x
                    self.y = y
                    # Reshape c to multi-dimensional array
                    c = c.reshape([2*(self.spectrum_size-1)*freq_density+1]*input_dim)
                    
                with open(self.original_coeffs_file, 'wb') as file:
                    pickle.dump(c, file, protocol=pickle.HIGHEST_PROTOCOL)
                    
                np.savez(self.data_file, x=x, y=y)
                print(f"Data generated and saved to {self.data_file}")
                print(f"max: {self.y.max()}, min: {self.y.min()}")
            
        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            x = self.x[idx]
            y = self.y[idx]
            if self.transform:
                x = self.transform(x)
                y = self.transform(y)
            return x, y
        
        
        
class CustomFFTInputs(Dataset):
        def __init__(self, input_dim: int = 2, 
                     fft_density: int = 5,
                     sampling_overhead_multiplier: int = 1,
                     transform = None,
                     root: str = './fft_data',
                     input_data_file_suffix: str = None,
                     rescale: float = 0.0,
                     ):

            self.input_dim = input_dim
            self.fft_density = fft_density
            self.sampling_overhead_multiplier = sampling_overhead_multiplier
            self.transform = transform
            self.root = root
            self.data_file = os.path.join(root, f"CustomFFTInputs_{input_dim}_{fft_density}_{sampling_overhead_multiplier}_{input_data_file_suffix}.npz")
            if os.path.exists(self.data_file):
                data = np.load(self.data_file)
                self.x = data['arr_0'] if 'arr_0' in data else data['x']
                print(f"FFT input loaded from {self.data_file}")
            else:
                homogeneous_inputs_generators = [np.linspace(0, 2*np.pi, self.fft_density*self.sampling_overhead_multiplier, endpoint=True) - rescale + 1e-5 
                                                 for i in range(self.input_dim)]

                x = cartesian_product_transpose_pp(homogeneous_inputs_generators)

                self.x = x
                np.savez(self.data_file, x=x)
                print(f"FFT input data generated and saved to {self.data_file}")

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            x = self.x[idx]
            if self.transform:
                x = self.transform(x)
            return x
        

class ExtendedCustomFFTInputs(Dataset):
        def __init__(self, input_dim: int = 2, 
                     fft_density: int = 5,
                     sampling_precision_overhead_multiplier: int = 1,
                     frequency_density_overhead_multiplier: int = 1,
                     transform = None,
                     root: str = './fft_data',
                     input_data_file_suffix: str = None,
                     ):

            self.input_dim = input_dim
            self.fft_density = fft_density
            self.sampling_precision_overhead_multiplier = sampling_precision_overhead_multiplier
            self.frequency_density_overhead_multiplier = frequency_density_overhead_multiplier
            self.transform = transform
            self.root = root
            self.data_file = os.path.join(root, f"CustomFFTInputs_{input_dim}_{fft_density}_{sampling_precision_overhead_multiplier}_{frequency_density_overhead_multiplier}_{input_data_file_suffix}.npz")
            if os.path.exists(self.data_file):
                data = np.load(self.data_file)
                self.x = data['arr_0'] if 'arr_0' in data else data['x']
                print(f"FFT input loaded from {self.data_file}")
            else:
                homogeneous_inputs_generators = [np.linspace(0, 
                                                             2*np.pi*frequency_density_overhead_multiplier, 
                                                             fft_density*sampling_precision_overhead_multiplier*frequency_density_overhead_multiplier,
                                                             endpoint=True) + 1e-5 
                                                 for i in range(self.input_dim)]

                x = cartesian_product_transpose_pp(homogeneous_inputs_generators)

                self.x = x
                np.savez(self.data_file, x=x)
                print(f"FFT input data generated and saved to {self.data_file}")

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            x = self.x[idx]
            if self.transform:
                x = self.transform(x)
            return x
        


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CustomWaveDataset(input_dim=2, spectrum_size=4, num_samples=10000, noise_std=0.1)
    subset = Subset(dataset, indices=range(1000,2000))
    dataloader = DataLoader(subset, batch_size=5, shuffle=True)

    for batch_idx, (data, target) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print("Data:", data)
        print(data.view(data.size(0), -1).to(device))
        print("Target:", target)
        if batch_idx == 0:  # Just show first 3 batches
            break