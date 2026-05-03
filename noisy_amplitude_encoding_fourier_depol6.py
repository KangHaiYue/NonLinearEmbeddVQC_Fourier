# Import required libraries
import numpy as np
import torch
import torch.multiprocessing as mp
from jobs import fourier_analysis_single_amplitude_encoding_VQC

#from multiprocessing import Pool
if __name__ == "__main__":
    ######
    
    size = 2
    epoch_size = 10000
    train_size = 5000
    batch_size = 100
    minibatch_size = 50
    test_size = 3000
    
    num_qubits = 2
    #input_dim = 1
    layers = 10
    
    p_depolarising_idx = 6
    p_damping_idx = -1
    spectrum_size = 3
    fft_density = (spectrum_size-1)*2 + 1
    noise_std = 0
    coeff_sparsity = 0.02
    predefined_sparse_coeffs = {(0,0,0,0): 1, (1,0,0,0): 1, (1,1,0,0): 1, (1,1,1,0): 1, (1,1,1,1): 1,
                                (2,0,0,0): 1,(2,1,0,0): 1, (2,1,1,0): 1, (2,1,1,1): 1, 
                                (2,2,0,0): 1, (2,2,1,0): 1, (2,2,1,1): 1,
                                (2,2,2,0): 1, (2,2,2,1): 1, 
                                (2,2,2,2): 1}
    learning_rate = 0.01
    
    data_file_suffix = 'restart6_positive_x_with_endpoint'
    original_coeffs_file_suffix = 'restart6_positive_x_with_endpoint'
    input_data_file_suffix = 'restart6_positive_x_with_endpoint'
    
    dataset_choice = 'sparse_coeffs'  # Options: 'normal', 'amplified_c0', 'sparse_coeffs'
    rescale = 0
    
    sampling_overhead_multiplier = 3
    
    mp.spawn(
        fourier_analysis_single_amplitude_encoding_VQC, 
        args=(size, 
              epoch_size,
              train_size,
              batch_size,
              minibatch_size,
              test_size,
              num_qubits,
              #input_dim,
              layers,
              spectrum_size,
              fft_density,
              noise_std,
              coeff_sparsity,
              predefined_sparse_coeffs,
              learning_rate,
              data_file_suffix,
              original_coeffs_file_suffix,
              input_data_file_suffix,
              dataset_choice,
              rescale,
              sampling_overhead_multiplier,
              p_depolarising_idx,
              p_damping_idx,), 
        nprocs=size, 
        join=True
        )