# Import required libraries
import numpy as np
import torch
import torch.multiprocessing as mp
from jobs import fourier_analysis_single_amplitude_encoding_VQC_custum_frequencies

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
    
    p_depolarising_idx = -1
    p_damping_idx = -1
    spectrum_size = 3
    fft_density = (spectrum_size-1)*2 + 1
    noise_std = 0
    coeff_sparsity = 0.02
    predefined_sparse_coeffs = {(0.0, 0.5, 0.75, 0.0): -0.1130176626178776 + 0.508127758198351j, 
                                (0.5, 0.25, 0.0, 2.0): 0.49591022189066347 - 0.38454072013352536j, 
                                (0.5, 0.75, 0.75, 0.25): -0.6152395773679593 - 0.8885973049794j, 
                                (0.5, 1.5, 0.5, 0.0): -0.3206785006303474 - 0.029818341075093624j, 
                                (0.5, 1.75, 1.75, 0.5): 0.28685967059595696 + 0.25674098628597264j, 
                                (1.25, 0.5, 1.0, 1.0): 0.16040224255953306 - 0.8880035067765346j, 
                                (1.25, 2.0, 2.0, 1.0): -0.3781762067265708 + 0.36493117566449573j, 
                                (1.5, 1.0, 0.75, 0.0): -0.9932598550965481 - 0.3473215704134345j, 
                                (1.5, 1.25, 0.5, 1.0): -0.9942027156912354 - 0.06027596270866886j, 
                                (1.5, 1.5, 1.5, 1.5): 0.9059582138923219 - 0.5906894305025228j}
    freq_density = 4
    learning_rate = 0.01
    
    data_file_suffix = 'non_int_freqs_positive_x_with_endpoint'
    original_coeffs_file_suffix = 'non_int_freqs_positive_x_with_endpoint'
    input_data_file_suffix = 'non_int_freqs_positive_x_with_endpoint'
    
    dataset_choice = 'sparse_coeffs'  # Options: 'normal', 'amplified_c0', 'sparse_coeffs'
    rescale = 0
    
    sampling_overhead_multiplier = 3
    dtft_N = (2+2)*freq_density + 1
    dtft_f_max = 2/(2*np.pi)
    
    mp.spawn(
        fourier_analysis_single_amplitude_encoding_VQC_custum_frequencies, 
        args=(size,
              epoch_size,
              train_size,
              batch_size,
              minibatch_size,
              test_size,
              dtft_N,
              dtft_f_max,
              num_qubits,
              #input_dim,
              layers,
              spectrum_size,
              fft_density,
              noise_std,
              coeff_sparsity,
              predefined_sparse_coeffs,
              freq_density,
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