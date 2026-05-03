# Import required libraries
import numpy as np
import torch
from qml_modules import *
import os
import random
from multiprocessing import Pool

def label_check(args):
    model, data, target = args
    data = data.view(data.size(0), -1)
    output = model(data)
    _, predicted = torch.max(output.data, 1)
    total = target.size(0)
    correct = (predicted == target).sum().item()
    
    return total, correct

#torch.autograd.set_detect_anomaly(True) 
# MPI-based parallel batch processing for CPU
def process_batch_cpu_mpi(model, data, target, batch_size, minibatch_size, criterion,rank, size):
    """
    Parallel batch processing using MPI for backpropagation within a batch.
    Each MPI rank processes a subset of the minibatches, accumulates gradients locally,
    and then gradients are averaged across all ranks.
    """

    total_loss = 0.0
    # Split minibatches among ranks
    indices = list(range(0, len(data), minibatch_size))
    current_rank_indicies = indices[rank::size]

    for i in current_rank_indicies:
        mini_data = data[i:i+minibatch_size]
        mini_target = target[i:i+minibatch_size]
        output = model(mini_data)

        loss = criterion(output, mini_target)
        loss = loss / batch_size * len(mini_data)
        loss.backward()
        total_loss += loss.item()

    return total_loss


def loss_eval(args):
    model, criterion, data, target, batch_size = args
    output = model(data)
    loss = criterion(output, target)
    loss = loss / batch_size * len(data)
    loss.backward()  # Accumulate gradients
    
    return loss.item()

def process_batch_cpu_multiprocessing(model, data, target, batch_size, batch_idx, minibatch_size, criterion):
    
    size = len(data)
    total_loss = 0
    nan_counts = 0
    
    num_threads = size // minibatch_size + 1
    
    mini_data_list = [data[i:i+minibatch_size] for i in range(0, size, minibatch_size)]
    mini_target_list = [target[i:i+minibatch_size] for i in range(0, size, minibatch_size)]
    data_target_pairs_list = list(zip(mini_data_list, mini_target_list))
    
    with Pool(num_threads) as p:
        for result in p.map(loss_eval, [(model, criterion, mini_data, mini_target, batch_size) for mini_data, mini_target in data_target_pairs_list]):
            #result = p.map(loss_eval, mini_data_list)
            #print(f'Batch {batch_idx}, processed {len(result)} samples', flush=True)
            total_loss += result
    
    total_loss *= batch_size/(size-nan_counts)
    
    return total_loss, nan_counts

def process_minibatch(model, data, target, batch_size, minibatch_size, criterion, backpropagate=True):
    
    # Process one sample at a time

    # Forward pass with gradient tracking
    output = model(data)
    
    loss = criterion(output, target)
    
    # Accumulate gradients (scaled by 1/batch_size)
    loss = loss / batch_size * len(data)  # Scale loss for gradient accumulation
    if backpropagate:
        loss.backward()  # Gradients accumulate across samples
    
    #print(optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy(), flush=True)
    #print(model.clamped_classical.weight.grad, flush=True)
    #for param in model.parameters():
    #    if param.shape == (3*layers, num_qubits):
    #        print(torch.mean(param.grad**2))
    
    return loss.item()

# Modified evaluation and training code with per-sample processing
def process_batch(model, data, target, batch_size, batch_idx, minibatch_size, criterion):
    
    size = len(data)
    total_loss = 0
    nan_counts = 0
    # Process one sample at a time
    for i in range(0, size, minibatch_size):
        mini_data = data[i:i+minibatch_size]  # Keep batch dimension (shape [1, 16])
        mini_target = target[i:i+minibatch_size]
        #print(mini_target, flush=True)
        # Forward pass with gradient tracking
        output = model(mini_data)
        
        if torch.any(torch.isnan(output)):
            nan_counts += 1
            print(f'{batch_idx * batch_size + i + minibatch_size} nan encountered', flush=True)
            #with open(f'noisy_QNN_test/nan_image_{batch_idx * batch_size + i + minibatch_size}.pkl', 'wb') as file:
            #    pickle.dump(mini_data, file, protocol=pickle.HIGHEST_PROTOCOL)
                        
            del mini_data
            del mini_target
            del output
            continue
        
        loss = criterion(output, mini_target)
        
        #print(f'output: {output}', flush=True)
        print(f'{batch_idx * batch_size + i + minibatch_size} done', flush=True)
        
        # Accumulate gradients (scaled by 1/batch_size)
        loss = loss / batch_size * len(mini_data)  # Scale loss for gradient accumulation
        loss.backward()  # Gradients accumulate across samples
        total_loss += loss.item()
        
        #print(optimizer.param_groups[0]['params'][0].grad.data.cpu().numpy(), flush=True)
        #print(model.clamped_classical.weight.grad, flush=True)
        #for param in model.parameters():
        #    if param.shape == (3*layers, num_qubits):
        #        print(torch.mean(param.grad**2))
    
    total_loss *= batch_size/(size-nan_counts)
    
    return total_loss, nan_counts



def load_checkpoint(model, optimizer, device, filename='noisy_QNN_test/checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_image_index = checkpoint['last_image_index']
        print("=> loaded checkpoint '{}'"
                  .format(filename))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        
        
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    return model.to(device), optimizer, last_image_index


def calc_amplitude_damping_params(decay_rate: float, relative_T: int|float, dt: float) -> tuple:
    """
    Calculate the amplitude damping parameters based on decay rate, temperature, and time step.
    
    Args:
        decay_rate (float): The decay rate of the amplitude damping at 0 K
        relative (float): The relative temperature = hw/2kT.
        dt (float): The time step.
    
    Returns:
        tuple: Parameters for amplitude damping.
    """
    p_damping = np.exp(relative_T)/(2*np.cosh(relative_T))
    
    Gamma = decay_rate/np.tanh(relative_T)
    gamma = 1 - np.exp(-Gamma * dt)
    
    return p_damping, gamma, Gamma * dt


def rfftn_to_full_fftn(rfft_output, N):
    """
    Convert N-dimensional rfftn output to full FFT for arbitrary dimensions.
    Uses Hermitian symmetry for real-valued functions: F[i0,i1,...,iN] = conj(F[-i0,-i1,...,-iN])
    
    Args:
        rfft_output: Complex array of shape (N, N, ..., N, N//2+1) with ndim dimensions
        N: Size of each dimension in the original input
    
    Returns:
        Full FFT array of shape (N, N, ..., N) with ndim dimensions
    
    Example:
        For 2D: rfftn_to_full_fftn(rfft_2d, 100) -> shape (100, 100)
        For 4D: rfftn_to_full_fftn(rfft_4d, 30) -> shape (30, 30, 30, 30)
        For 8D: rfftn_to_full_fftn(rfft_8d, 10) -> shape (10, 10, ..., 10)
    """
    ndim = rfft_output.ndim
    n_rfft = rfft_output.shape[-1]
    
    # Create full FFT array with all dimensions having size N
    full_shape = (N,) * ndim
    full_fft = np.zeros(full_shape, dtype=np.complex128)
    
    # Copy the rfft part (fill first n_rfft elements in last dimension)
    slices = [slice(None)] * (ndim - 1) + [slice(0, n_rfft)]
    full_fft[tuple(slices)] = rfft_output
    
    # Fill in conjugate symmetric part for the last dimension
    # F[i0, i1, ..., i(n-2), N-j] = conj(F[-i0, -i1, ..., -i(n-2), j]) for j > 0
    for j in range(1, n_rfft - 1):  # Don't duplicate j=0 or j=N//2
        # Generate all possible indices for dimensions 0 to ndim-2
        for indices in itertools.product(range(N), repeat=ndim-1):
            # Create conjugate index by negating all indices (with wraparound)
            conj_indices = tuple((-idx) % N for idx in indices)
            
            # Set the conjugate symmetric value
            full_indices = indices + (N - j,)
            rfft_indices = conj_indices + (j,)
            full_fft[full_indices] = np.conjugate(rfft_output[rfft_indices])
    
    return full_fft


def custom_nd_dtft_rfftn(outputs_sorted, N, f_max):
    """
    Compute the N-dimensional discrete-time Fourier transform (DTFT) at uniformly spaced 
    frequencies from -f_max to f_max in each dimension (except the last dimension which has N//2 + 1 elements,
    starting from 0 to f_max). Replicates the shape and layout of torch.fft.rfftn.
    """

    D = outputs_sorted.ndim
    M = outputs_sorted.shape[0]
    device = outputs_sorted.device
    dtype = torch.complex128 if outputs_sorted.dtype == torch.float64 else torch.complex64
    
    j_full = torch.arange(N, device=device, dtype=torch.float64)
    # The frequencies match rfftn structure: 0, 1, ..., N//2, -N//2+1, ..., -1 scaled
    omega_max = 2.0 * np.pi * f_max
    
    # By dividing by N//2 instead of N/2.0, we guarantee the maximum positive index 
    # exactly hits omega_max, for both odd and even N.
    step_size = omega_max / (N // 2)
    
    freqs_full = torch.where(j_full <= N//2, j_full, j_full - N) * step_size
    
    j_half = torch.arange(N//2 + 1, device=device, dtype=torch.float64)
    freqs_half = j_half * step_size
    
    n = torch.arange(M, device=device, dtype=torch.float64)
    x_coords = n * (2.0 * np.pi / M)
    
    W_full = torch.exp(-1j * freqs_full.unsqueeze(1) * x_coords.unsqueeze(0)).to(dtype)
    W_half = torch.exp(-1j * freqs_half.unsqueeze(1) * x_coords.unsqueeze(0)).to(dtype)
    
    out = outputs_sorted.to(dtype)
    for d in range(D):
        W = W_half if d == D - 1 else W_full
        out = torch.tensordot(out, W, dims=([0], [1]))
        
    return out

def generate_random_tuples(num_tuples, dim, min_val, max_val):
    unordered= [tuple([random.randint(min_val, max_val) for _ in range(dim)]) 
                for _ in range(num_tuples)]
    print(unordered)
    ordered = sorted(unordered, key=lambda x: (x[0], x[1], x[2], x[3]))
    return ordered

class TreeNode:
    def __init__(self):
        self.data = None
        self.coefficient = None
        self.leftFactor = None
        self.rightFactor = None
        self.SinCosFactor = None
        self.leftChild = None
        self.rightChild = None
        
def binary_graph_expansion(O, pauli_generators):
    if isinstance(O, str):
        num_qubits = len(O)
        O_x = 0
        O_z = 0
        for i in range(len(O)):
            if O[i] == 'I':
                O_x += 0
                O_z += 0
            elif O[i] == 'X':
                O_x += 2**(num_qubits-i-1)
                O_z += 0
            elif O[i] == 'Z':
                O_x += 0
                O_z += 2**(num_qubits-i-1)
            elif O[i] == 'Y':
                O_x += 2**(num_qubits-i-1)
                O_z += 2**(num_qubits-i-1)
            else:
                raise Exception(f'{O} contains {O[i]} which is an invalid Pauli string')
        O_num = (O_x, O_z)
    
    pauli_generators_num = []
    for pauli in pauli_generators:
        if isinstance(pauli, str):
            num_qubits = len(pauli)
            pauli_x = 0
            pauli_z = 0
            for i in range(len(pauli)):
                if pauli[i] == 'I':
                    pauli_x += 0
                    pauli_z += 0
                elif pauli[i] == 'X':
                    pauli_x += 2**(num_qubits-i-1)
                    pauli_z += 0
                elif pauli[i] == 'Z':
                    pauli_x += 0
                    pauli_z += 2**(num_qubits-i-1)
                elif pauli[i] == 'Y':
                    pauli_x += 2**(num_qubits-i-1)
                    pauli_z += 2**(num_qubits-i-1)
                else:
                    raise Exception(f'{pauli} contains {pauli[i]} which is an invalid Pauli string')
            pauli_num = (pauli_x, pauli_z)
        else:
            pauli_num = pauli
        
        pauli_generators_num.append(pauli_num)
    

    root = TreeNode()
    root.data = (pauli_generators_num, O_num)
    root.coefficient = 1
    root.CosSinFactor = ['', '']  # [num of cos, num of sin]
    nodes_to_process = [(pauli_generators_num, O_num),]
    tree = [root]
    num_nodes_processed = 0
    while nodes_to_process:
        new_nodes_to_process = []
        num_nodes_to_process = len(nodes_to_process)
        for node in tree[num_nodes_processed : num_nodes_processed+num_nodes_to_process]:
            #current_node = TreeNode()
            if len(node.data[0]) > 1:
                new_nodes_to_process.append((node.data[0][:-1], node.data[1]))
            
            node.leftChild = TreeNode()
            node.leftChild.data = (node.data[0][:-1], node.data[1])
            node.leftChild.coefficient = node.coefficient
            tree.append(node.leftChild)
            
            if not is_commuting(node.data[0][-1], node.data[1]):
                if len(node.data[0]) > 1:
                    new_nodes_to_process.append((node.data[0][:-1], node.data[0][-1]^node.data[1]))
                
                node.rightChild = TreeNode()
                node.rightChild.data = (node.data[0][:-1], node.data[0][-1]^node.data[1])
                node.rightChild.coefficient = node.coefficient * 1j
                node.rightChild.CosSinFactor = [node.CosSinFactor[0]+'0', node.CosSinFactor[1]+'1']
                tree.append(node.rightChild)
                
                node.leftChild.CosSinFactor = [node.CosSinFactor[0]+'1', node.CosSinFactor[1]+'0']
                
            else:
                node.leftChild.CosSinFactor = [node.CosSinFactor[0]+'0', node.CosSinFactor[1]+'0']
            
            num_nodes_processed += 1
            
        nodes_to_process += new_nodes_to_process
        nodes_to_process = nodes_to_process[num_nodes_to_process : ]
    
    tree_leaves = [(node.data, node.SinCosFactor, node.coefficient) 
                   for node in tree if node.leftChild is None and node.rightChild is None]
    
    
    return tree_leaves


def binary_graph_expansion_raw(O, pauli_generators):
    if isinstance(O, str):
        num_qubits = len(O)
        O_x = 0
        O_z = 0
        for i in range(len(O)):
            if O[i] == 'I':
                O_x += 0
                O_z += 0
            elif O[i] == 'X':
                O_x += 2**(num_qubits-i-1)
                O_z += 0
            elif O[i] == 'Z':
                O_x += 0
                O_z += 2**(num_qubits-i-1)
            elif O[i] == 'Y':
                O_x += 2**(num_qubits-i-1)
                O_z += 2**(num_qubits-i-1)
            else:
                raise Exception(f'{O} contains {O[i]} which is an invalid Pauli string')
        O_num = (O_x, O_z)
    
    pauli_generators_num = []
    for pauli in pauli_generators:
        if isinstance(pauli, str):
            num_qubits = len(pauli)
            pauli_x = 0
            pauli_z = 0
            for i in range(len(pauli)):
                if pauli[i] == 'I':
                    pauli_x += 0
                    pauli_z += 0
                elif pauli[i] == 'X':
                    pauli_x += 2**(num_qubits-i-1)
                    pauli_z += 0
                elif pauli[i] == 'Z':
                    pauli_x += 0
                    pauli_z += 2**(num_qubits-i-1)
                elif pauli[i] == 'Y':
                    pauli_x += 2**(num_qubits-i-1)
                    pauli_z += 2**(num_qubits-i-1)
                else:
                    raise Exception(f'{pauli} contains {pauli[i]} which is an invalid Pauli string')
            pauli_num = (pauli_x, pauli_z)
        else:
            pauli_num = pauli
        
        pauli_generators_num.append(pauli_num)
    

    #root = TreeNode()
    #root.data = (pauli_generators_num, O_num)
    #root.coefficient = 1
    #root.CosSinFactor = ['', '']  # [num of cos, num of sin]
    nodes_to_process = [(pauli_generators_num, O_num),]
    tree = [[pauli_generators_num, O_num, 1, ['', '']],]
    num_nodes_processed = 0
    while nodes_to_process:
        new_nodes_to_process = []
        num_nodes_to_process = len(nodes_to_process)
        for node in tree[num_nodes_processed : num_nodes_processed+num_nodes_to_process]:
            #current_node = TreeNode()
            if len(node[0]) > 1:
                new_nodes_to_process.append((node[0][:-1], node[1]))
            
            #node.leftChild = TreeNode()
            #node.leftChild.data = (node.data[0][:-1], node.data[1])
            #node.leftChild.coefficient = node.coefficient
            
            branch_left = [node[0][:-1], node[1], node[2], node[3]]
            tree.append(branch_left)
            
            if not is_commuting(node[0][-1], node[1]):
                if len(node[0]) > 1:
                    new_nodes_to_process.append((node[0][:-1], node[0][-1]^node[1]))
                
                #node.leftChild.CosSinFactor = [node.CosSinFactor[0]+'1', node.CosSinFactor[1]+'0']
                tree[-1][3] = [node.CosSinFactor[0]+'1', node.CosSinFactor[1]+'0']
                
                #node.rightChild = TreeNode()
                #node.rightChild.data = (node[0][:-1], node[0][-1]^node[1])
                #node.rightChild.coefficient = node.coefficient * 1j
                #node.rightChild.CosSinFactor = [node.CosSinFactor[0]+'0', node.CosSinFactor[1]+'1']
                branch_right = [node[0][:-1], node[0][-1]^node[1], node[2]*1j, [node[3][0]+'0', node[3][1]+'1']]
                tree.append(branch_right)
                
            else:
                tree[-1][3] = [node.CosSinFactor[0]+'0', node.CosSinFactor[1]+'0']
            
            num_nodes_processed += 1
            
        nodes_to_process += new_nodes_to_process
        nodes_to_process = nodes_to_process[num_nodes_to_process : ]
    
    tree_leaves = [node for node in tree if len(node[0]) <= 1]
    
    
    return tree_leaves


def Pauli_product(A, B):
    X_A, Z_A = A
    X_B, Z_B = B
    
    X_C = X_A^X_B
    Z_C = Z_A^Z_B
    
    induced_coefficient = (1j)**( hamming_weight(X_A&(~Z_A)&(~X_B)&Z_B)#XZ
                         +hamming_weight(X_A&Z_A&X_B&(~Z_B))#YX
                         +hamming_weight((~X_A)&Z_A&X_B&Z_B)#ZY
                         -hamming_weight((~X_A)&Z_A&X_B&(~Z_B))#ZX
                         -hamming_weight(X_A&(~Z_A)&X_B&Z_B)#XY
                         -hamming_weight(X_A&Z_A&(~X_B)&Z_B))#YZ
    
    return X_C, Z_C, induced_coefficient

def is_commuting(A, B):
    X_A, Z_A = A
    X_B, Z_B = B
    
    X_AB, Z_AB, coeff_AB = Pauli_product(A,B)
    X_BA, Z_BA, coeff_BA = Pauli_product(B,A)
    
    if X_AB == X_BA and Z_AB == Z_BA and coeff_AB == coeff_BA:
        return True
    else:
        return False

# ref: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetTable
POPCOUNT_TABLE16 = [0, 1] * 2**15
for index in range(2, len(POPCOUNT_TABLE16)):  # 0 and 1 are trivial
    POPCOUNT_TABLE16[index] += POPCOUNT_TABLE16[index >> 1]

def hamming_weight(n):
    """return the Hamming weight of an integer (check how many '1's for an integer after converted to binary)

    Args:
        n (int): any integer

    Returns:
        int: number of ones of the integer n after converted to binary
    """
    c = 0
    while n:
        c += POPCOUNT_TABLE16[n & 0xffff]
        n >>= 16
    return c