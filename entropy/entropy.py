import numpy as np
from collections import Counter
import math
# Define the EntropyCalculator class
# which includes methods to calculate entropy of spike trains
# these entropy calculation include:
# 1. Shannon entropy
# 2. Entropy rate
# 3. Conditional entropy
# 4. Mutual information
# 5. Cross-entropy

# These methods take a binary spike train as input, usually a matrix where each row represents a neuron and each column represents a time step.

class EntropyCalculator:
    def __init__(self):
        pass
    
    def joint_probabilities(self, X, Y):
        """
        Compute joint probabilities of two binary spike trains.

        Parameters:
        - X (np.ndarray): 1D binary spike train
        - Y (np.ndarray): 1D binary spike train

        Returns:
        - dict: Joint probabilities for each combination of (X, Y)
        """
        if len(X) != len(Y):
            raise ValueError("Spike trains must be the same length")

        joint = list(zip(X, Y))
        n = len(joint)
        
        joint_counts = Counter(joint)
        
        return {k: v / n for k, v in joint_counts.items()}
    
    # 1. Shannon entropy
    def shannon_entropy(self, spike_train, output='mean'):
        """
        Compute Shannon entropy of a binary spike train.

        Parameters:
        - spike_train (np.ndarray): 1D or 2D array (binary 0/1)
        - output (str): 'none' for arrays of entropies for each neuron, 'mean' for mean entropy across neurons, 'sum' for sum of entropies across neurons.

        Returns:
        - float or np.ndarray: Entropy
        """
        def compute_entropy_1d(spike_train):
            n = len(spike_train)
            counts = Counter(spike_train)
            probabilities = np.array([count / n for count in counts.values()])
            entropy = - np.sum( p * np.log2(p) for p in probabilities if p > 0)
            return entropy
            
        spike_train = np.asarray(spike_train)

        if spike_train.ndim == 1:
            return compute_entropy_1d(spike_train)
        elif spike_train.ndim == 2:
            entropies = []
            for neuron_train in spike_train:
                entropies.append(compute_entropy_1d(neuron_train))
            entropies = np.array(entropies)
            
            if output == 'none':
                return entropies
            elif output == 'mean':
                return np.mean(entropies)
            elif output == 'sum':
                return np.sum(entropies)
            else:
                raise ValueError("Output must be 'none', 'mean', or 'sum'")
        else:
            raise ValueError("Input must be a 1D or 2D binary array")
        
    # 2. Entropy rate
    def entropy_rate(self, spike_train, output='mean', window_size=3):
        """
        Compute entropy rate for each neuron in a binary spike matrix.

        Parameters:
        - spike_train (np.ndarray): 1D or 2D array (binary 0/1)
        - window_size (int): sliding window size k
        - reduction (str): 'none' → return vector per neuron,
                        'mean' → return mean across neurons,
                        'sum' → total entropy

        Returns:
        - np.ndarray or float: entropy rate per neuron or reduced value
        """
        def compute_entropy_rate_1d(spike_train, k):
            
            if isinstance(spike_train, np.ndarray):
                spike_train = spike_train.tolist()
            
            if isinstance(spike_train, list):
                spike_train = ''.join(str(b) for b in spike_train)
            
            n = len(spike_train)
            if n < k:
                raise ValueError("Sequence length must be at least k")
            
            blocks = [spike_train[i:i+k] for i in range(n - k + 1)]
            block_counts = Counter(blocks)
            total_blocks = len(blocks)
            probs = np.array([count / total_blocks for count in block_counts.values()])
            block_entropy = -np.sum(p * np.log2(p) for p in probs if p > 0)
            
            entropy_rate = block_entropy / k
            return entropy_rate

        spike_train = np.asarray(spike_train)
        
        if spike_train.ndim == 1:
            return compute_entropy_rate_1d(spike_train, window_size)
        elif spike_train.ndim == 2:
            entropy_list = [
                compute_entropy_rate_1d(neuron_train, window_size)
                for neuron_train in spike_train
            ]
            
            entropy_array = np.array(entropy_list)
            if output == 'none':
                return entropy_array
            elif output == 'mean':
                return np.mean(entropy_array)
            elif output == 'sum':
                return np.sum(entropy_array)
            else:
                raise ValueError("Invalid reduction type. Choose 'none', 'mean', or 'sum'.")
        else:
            raise ValueError("Input must be a 1D or 2D binary array")
    
    # 3. Conditional entropy
    def conditional_entropy(self, X, Y, output='none'):
        """
        Compute conditional entropy H(X|Y) between two binary spike trains X and Y.

        Parameters:
        - X, Y (np.ndarray): 1D or 2D binary spike train
        - output (str): 'none' for array of entropies per neuron,
                        'mean' for mean entropy across neurons,
                        'sum' for sum of entropies across neurons.

        Returns:
        - float or np.ndarray: entropy per neuron (1D) or reduced value
        """
        def compute_cond_entropy_1d(x, y):
            joint_probs = self.joint_probabilities(x, y)
            py = Counter(y)
            n = len(y)
            
            py = {k: v / n for k, v in py.items()}
            
            h = 0.0
            for (x_val, y_val), p_xy in joint_probs.items():
                if py[y_val] > 0:
                    p_x_given_y = p_xy / py[y_val]
                    h += p_xy * np.log2(p_x_given_y)
            return -h

        X = np.array(X)
        Y = np.array(Y)
        
        if X.shape != Y.shape:
            raise ValueError("X and Y must have the same shape")
        
        if X.ndim == 1 and Y.ndim == 1:
            return compute_cond_entropy_1d(X, Y)
        elif X.ndim == 2 and Y.ndim == 2:
            hxy_list = []
            for i in range(X.shape[0]):
                hxy_list.append(compute_cond_entropy_1d(X[i], Y[i]))
            hxy_list = np.array(hxy_list)
            
            if output == 'none':
                return hxy_list
            elif output == 'mean':
                return np.mean(hxy_list)
            elif output == 'sum':
                return np.sum(hxy_list)
            else:
                raise ValueError("Output must be 'none', 'mean', or 'sum'")
        else:
            raise ValueError("Input must be 1D or 2D binary arrays")
        
    # 4. Mutual information
    def mutual_information(self, X, Y, output='none'):
        """
        Compute mutual information I(X; Y) between two binary spike trains.

        Parameters:
        - X, Y (np.ndarray): 1D or 2D binary spike train
        - output (str): 'none' for array of entropies per neuron,
                        'mean' for mean entropy across neurons,
                        'sum' for sum of entropies across neurons.

        Returns:
        - float or np.ndarray: entropy per neuron (1D) or reduced value
        """
        
        def compute_mutual_info_1d(x, y):
            joint_probs = self.joint_probabilities(x, y)
            p_x = Counter(x)
            p_y = Counter(y)
            n = len(x)
            
            p_x = {k: v / n for k, v in p_x.items()}
            p_y = {k: v / n for k, v in p_y.items()}

            mi = 0.0
            for (x_val, y_val), p_xy in joint_probs.items():
                if p_x[x_val] > 0 and p_y[y_val] > 0:
                    mi += p_xy * np.log2(p_xy / (p_x[x_val] * p_y[y_val]))
            return mi
        
        X = np.array(X)
        Y = np.array(Y)
        
        if X.shape != Y.shape:
            raise ValueError("X and Y must have the same shape")
        
        if X.ndim == 1 and Y.ndim == 1:
            return compute_mutual_info_1d(X, Y)
        elif X.ndim == 2 and Y.ndim == 2:
            mi_list = []
            for i in range(X.shape[0]):
                mi_list.append(compute_mutual_info_1d(X[i], Y[i]))
            mi_list = np.array(mi_list)
            
            if output == 'none':
                return mi_list
            elif output == 'mean':
                return np.mean(mi_list)
            elif output == 'sum':
                return np.sum(mi_list)
            else:
                raise ValueError("Output must be 'none', 'mean', or 'sum'")
        else:
            raise ValueError("Input must be 1D or 2D binary arrays")
