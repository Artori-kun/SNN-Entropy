import numpy as np

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
    
    # 1. Shannon entropy
    def shannon_entropy(self, spike_train, output):
        """
        Compute Shannon entropy of a binary spike train.

        Parameters:
        - spike_train (np.ndarray): 2D array (binary 0/1)
        - output (str): 'none' for arrays of entropies for each neuron, 'mean' for mean entropy across neurons, 'sum' for sum of entropies across neurons.

        Returns:
        - float or np.ndarray: Entropy
        """
        def compute_entropy(p):
            if p == 0 or p == 1:
                return 0.0
            return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

        spike_train = np.asarray(spike_train)

        if spike_train.ndim != 2:
            raise ValueError("Input must be a 2D binary array")
        else:
            entropies = []
            for neuron_train in spike_train:
                p = np.mean(neuron_train)
                entropies.append(compute_entropy(p))
            entropies = np.array(entropies)
            
        if output == 'none':
            return entropies
        elif output == 'mean':
            return np.mean(entropies)
        elif output == 'sum':
            return np.sum(entropies)
        else:
            raise ValueError("Output must be 'none', 'mean', or 'sum'")
        
    # 2. Entropy rate
    def entropy_rate_matrix(self, spike_train, output, window_size=3):
        """
        Compute entropy rate for each neuron in a binary spike matrix.

        Parameters:
        - spike_train (np.ndarray): shape (n_neurons, n_timesteps)
        - window_size (int): sliding window size k
        - reduction (str): 'none' → return vector per neuron,
                        'mean' → return mean across neurons,
                        'sum' → total entropy

        Returns:
        - np.ndarray or float: entropy rate per neuron or reduced value
        """
        def compute_entropy_rate_single(seq, k):
            if len(seq) < k:
                return 0.0
            patterns = [tuple(seq[i:i + k]) for i in range(len(seq) - k + 1)]
            unique, counts = np.unique(patterns, axis=0, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log2(probs))
            return entropy / k

        spike_train = np.asarray(spike_train)
        if spike_train.ndim != 2:
            raise ValueError("Input must be a 2D binary array")

        entropy_list = [
            compute_entropy_rate_single(spike_train[i], window_size)
            for i in range(spike_train.shape[0])
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
        
    
    # 3. Conditional entropy
    # measure the uncertainty of the spike train in time t given the spike train in time t-1
    # does not measure the uncertainty between 2 spike trains
    def conditional_entropy(self, spike_train, output):
        """
        Compute conditional entropy H(X_t | X_{t-1}) for a binary spike train.

        Parameters:
        - spike_train (np.ndarray): 2D binary spike train
        - output (str): 'none' for array of entropies per neuron,
                        'mean' for mean entropy across neurons,
                        'sum' for sum of entropies across neurons.

        Returns:
        - float or np.ndarray: entropy per neuron (1D) or reduced value
        """
        def compute_cond_entropy(seq):
            if len(seq) < 2:
                return 0.0
            
            # Count joint occurrences of (x_{t-1}, x_t)
            joint_counts = {
                (0, 0): 0, (0, 1): 0,
                (1, 0): 0, (1, 1): 0
            }
            for t in range(1, len(seq)):
                pair = (seq[t-1], seq[t])
                joint_counts[pair] += 1

            total = sum(joint_counts.values())
            if total == 0:
                return 0.0

            joint_probs = {k: v / total for k, v in joint_counts.items()}
            cond_entropy = 0.0

            for (x_prev, x_curr), p_joint in joint_probs.items():
                p_prev = sum(joint_probs[(x_prev, b)] for b in [0, 1])
                if p_joint > 0 and p_prev > 0:
                    cond_entropy += -p_joint * np.log2(p_joint / p_prev)
            return cond_entropy

        spike_train = np.asarray(spike_train)

        if spike_train.ndim == 2:
            entropies = np.array([compute_cond_entropy(neuron) for neuron in spike_train])
        else:
            raise ValueError("Input must be 2D binary array")
        
        if output == 'none':
            return entropies
        elif output == 'mean':
            return np.mean(entropies)
        elif output == 'sum':
            return np.sum(entropies)
        else:
            raise ValueError("Output must be 'none', 'mean', or 'sum'")

    # 4. Mutual information
    def mutual_information(self, spike_train_x, spike_train_y):
        """
        Compute mutual information I(X; Y) between two binary spike trains.

        Parameters:
        - spike_train_x (np.ndarray): 1D binary spike train
        - spike_train_y (np.ndarray): 1D binary spike train

        Returns:
        - float: Mutual information in bits
        """
        if len(spike_train_x) != len(spike_train_y):
            raise ValueError("Spike trains must be the same length")

        # Count joint occurrences
        joint_counts = {
            (0, 0): 0, (0, 1): 0,
            (1, 0): 0, (1, 1): 0
        }
        for x, y in zip(spike_train_x, spike_train_y):
            joint_counts[(x, y)] += 1

        total = sum(joint_counts.values())
        if total == 0:
            return 0.0

        joint_probs = {k: v / total for k, v in joint_counts.items()}
        p_x = {
            0: sum(joint_probs[(0, b)] for b in [0, 1]),
            1: sum(joint_probs[(1, b)] for b in [0, 1])
        }
        p_y = {
            0: sum(joint_probs[(a, 0)] for a in [0, 1]),
            1: sum(joint_probs[(a, 1)] for a in [0, 1])
        }

        mi = 0.0
        for (x, y), p_xy in joint_probs.items():
            if p_xy > 0 and p_x[x] > 0 and p_y[y] > 0:
                mi += p_xy * np.log2(p_xy / (p_x[x] * p_y[y]))
        return mi
