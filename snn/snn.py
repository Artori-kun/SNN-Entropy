import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from brian2 import *
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import mutual_info_score
import pandas as pd
from entropy.entropy import EntropyCalculator

class SpikingNeuralNetwork:
    def __init__(self):
        self.entropy_calculator = EntropyCalculator()
        super().__init__()
    
    def poisson_encode(self, images, duration=100):
        """
        Convert static images (pixel 0–1) to Poisson spike trains.
        Args:
            images (Tensor): [batch, 1, 28, 28]
            duration (int): number of time steps
        Returns:
            spikes (Tensor): [batch, 1, 28, 28, time]
        """
        batch_size = images.shape[0]
        images = images.unsqueeze(-1)  # shape: [B, 1, 28, 28, 1]
        random_tensor = torch.rand((batch_size, 1, 28, 28, duration))
        spikes = (random_tensor < images).float()
        return spikes  # 0/1 spike tensor
    
    def encode_audio_to_spike_train(self, filepath='', n_neurons=100, duration_ms=500, plot=True):
        """
        Encode 1 file .wav thành spike train (Poisson encoding).
        
        Parameters:
            filepath (str): đường dẫn file .wav
            n_neurons (int): số lượng neuron (chiều dọc spike train)
            duration_ms (int): thời gian mô phỏng (ms)
            max_rate (float): tốc độ bắn tối đa (Hz)
            plot (bool): có vẽ raster plot không
            
        Returns:
            spike_train (np.ndarray): [n_neurons, time_steps]
        """
        # Đọc file âm thanh
        audio, sr = sf.read(filepath)
        audio = audio[int(0.2 * sr):int(sr * (duration_ms / 1000 + 0.2))]  # cắt theo duration
        audio = audio.astype(np.float32)  
        audio /= np.max(np.abs(audio) + 1e-8)  # chuẩn hóa về [0,1]

        # Resize tín hiệu để phù hợp số neuron
        audio_resized = np.interp(np.linspace(0, len(audio), duration_ms), np.arange(len(audio)), audio)

        # Poisson encoding
        rate = np.clip(audio_resized, 0, 1) * 10000
        prob = np.clip(rate * 1e-3, 0, 1)
        spike_train = (np.random.rand(n_neurons, duration_ms) < prob).astype(int)

        if plot:
            plt.plot(audio)
            plt.title(f"Audio Signal from: {filepath}")
            plt.show()
            
            plt.figure(figsize=(10, 4))
            for i in range(n_neurons):
                ts = np.where(spike_train[i])[0]
                plt.vlines(ts, i, i + 1, color='black', linewidth=0.5)
            plt.xlabel("Time (ms)")
            plt.ylabel("Neuron index")
            plt.title(f"Spike Train from: {filepath}")
            plt.tight_layout()
            plt.show()

        return spike_train.astype(int)
    
    def save_to_csv(self, result, filename=f"result/entropy_results_{int(time.time())}.csv"):
        """
        Save the results to a CSV file.
        
        Args:
            result (dict): Dictionary containing entropy results.
            filename (str): Name of the output CSV file.
        """
        results_df = pd.DataFrame({
            "Epoch": list(range(1, len(result['shannon_entropies']) + 1)),
            "Shannon Entropy": result['shannon_entropies'],
            "Entropy Rate": result['entropy_rates'],
            "Conditional Entropy": result['conditional_entropies'],
            "Mutual Information": result['mutual_infos'],
            "Weight": result['weights']
        })
        
        results_df.to_csv(filename, index=False)
        
        print(results_df.head(10))
    
    def train_single_stdp(
        self,
        spike_train,
        duration=1000*ms,
        interval=100*ms,
        v_threshold=0.05,
        refractory=5*ms,
        gmax=0.1,
        title=""
    ):
        """
        Train a simple SNN (1 input → 1 output neuron) using STDP with repeated spike input.
        Track Shannon entropy, entropy rate, conditional entropy, and mutual information at each interval.
        
        Parameters:
            spike_train (np.ndarray): Binary spike train of shape (1, T).
            duration (Quantity): Duration of each simulation.
            interval (int): Simulation interval, where the entropies are logged.

        Returns:
            dict: Metrics across epochs (entropy, weight, spikes).
        """
        
        defaultclock.dt = 1 * ms
        spike_train = np.asarray(spike_train)
        assert spike_train.shape[0] == 1, "Spike train must have 2 dimensions"
        n_steps = int(duration / interval)
        T = int(interval / defaultclock.dt)
        
        start_scope()
        
        indices, times = np.where(spike_train == 1)
        spike_times = times * defaultclock.dt
        input_group = SpikeGeneratorGroup(1, indices, spike_times)
        
        # model parameters
        taupre = taupost = 20 * ms
        taum = 10 * ms
        taue = 15*ms
        gmax = gmax  # max synaptic weight
        dApre = 0.01
        dApost = -dApre * taupre / taupost * 1.05
        #dApost *= gmax
        #dApre *= gmax
        v_threshold = v_threshold 
        
        # neuron model
        eqs = '''
        dv/dt = (ge - v) / taum : 1
        dge/dt = -ge / taue : 1
        '''
        
        output = NeuronGroup(1, eqs, threshold='v > v_threshold', reset='v = 0.0', method='exact', refractory=refractory)
        output.v = 0  # reset voltage
        
        # synapse
        syn = Synapses(input_group, 
                       output,
                       '''
                       w : 1
                       dApre/dt = -Apre / taupre : 1 (event-driven)
                       dApost/dt = -Apost / taupost : 1 (event-driven)
                       ''',
                       on_pre='''
                       ge_post += w
                       Apre += dApre
                       w = clip(w + Apost, 0, gmax)
                       ''',
                       on_post='''
                       Apost += dApost
                       w = clip(w + Apre, 0, gmax)
                       '''
                    )
        
        syn.connect()
        syn.w = 0.05  # random initial weights
        
        shannon_entropies = []
        entropy_rates = []
        conditional_entropies = []
        mutual_infos = []
        
        prev_spike_vector = None
        
        spikemon = SpikeMonitor(output)
        statemon = StateMonitor(syn, 'w', record=True)
        v_mon = StateMonitor(output, 'v', record=True)

        run(duration)
        
        spike_matrix = np.zeros((n_steps, T))
        for i, t in zip(spikemon.i, spikemon.t):
            interval_idx = int(t / interval)
            time_idx = int((t % interval) / defaultclock.dt)
            if interval_idx < n_steps and time_idx < T:
                spike_matrix[interval_idx, time_idx] = 1
        
        for i in range(n_steps):
            spike_vector = spike_matrix[i]
            
            shannon_entropy = self.entropy_calculator.shannon_entropy(spike_vector)
            entropy_rate = self.entropy_calculator.entropy_rate(spike_vector)
            # p = np.mean(spike_vector)
            # shannon_entropy = -p * np.log2(p + 1e-10) - (1 - p) * np.log2(1 - p + 1e-10) if p not in [0, 1] else 0
            # entropy_rate = shannon_entropy / T
            
            conditional_entropy = 0
            mutual_info = 0
            if prev_spike_vector is not None:
                if np.any(spike_vector) and np.any(prev_spike_vector):
                    mutual_info = self.entropy_calculator.mutual_information(prev_spike_vector, spike_vector)
                    conditional_entropy = self.entropy_calculator.conditional_entropy(spike_vector, prev_spike_vector)
            
            shannon_entropies.append(shannon_entropy)
            entropy_rates.append(entropy_rate)
            conditional_entropies.append(conditional_entropy)
            mutual_infos.append(mutual_info)
            prev_spike_vector = spike_vector.copy()
        
        print(f"Spikes: {spikemon.count[0]}")
        
        # Plot results
        time_axis = np.arange(n_steps) * float(interval / second)
        plt.figure(figsize=(14, 6))
        
        plt.subplot(2, 2, 1)
        plt.plot(time_axis, shannon_entropies, label='Shannon Entropy')
        plt.plot(time_axis, entropy_rates, label='Entropy Rate')
        plt.legend()
        plt.title("Shannon Entropy and Entropy Rate")
        
        plt.subplot(2, 2, 2)
        plt.plot(time_axis, conditional_entropies, label='Conditional Entropy')
        plt.plot(time_axis, mutual_infos, label='Mutual Information')
        plt.legend()
        plt.title("Conditional Entropy and Mutual Information")
        
        plt.subplot(2, 2, 3)
        plt.plot(statemon.t[::100] / second, statemon.w[0][::100])
        plt.title("Synaptic Weight Evolution")
        plt.xlabel("Time (s)")
        plt.ylabel("Weight")
        
        
        plt.subplot(2, 2, 4)
        plt.plot(v_mon.t[::100] / second, v_mon.v[0][::100])
        plt.title("Output Neuron Spike Times")
        plt.xlabel("Time (s)")
        plt.ylabel("Output Neuron Voltage")
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
        
        return {
            'shannon_entropies': shannon_entropies,
            'entropy_rates': entropy_rates,
            'conditional_entropies': conditional_entropies,
            'mutual_infos': mutual_infos,
            # 'weights': weights
        }
    
                
    def train_binary_classifier_stdp(
        self,
        spike_trains,
        v_threshold=0.05,
        gmax=1.0,
        n_output_neurons=2,
        title="Binary Classifier STDP Training"
    ):
        """
        Train a binary classifier using STDP on multiple spike trains. The network consists of 1 input neuron, connected to 2 output neurons via STDP synapses.
        Each output neuron corresponds to a class (0 or 1). The network learns to classify the input spike trains based on their firing patterns.
        Entropies are measured at each epoch.

        Args:
            spike_trains: List of binary spike trains, where each train is a 1D numpy array.
            v_threshold (float, optional): Voltage threshold for spike generation. Defaults to 0.05.
            gmax (float, optional): Maximum synaptic conductance. Defaults to 1.0.
            title (str, optional): Title for the training plot. Defaults to "Binary Classifier STDP Training".
        """
        
        defaultclock.dt = 1 * ms
        start_scope()
        
        P = PoissonGroup
        
        