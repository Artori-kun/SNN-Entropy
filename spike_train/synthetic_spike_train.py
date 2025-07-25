import numpy as np
from scipy.io import wavfile
from scipy.signal import resample, butter, lfilter, hilbert
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import librosa
import librosa.display

# Generate a synthetic spike train with different types of spikes
# including regular spike (poisson), burst spikes, periodic spikes, and silent spikes
# The generated spike trains are in binary format, where 1 indicates a spike and 0 indicates no spike.
# These spike trains have dimensions (n, t), where n is the number of neurons and t is the time steps.

class SpikeTrainGenerator:
    # 1. Regular spikes (Poisson process)
    # generate 2 types of regular spikes: homogeneous and inhomogeneous
    # homogeneous: constant rate
    # inhomogeneous: varying rate over time
    # thank you https://medium.com/@baxterbarlow/poisson-spike-generators-stochastic-theory-to-python-code-a76f8cc7cc32

    def __init__(self, seed=None):
        self.seed = seed
        if seed is not None:
            np.random.seed(self.seed)  # Set seed for reproducibility
        pass
    
    def plot_spike_train(self, spike_train, title="Spike Train", xlabel="Time (s)", ylabel="Neurons"):
        """
        Plot the spike train using matplotlib.
        
        Parameters:
        spike_train (np.ndarray): Binary spike train of shape (n_neurons, t).
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        """
        import matplotlib.pyplot as plt
        
        n_neurons, n_steps = spike_train.shape
        fig, ax = plt.subplots(figsize=(12, 6))
        for neuron in range(n_neurons):
            spike_times = np.where(spike_train[neuron])[0]
            ax.scatter(spike_times, [neuron]*len(spike_times), s=2)
            
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yticks(range(n_neurons))
        plt.tight_layout()
        plt.show()
    
    def generate_homogeneous_poisson_spikes(self, n_neurons, rate, duration, step=0.001):
        """
        Generate a Poisson spike train with a constant firing rate.
        
        Parameters:
        n_neurons (int): Number of neurons.
        rate (float): Spike rate/frequency in Hz.
        duration (float): Duration of the spike train in seconds.
        step (float): Time step for the spike train in seconds.
        
        Returns:
        np.ndarray: Binary spike train of shape (n_neurons, t), where t is the number of time steps.
        """
        
        n_steps = int(duration / step)
        spike_train = np.random.rand(n_neurons, n_steps) < (rate * step)
        return spike_train.astype(int)
    
    def generate_inhomogeneous_poisson_spikes(self, n_neurons, rate_fn, duration, step=0.001):
        """
        Generate a Poisson spike train with a varying firing rate.
        
        Parameters:
        n_neurons (int): Number of neurons.
        rate_fn (callable): Function that takes time as input and returns the firing rate at that time.
        duration (float): Duration of the spike train in seconds.
        step (float): Time step for the spike train in seconds.
        
        Returns:
        np.ndarray: Binary spike train of shape (n_neurons, t), where t is the number of time steps.
        """
        
        n_steps = int(duration / step)
        spike_matrix = np.zeros((n_neurons, n_steps), dtype=int)
        for t in range(n_steps):
            rate = rate_fn(t=t)
            spike_matrix[:, t] = (np.random.rand(n_neurons) < (rate * step)).astype(int)
        return spike_matrix
    
    # rate function for inhomogeneous spikes
    # using sinousoidal function to vary the rate over time
    def sinusoidal_rate_fn(self, t=0.001, amplitude=20, frequency=0.5, base_rate=10):
        """
        Sinusoidal rate function for varying firing rate.
        
        Parameters:
        t (int): Time step.
        amplitude (float): Amplitude of the sinusoidal function.
        frequency (float): Frequency of the sinusoidal function.
        base_rate (float): Base firing rate.
        
        Returns:
        float: Firing rate at time t.
        """
        return base_rate + amplitude * np.sin(2 * np.pi * frequency * t) + 1  # Ensure non-negative rates
    
    # 2. Burst spikes
    # the signal is temporally clustered, with bursts of spikes followed by silent periods
    # attributes:
    # - burst interval: time between burst clusters
    # -burst size: number of spikes in each burst
    # - silent period: time after each burst with no spikes
    # - intra-burst rate: rate of spikes within a burst
    def generate_burst_spike(
        self,
        n_neurons,
        duration,
        step=0.001,
        burst_interval=100,
        burst_size=4
    ):
        """
        Generate a burst spike train with clusters of spikes.
        
        Parameters:
        n_neurons (int): Number of neurons.
        duration (float): Duration of the spike train in seconds.
        step (float): Time step for the spike train in seconds.
        burst_interval (int): Time steps between bursts.
        burst_size (int): Number of spikes in each burst.
        
        Returns:
        np.ndarray: Binary spike train of shape (n_neurons, t), where t is the number of time steps.
        """
        
        n_steps = int(duration / step)
        spike_train = np.zeros((n_neurons, n_steps), dtype=int)
        
        for neuron in range(n_neurons):
            t = 0
            while t < n_steps:
                # Randomly decide when the burst starts within the burst interval
                burst_start = t + np.random.randint(0, max(1, burst_interval // 4))
                
                # Generate burst spikes
                for b in range(burst_size):
                    t_burst = burst_start + b
                    if t_burst < n_steps:
                        spike_train[neuron, t_burst] = 1
                # Move to the next burst interval
                t = burst_start + burst_interval
        
        return spike_train
    
    # 3. Periodic spikes
    # the signal has a regular pattern of spikes at fixed intervals
    # each neuron fire spike at a fixed interval
    # each nẻuons can have different firing intervals or the same firing interval
    def generate_periodic_spikes(self, n_neurons, intervals, duration, step=0.001):
        """
        Generate a periodic spike train with spikes at fixed intervals.
        
        Parameters:
        n_neurons (int): Number of neurons.
        intervals (list): List of spike intervals for each neuron in time steps. Can set to None for uniform intervals.
        duration (float): Duration of the spike train in seconds.
        step (float): Time step for the spike train in seconds.
        
        Returns:
        np.ndarray: Binary spike train of shape (n_neurons, t), where t is the number of time steps.
        """
        
        n_steps = int(duration / step)
        spike_train = np.zeros((n_neurons, n_steps), dtype=int)
        
        if intervals is None:
            # If no specific intervals are provided, use a default interval of 50 time steps
            intervals = [50] * n_neurons
        
        for neuron in range(n_neurons):
            interval = intervals[neuron]
            phase = np.random.randint(0, interval)
            spike_times = np.arange(phase, n_steps, interval)
            spike_train[neuron, spike_times] = 1
            
        return spike_train
    
    # 4. Silent spikes
    # the signal has almost no spikes, with only a few random spikes occurring or completely silent
    # represented a neuron that is silent most of the time
    def generate_silent_spikes(self, n_neurons, duration, step=0.001, silent_rate=0.01):
        """
        Generate a spike train with very few spikes (silent spikes).
        
        Parameters:
        n_neurons (int): Number of neurons.
        duration (float): Duration of the spike train in seconds.
        step (float): Time step for the spike train in seconds.
        silent_rate (float): Probability of a spike occurring at each time step.
        
        Returns:
        np.ndarray: Binary spike train of shape (n_neurons, t), where t is the number of time steps.
        """
        
        assert 0 <= silent_rate <= 1, "silent_rate must be between 0 and 1"
        
        n_steps = int(duration / step)
        spike_train = (np.random.rand(n_neurons, n_steps) < silent_rate).astype(int)
        return spike_train
    
    # Convert an audio signal to a spike train
    # using LIF encoding
    def audio_to_spike_lif(
        self, 
        filepath, 
        duration=1000,
        dt=1, 
        tau=15.0, 
        theta=0.7,
        resample = True,
        plot=True
    ):
        """
        Convert an audio signal to a spike train using LIF encoding.
        
        Parameters:
        filepath (string): Path to audio file.
        duration (int): Duration of the audio signal in ms.
        dt (int): Time step for the spike train in ms.
        tau (float): Membrane time constant in ms.
        theta (float): Spike threshold.
        silence_threshold (float): Threshold to consider a value as silent.
        
        plot (bool): Whether to plot the audio signal and spike train.
        
        Returns:
        np.ndarray: Binary spike train of shape (n_neurons, time_steps).
        """
        
        data, rate = librosa.load(filepath, sr=None, mono=True)
        if data.ndim > 1:
            data = data[:, 0] # Use only one channel if stereo
        
        # Normalize the audio signal
        # data = data[int(0.2 * rate):int(rate * (duration / 1000 + 0.2))]  # Trim to duration
        data = data.astype(np.float32)
        if resample:
            n_steps = int(duration / dt)
            data = resample(data, n_steps)  # Resample to match duration
        
        max_val = max(abs(data.max()), abs(data.min()))
        if max_val > 0:
            data /= max_val
        
        # LIF encoding
        data_abs = np.asarray(np.abs(data))
        # print(data)
        n_steps = len(data)
        w = np.zeros(n_steps) 
        u = np.zeros(n_steps + 1)
        
        u[0] = data[0]  # Initial membrane potential
        decay = np.exp(-dt / tau)
        
        for t in range(n_steps):
            u[t + 1] = u[t] * decay + data_abs[t]
            if u[t + 1] >= theta:
                w[t] = 1
                u[t + 1] = 0  # Reset membrane potential after spike
        
        if plot:
            plt.figure(figsize=(10, 4))
            librosa.display.waveshow(data, sr=rate, alpha=0.5)
            plt.title("Waveform of sample_2_label_7.wav")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            plt.show()
            
            self.plot_spike_train(w.reshape(1, -1), title="LIF Encoded Spike Train", xlabel="Time (ms)", ylabel="Neuron Index")
        
        return w.astype(int).reshape(1, -1)  # Reshape to (1, n_steps) for single neuron spike train
    
    # === Audio encoding to spike train, biologically inspired ===
    # includes 3 steps:
    # Basilar Membrane filtering
    # Hair Cell Encoding
    # Bushy Cell Encoding
    def audio_to_spike_bio(
        self,
        filepath,
        duration=1000,
        dt=1,
        num_bands=16,
        tau=10.0,
        theta=0.9,
        threshold=4,
        drop_weak_channels=True,
        drop_percent=30,
    ):
        """
        Convert an audio signal to a spike train using a biologically inspired model.
        
        Parameters:
        filepath (str): Path to the audio file.
        duration (int): Duration of the audio signal in ms.
        dt (int): Time step for the spike train in ms.
        num_bands (int): Number of Mel bands for frequency selectivity.
        tau (float): Membrane time constant for LIF encoding.
        theta (float): Spike threshold for LIF encoding.
        threshold (float): Threshold for bushy cell filter.
        drop_weak_channels (bool): Whether to drop weak channels based on activity.
        drop_percent (int): Percentage of weak channels to drop if drop_weak_channels is True.
        
        Returns:
        np.ndarray: Binary spike train of shape (n_neurons, time_steps).
        """
        import librosa
        from librosa.filters import mel 
        
        y, sr = librosa.load(filepath, sr=None)
        y = y / (np.max(np.abs(y)) + 1e-9)
        
        num_steps = int(duration / dt)
        
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=num_bands, 
            n_fft=1024, 
            hop_length=len(y) // num_steps
        )
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        if drop_weak_channels:
            mel_energy = np.mean(mel_spectrogram, axis=1)
            drop_threshold = np.percentile(mel_energy, drop_percent)
            mel_spectrogram = mel_spectrogram[mel_energy >= drop_threshold]
        
        envelopes = [np.abs(hilbert(resample(band, num_steps))) for band in mel_spectrogram]
        max_env = np.max([np.max(env) for env in envelopes])
        envelopes = [env / (max_env + 1e-9) for env in envelopes]
      
        spike_matrix = []
        
        for env in envelopes:
            print(f"Envelope: {env}")
            v = 0.0
            decay = np.exp(-dt / tau)
            spikes = np.zeros(num_steps, dtype=int)
            
            for t in range(num_steps):
                v = v * decay + env[t]
                print(f"Time {t}, Membrane potential: {v}")
                if v >= theta:
                    spikes[t] = 1
                    v = 0.0  # Reset membrane potential after spike
            
            spike_matrix.append(spikes)
            
        spike_matrix = np.array(spike_matrix)
        # channel_activity = np.mean(spike_matrix, axis=1)
        # top_k_idx = np.argsort(channel_activity)[-num_bands:]
        # spike_matrix = spike_matrix[top_k_idx]
        summed = np.sum(spike_matrix, axis=0)
        smoothed = uniform_filter1d(summed, size=15)
        bc_threshold = np.percentile(smoothed, 80)
        spike_output = (smoothed >= bc_threshold).astype(int)
        spike_output = spike_output.reshape(1, -1)  # Reshape to (1, n_steps) for single neuron spike train
        
        plt.imshow(spike_matrix, aspect='auto', cmap='gray_r')
        plt.title("Spike Matrix từ HC (trước khi BC nén)")
        plt.xlabel("Time")
        plt.ylabel("Mel Channel")
        plt.show()
        
        
        # Plot the spike train
        self.plot_spike_train(spike_output, title="Biologically Inspired Spike Train", xlabel="Time (ms)", ylabel="Neuron Index")
        
        return spike_output