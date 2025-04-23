"""
ECE-508 Project - FM Radio Receiver - Custom Python Blocks - WBFM

Author: Phil Nevins
"""

import numpy as np
from gnuradio import gr
from scipy.signal import firwin, lfilter

class blk(gr.sync_block):  
    def __init__(self, quad_rate=2400000, audio_rate=48000):  
        gr.sync_block.__init__(
            self,
            name="Clean WBFM Demodulator",
            in_sig=[np.complex64],
            out_sig=[np.float32]
        )
        self.quad_rate = quad_rate
        self.audio_rate = audio_rate
        self.decim = int(quad_rate // audio_rate)

        # FIR low-pass filter (cutoff just below audio range)
        self.lpf_taps = firwin(numtaps=101, cutoff=16e3, fs=quad_rate)

        # De-emphasis filter (75us for US)
        tau = 75e-6
        self.alpha = np.exp(-1.0 / (audio_rate * tau))
        self.prev_y = 0.0

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]

        # FM demod via conjugate product
        phasor_diff = in0[1:] * np.conj(in0[:-1])
        demod = np.angle(phasor_diff)
        demod = np.insert(demod, 0, 0)

        # FIR LPF before decimation to prevent aliasing
        filtered_baseband = lfilter(self.lpf_taps, 1.0, demod)

        # Decimate safely
        decimated = filtered_baseband[::self.decim]

        # De-emphasis filter
        filtered = np.zeros_like(decimated)
        for i in range(len(decimated)):
            self.prev_y = (1 - self.alpha) * decimated[i] + self.alpha * self.prev_y
            filtered[i] = self.prev_y

        # Remove residual DC
        filtered -= np.mean(filtered)

        # Scale gain to avoid clipping
        filtered *= 0.2

        out[:len(filtered)] = filtered.astype(np.float32)
        return len(filtered)

