"""
ECE-508 Project - FM Radio Receiver - Custom Python Blocks - WBFM

Author: Phil Nevins, Arie J
"""

import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    """Drop-in WBFM demodulator that matches GNU Radio's WBFM Receive block"""

    def __init__(self, quad_rate=2.4e6, audio_decim=10):
        gr.sync_block.__init__(
            self,
            name="WBFM Receive Custom",
            in_sig=[np.complex64],
            out_sig=[np.float32]
        )

        self.quad_rate = quad_rate
        self.audio_decim = audio_decim
        self.prev = 0j

    def work(self, input_items, output_items):
        x = input_items[0]
        out = output_items[0]

        # FM demodulation: conjugate multiply
        x = np.array(x)
        y = x * np.conj(np.roll(x, 1))
        y[0] = x[0] * np.conj(self.prev)
        self.prev = x[-1]
        demod = np.angle(y)

        # Decimate by audio_decim
        decimated = demod[::self.audio_decim]

        # Normalize and scale to make it audible
        decimated *= 1.0

        # Limit to available output buffer
        n = min(len(out), len(decimated))
        out[:n] = decimated[:n]

        return n


