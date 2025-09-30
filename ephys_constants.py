import platform
import sys
import os
import pandas as pd
import numpy as np

SAMPLING_RATE = 20_000
MAX_AMPL_mV = 3300.
ADC_RESOLUTION = 2**10

DEVICE_NAME_RAT006 = '241016_MEA1K03_H1278pad4shankB5'
DEVICE_NAME_RAT011 = '241211_MEA1K06_H1278pad4shankB5'

MEA_LOGGER_DEFAULT_GAIN = 512 # used for logger recordings
MEA_OVERRIDE_GAIN = 7
# MEA_OVERRIDE_GAIN = None

SHANK_BASE_COLORS = {1.0: np.array((4, 81, 128))/255,
                     3.0: np.array((0, 96, 51))/255,
                     2.0: np.array((111, 0, 116))/255,
                     4.0: np.array((158, 50, 3))/255,
}
METALLIZATION_COLOR_OFFSET = .5