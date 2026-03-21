"""
Configuration Template for Closed-Loop Experiments

Copy this file and modify the parameters for your specific experiment.
Then import it in closedLoopSetup.py:
    from config_template import *
"""

# ==================== ELECTRODE CONFIGURATION ====================

# Primary electrodes for closed-loop experiment
TRIGGER_ELECTRODE = 13248           # Electrode that triggers stimulation
CLOSED_LOOP_ELECTRODE = 13378       # Electrode that delivers response

# All electrodes to record from
RECORDING_ELECTRODES = [
    13248,  # Trigger electrode
    13378,  # Closed-loop electrode
    13200,  # Additional recording electrodes
    13400,
    13300,
    13500,
]


# ==================== STIMULATION PARAMETERS ====================

# Amplitude Configuration (in bits, 1 bit = 2.9 mV)
# To calculate: amplitude_bits = voltage_mV / 2.9
# Example: 29 mV = 10 bits
TRIGGER_STIMULATION_AMPLITUDE = 6        # For simulation only
CLOSED_LOOP_STIMULATION_AMPLITUDE = 10   # Main response amplitude

# Pulse Timing (in samples, 1 sample = 50 microseconds)
PHASE_SAMPLES = 4                   # Duration of each pulse phase
INTER_PULSE_INTERVAL = 200          # Delay between pulses in a train
DELAY_SAMPLES = 4                   # Phase delay for pulse creation

# Pulse Train Configuration
STIMULATION_PULSES_NUMBER = 10      # Number of pulses per train


# ==================== DETECTION PARAMETERS ====================

# Spike detection threshold (standard deviations above noise)
DETECTION_THRESHOLD = 8.5

# Additional detection parameters for C++ code
# (These are used in the C++ scripts, documented here for reference)
RAW_AMPLITUDE_THRESHOLD = 500       # Threshold for raw data processing
BLANKING_PERIOD_FRAMES = 8000       # Post-stimulation blanking (~0.4s at 20kHz)


# ==================== RECORDING PARAMETERS ====================

# Recording Duration
RECORDING_DURATION = 60             # Duration in seconds

# File Configuration
RECORDING_NAME = "closed_loop_experiment"
RECORDING_DIRECTORY = "/tmp"        # Change to your data directory

# For MaxTwo: Specify wells to record from
# For MaxOne: Use [0] or leave as is
RECORDING_WELLS = [0]


# ==================== SYSTEM CONFIGURATION ====================

# Sampling rate (read-only, determined by device)
SAMPLING_RATE = 20000               # Hz (20 kHz)

# Filter settings for spike detection (used in C++ filtered stream)
FILTER_TYPE = "IIR"                 # IIR or other supported types


# ==================== EXPERIMENTAL CONDITIONS ====================

# Custom experimental parameters
EXPERIMENT_TYPE = "closed_loop"
EXPERIMENTER = "YourName"
EXPERIMENT_DATE = "2026-01-26"
NOTES = """
Add any notes about this experiment configuration here.
Examples:
- Cell type
- Culture conditions
- Expected responses
- Special protocols
"""


# ==================== ADVANCED SETTINGS ====================

# DAC Configuration
DAC_CHANNEL = 0                     # DAC channel for stimulation
DAC_NEUTRAL = 512                   # Neutral voltage (0V) in bits

# Stimulation unit configuration
STIM_UNIT_1 = 1                     # Unit for trigger electrode
STIM_UNIT_2 = 2                     # Unit for closed-loop electrode

# Array configuration
ARRAY_NAME = "stimulation"          # Name for the electrode array


# ==================== HELPER FUNCTIONS ====================

def voltage_to_bits(voltage_mv):
    """
    Convert voltage in millivolts to DAC bits.
    
    Args:
        voltage_mv: Voltage in millivolts
    
    Returns:
        DAC value in bits (rounded to nearest integer)
    """
    return round(voltage_mv / 2.9)


def bits_to_voltage(bits):
    """
    Convert DAC bits to voltage in millivolts.
    
    Args:
        bits: DAC value in bits
    
    Returns:
        Voltage in millivolts
    """
    return bits * 2.9


def samples_to_time(samples):
    """
    Convert number of samples to time in seconds.
    
    Args:
        samples: Number of samples
    
    Returns:
        Time in seconds
    """
    return samples / SAMPLING_RATE


def time_to_samples(time_seconds):
    """
    Convert time in seconds to number of samples.
    
    Args:
        time_seconds: Time in seconds
    
    Returns:
        Number of samples (rounded to nearest integer)
    """
    return round(time_seconds * SAMPLING_RATE)


# ==================== VALIDATION ====================

def validate_config():
    """
    Validate configuration parameters.
    Raises ValueError if configuration is invalid.
    """
    errors = []
    
    # Check amplitude values
    if CLOSED_LOOP_STIMULATION_AMPLITUDE < 0 or CLOSED_LOOP_STIMULATION_AMPLITUDE > 511:
        errors.append(f"Invalid amplitude: {CLOSED_LOOP_STIMULATION_AMPLITUDE} (must be 0-511)")
    
    # Check electrode IDs
    if TRIGGER_ELECTRODE == CLOSED_LOOP_ELECTRODE:
        errors.append("Trigger and closed-loop electrodes cannot be the same")
    
    # Check recording duration
    if RECORDING_DURATION <= 0:
        errors.append(f"Invalid recording duration: {RECORDING_DURATION}")
    
    # Check detection threshold
    if DETECTION_THRESHOLD < 0:
        errors.append(f"Invalid detection threshold: {DETECTION_THRESHOLD}")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    print("Configuration validation passed!")
    return True


# ==================== CONFIGURATION SUMMARY ====================

def print_config_summary():
    """Print a summary of the current configuration."""
    print("="*60)
    print("CLOSED-LOOP EXPERIMENT CONFIGURATION")
    print("="*60)
    print(f"Trigger Electrode: {TRIGGER_ELECTRODE}")
    print(f"Closed-Loop Electrode: {CLOSED_LOOP_ELECTRODE}")
    print(f"Recording Electrodes: {len(RECORDING_ELECTRODES)} electrodes")
    print(f"Stimulation Amplitude: {CLOSED_LOOP_STIMULATION_AMPLITUDE} bits ({bits_to_voltage(CLOSED_LOOP_STIMULATION_AMPLITUDE):.1f} mV)")
    print(f"Pulses per Train: {STIMULATION_PULSES_NUMBER}")
    print(f"Detection Threshold: {DETECTION_THRESHOLD} σ")
    print(f"Recording Duration: {RECORDING_DURATION} seconds")
    print(f"Output File: {RECORDING_DIRECTORY}/{RECORDING_NAME}.raw.h5")
    print("="*60)


if __name__ == "__main__":
    # When run directly, validate and print configuration
    print_config_summary()
    validate_config()
