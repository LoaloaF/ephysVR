"""
Closed-Loop Experiment Setup Script
This script sets up a closed-loop experiment following the MaxLab API tutorial.
It initializes the system, configures electrodes, creates stimulation sequences,
and prepares for data recording.
"""

import maxlab as mx
import maxlab.system
import maxlab.chip
import maxlab.util
import maxlab.saving
from pathlib import Path


# ==================== PARAMETERS ====================

# Electrode Configuration
trigger_electrode = 13248
closed_loop_electrode = 13378
recording_electrodes = [13248, 13378, 13200, 13400]  # Add more electrodes as needed

# Stimulation Parameters
trigger_stimulation_amplitude = 6  # in bits (for simulation purposes only)
closed_loop_stimulation_amplitude = 10  # in bits
stimulation_pulses_number = 10
phase_samples = 4  # Number of samples per phase (50us per sample)
inter_pulse_interval = 200  # Delay samples between pulses
delay_samples = 4  # Delay for stimulation pulse phase

# Recording Parameters
recording_duration = 60  # in seconds
recording_name = "closed_loop_experiment"
recording_directory = "/tmp"  # Change to desired directory
recording_wells = [0]  # For MaxTwo, specify wells (0 for MaxOne)

# Detection threshold
detection_threshold = 8.5  # Standard deviations for spike detection


# ==================== HELPER FUNCTIONS ====================

def create_stim_pulse(seq: mx.Sequence, amplitude: int, phase: int) -> mx.Sequence:
    """
    Creates a biphasic stimulation pulse.
    
    Args:
        seq: The sequence to append the pulse to
        amplitude: Amplitude of the pulse in bits
        phase: Number of samples per phase
    
    Returns:
        The modified sequence
    """
    seq.append(mx.DAC(0, 512 - amplitude))
    seq.append(mx.DelaySamples(phase))
    seq.append(mx.DAC(0, 512 + amplitude))
    seq.append(mx.DelaySamples(phase))
    seq.append(mx.DAC(0, 512))
    return seq


# ==================== SYSTEM INITIALIZATION ====================

print("Initializing closed-loop experiment...")

# Clear any pre-existing sequences
print("Clearing pre-existing sequences...")
try:
    s = mx.Sequence('trigger', persistent=False)
    del(s)
except:
    pass

try:
    s = mx.Sequence('closed_loop_sequence', persistent=False)
    del(s)
except:
    pass

# Initialize the system
print("Initializing system...")
mx.util.initialize()

# Set detection threshold
print(f"Setting detection threshold to {detection_threshold} standard deviations...")
mx.send_raw(f"stream_set_event_threshold {detection_threshold}")


# ==================== ARRAY CONFIGURATION ====================

print("\nConfiguring electrode array...")

# Create array
array = mx.chip.Array('stimulation')

# Clear any existing electrodes
array.reset()

# Select electrodes for recording
array.select_electrodes(recording_electrodes)

# Route electrodes to stimulation units
print(f"Routing trigger electrode {trigger_electrode} to stimulation unit 1...")
array.select_stimulation_electrode(trigger_electrode)
array.route_stimulation(trigger_electrode, 1)

print(f"Routing closed-loop electrode {closed_loop_electrode} to stimulation unit 2...")
array.select_stimulation_electrode(closed_loop_electrode)
array.route_stimulation(closed_loop_electrode, 2)

# Download array configuration to chip
print("Downloading array configuration to chip...")
array.download()

# Connect stimulation units
print("Connecting stimulation units to DAC...")
stim_units = mx.chip.StimulationUnit('all')
stim_units.connect(True)
stim_units.set_voltage_mode()
stim_units.dac_source(0)


# ==================== SEQUENCE PREPARATION ====================

print("\nPreparing stimulation sequences...")

# Create trigger sequence (for simulation only)
print("Creating trigger sequence...")
sequence_1 = mx.Sequence('trigger', persistent=True)
for _ in range(stimulation_pulses_number):
    sequence_1 = create_stim_pulse(sequence_1, trigger_stimulation_amplitude, delay_samples)
    sequence_1.append(mx.DelaySamples(inter_pulse_interval))

# Create closed-loop response sequence
print("Creating closed-loop response sequence...")
sequence_2 = mx.Sequence('closed_loop_sequence', persistent=True)
for _ in range(stimulation_pulses_number):
    sequence_2 = create_stim_pulse(sequence_2, closed_loop_stimulation_amplitude, delay_samples)
    sequence_2.append(mx.DelaySamples(inter_pulse_interval))


# ==================== DATA RECORDING ====================

print("\nSetting up data recording...")

# Configure recording file
recording_path = Path(recording_directory) / f"{recording_name}.raw.h5"
print(f"Recording will be saved to: {recording_path}")

# Start recording
print(f"Starting recording for {recording_duration} seconds...")
mx.saving.open_file(str(recording_path))

# For MaxTwo, specify wells
if len(recording_wells) > 0:
    print(f"Recording from wells: {recording_wells}")
    mx.saving.start_recording(recording_wells, recording_duration)
else:
    mx.saving.start_recording(recording_duration)

print("\n" + "="*60)
print("SETUP COMPLETE!")
print("="*60)
print("\nThe C++ closed-loop script should now be running to monitor")
print("the data stream and send stimulation responses.")
print("\nWhen ready, execute the trigger sequence with:")
print("  >>> sequence_1.send()")
print("\nOr start your actual experimental stimulation.")
print("\nRecording will automatically stop after the specified duration.")
print("="*60)
