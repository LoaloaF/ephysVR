"""
Example: Custom Closed-Loop Setup
This demonstrates how to use the config_template for your experiment.
"""

# Import configuration
from config_template import *
import maxlab as mx
import maxlab.system
import maxlab.chip
import maxlab.util
import maxlab.saving
from pathlib import Path


# ==================== CUSTOM PARAMETERS ====================
# Override default configuration here

# Example: Custom electrode configuration
RECORDING_ELECTRODES = [13248, 13378, 13200, 13400, 13100, 13500]
CLOSED_LOOP_STIMULATION_AMPLITUDE = 15  # 43.5 mV

# Example: Longer recording
RECORDING_DURATION = 120  # 2 minutes

# Example: Custom file naming
RECORDING_NAME = f"experiment_{EXPERIMENT_DATE}_neuron1"
RECORDING_DIRECTORY = "/data/experiments"  # Your data directory


# ==================== SETUP SCRIPT ====================

print_config_summary()
validate_config()

print("\nInitializing closed-loop experiment...")

# Clear pre-existing sequences
print("Clearing pre-existing sequences...")
for seq_name in ['trigger', 'closed_loop_sequence']:
    try:
        s = mx.Sequence(seq_name, persistent=False)
        del(s)
    except:
        pass

# Initialize system
print("Initializing system...")
mx.util.initialize()

# Set detection threshold
print(f"Setting detection threshold to {DETECTION_THRESHOLD} standard deviations...")
mx.send_raw(f"stream_set_event_threshold {DETECTION_THRESHOLD}")


# Configure electrode array
print("\nConfiguring electrode array...")
array = mx.chip.Array(ARRAY_NAME)
array.reset()
array.select_electrodes(RECORDING_ELECTRODES)

print(f"Routing trigger electrode {TRIGGER_ELECTRODE}...")
array.select_stimulation_electrode(TRIGGER_ELECTRODE)
array.route_stimulation(TRIGGER_ELECTRODE, STIM_UNIT_1)

print(f"Routing closed-loop electrode {CLOSED_LOOP_ELECTRODE}...")
array.select_stimulation_electrode(CLOSED_LOOP_ELECTRODE)
array.route_stimulation(CLOSED_LOOP_ELECTRODE, STIM_UNIT_2)

print("Downloading array configuration...")
array.download()

# Connect stimulation units
print("Connecting stimulation units...")
stim_units = mx.chip.StimulationUnit('all')
stim_units.connect(True)
stim_units.set_voltage_mode()
stim_units.dac_source(DAC_CHANNEL)


# Create stimulation sequences
def create_stim_pulse(seq: mx.Sequence, amplitude: int, phase: int) -> mx.Sequence:
    """Create biphasic stimulation pulse."""
    seq.append(mx.DAC(DAC_CHANNEL, DAC_NEUTRAL - amplitude))
    seq.append(mx.DelaySamples(phase))
    seq.append(mx.DAC(DAC_CHANNEL, DAC_NEUTRAL + amplitude))
    seq.append(mx.DelaySamples(phase))
    seq.append(mx.DAC(DAC_CHANNEL, DAC_NEUTRAL))
    return seq


print("\nPreparing stimulation sequences...")

# Trigger sequence (for simulation)
sequence_1 = mx.Sequence('trigger', persistent=True)
for _ in range(STIMULATION_PULSES_NUMBER):
    sequence_1 = create_stim_pulse(sequence_1, TRIGGER_STIMULATION_AMPLITUDE, DELAY_SAMPLES)
    sequence_1.append(mx.DelaySamples(INTER_PULSE_INTERVAL))

# Closed-loop response sequence
sequence_2 = mx.Sequence('closed_loop_sequence', persistent=True)
for _ in range(STIMULATION_PULSES_NUMBER):
    sequence_2 = create_stim_pulse(sequence_2, CLOSED_LOOP_STIMULATION_AMPLITUDE, DELAY_SAMPLES)
    sequence_2.append(mx.DelaySamples(INTER_PULSE_INTERVAL))


# Start recording
print("\nSetting up data recording...")
recording_path = Path(RECORDING_DIRECTORY) / f"{RECORDING_NAME}.raw.h5"
print(f"Recording will be saved to: {recording_path}")

# Ensure directory exists
recording_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Starting recording for {RECORDING_DURATION} seconds...")
mx.saving.open_file(str(recording_path))

if len(RECORDING_WELLS) > 0:
    print(f"Recording from wells: {RECORDING_WELLS}")
    mx.saving.start_recording(RECORDING_WELLS, RECORDING_DURATION)
else:
    mx.saving.start_recording(RECORDING_DURATION)

print("\n" + "="*60)
print("CUSTOM SETUP COMPLETE!")
print("="*60)
print("\nThe C++ closed-loop script should now be running.")
print("\nExperiment configuration:")
print(f"  - Type: {EXPERIMENT_TYPE}")
print(f"  - Experimenter: {EXPERIMENTER}")
print(f"  - Date: {EXPERIMENT_DATE}")
print(f"  - Duration: {RECORDING_DURATION} seconds")
print(f"  - Output: {recording_path}")
print("\nNotes:")
print(NOTES)
print("="*60)
