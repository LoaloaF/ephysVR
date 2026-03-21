# Closed-Loop Experiment Setup

This directory contains a complete implementation of a closed-loop experiment for MaxOne/MaxTwo devices, following the official MaxLab API tutorial.

## Overview

Closed-loop experiments allow real-time monitoring of neural activity and immediate stimulation responses based on detected events. This implementation provides:

- **Python Setup Script**: Configures the system, electrodes, and stimulation sequences
- **C++ Data Processing**: Two approaches for monitoring data streams
  - Raw data stream processing
  - Filtered spike event processing
- **Build System**: CMake configuration for compiling C++ code

## Files

```
closed_loop/
├── README.md                    # This file
├── closedLoopSetup.py          # Python setup and configuration script
├── process_raw_data.cpp        # C++ raw data stream processor
├── process_spike_events.cpp    # C++ filtered spike event processor
└── CMakeLists.txt              # CMake build configuration
```

## Prerequisites

### Software Requirements
- MaxLab API installed (`~/MaxLab/` directory)
- **Valid MaxLab license** (required for runtime execution)
- Python 3.x with maxlab package
- C++ compiler (gcc 11.2.1 or newer)
- CMake 3.10 or newer
- GLIBC 2.3.2 or newer

### System Requirements
- MaxOne or MaxTwo device connected and configured
- `mxwserver` running with valid license
- Proper permissions to access device

### License Requirements

⚠️ **IMPORTANT**: The MaxLab library requires an active license to run. You must:
1. Have a valid license from MaxWell Biosystems
2. Start the MaxLab GUI application first (this activates the license)
3. Keep the GUI running while executing closed-loop experiments
4. If you see "No active license found", contact support@mxwbio.com

## Setup Instructions

### 1. Compile C++ Code

First, compile the C++ data processing scripts:

```bash
cd closed_loop

# Create build directory
mkdir -p build
cd build

# Configure with CMake (adjust MAXLAB_DIR if needed)
cmake .. -DMAXLAB_DIR=$HOME/MaxLab/share/maxlab_lib

# Build
make

# Executables will be in the build directory:
# - process_raw_data
# - process_spike_events
```

If the MaxLab library is in a different location, specify it:

```bash
cmake .. -DMAXLAB_DIR=/path/to/maxlab_lib
```

### 2. Configure Python Script

Edit [closedLoopSetup.py](closedLoopSetup.py) to adjust parameters for your experiment:

```python
# Electrode Configuration
trigger_electrode = 13248              # Trigger electrode ID
closed_loop_electrode = 13378          # Response electrode ID
recording_electrodes = [13248, 13378]  # Electrodes to record

# Stimulation Parameters
closed_loop_stimulation_amplitude = 10 # Amplitude in bits
stimulation_pulses_number = 10         # Pulses per train

# Recording Parameters
recording_duration = 60                # Seconds
recording_name = "my_experiment"       # Output file name
recording_directory = "/tmp"           # Output directory
```

## Running the Experiment

⚠️ **Before starting**: Ensure you have a valid MaxLab license activated by starting the MaxLab GUI application first.

The experiment must be started in a specific sequence:

### Step 1: Start MaxLab GUI (License Activation)

Launch the MaxLab GUI application to activate your license:
```bash
# Start MaxLab GUI (exact command may vary)
MaxLab
```

Verify the license is active in the GUI. Keep the GUI running.

### Step 2: Start mxwserver

Ensure `mxwserver` is running (usually starts automatically with the GUI).

### Step 3: Run C++ Data Processor

Choose either raw data or spike event processing:

**Option A: Raw Data Processing**
```bash
cd build
./process_raw_data 13248
```

**Option B: Spike Event Processing** (recommended for better performance)
```bash
cd build
./process_spike_events 13248
```

Replace `13248` with your detection channel electrode ID.

The C++ script will:
- Open a connection to the data stream
- Listen for spikes on the specified channel
- Send stimulation when condition is met
- Print status messages

### Step 3: Run Python Setup Script

In a separate terminal:

```bash
python3 closedLoopSetup.py
```

This script will:
- Initialize the system
- Configure electrodes and stimulation units
- Create stimulation sequences
- Start data recording

### Step 4: Monitor and Control

Once both scripts are running:

1. The C++ script monitors the data stream in real-time
2. When a spike is detected on the trigger channel:
   - The C++ script sends the closed-loop stimulation sequence
   - A blanking period prevents artifact detection
3. All activity is recorded to the specified file

To manually trigger stimulation (if needed):
```python
# In a Python console
import maxlab as mx
sequence_1.send()  # Send trigger sequence
```

## How It Works

### Data Flow

```
MaxOne/MaxTwo Device
        ↓
  Raw/Filtered Data Stream
        ↓
  C++ Processing Script
        ↓
  Condition Detection
        ↓
  Send Stimulation Sequence
        ↓
  Back to Device
```

### Blanking Period

After each stimulation, a blanking period (8000 frames ≈ 0.4 seconds at 20kHz) prevents:
- Detection of stimulation artifacts
- Multiple rapid stimulations
- False positives during voltage recovery

### Stimulation Sequence

The default sequence creates a biphasic pulse:

```
  0V ─────┐           ┌─────
          │           │
   -V ────┘     ┌─────┘
                │
               +V

  Phase 1  Delay  Phase 2  Delay
```

## Customization

### Modify Detection Condition (C++)

In `process_raw_data.cpp`, change the detection logic:

```cpp
// Current: Simple threshold
if (frameData.amplitudes[detection_channel] > 500)

// Custom: Multiple channel condition
if (frameData.amplitudes[channel1] > 500 && 
    frameData.amplitudes[channel2] > 300)

// Custom: Spike rate condition
if (spike_rate > threshold)
```

### Modify Stimulation Pattern (Python)

In `closedLoopSetup.py`, modify `create_stim_pulse()`:

```python
def create_stim_pulse(seq, amplitude, phase):
    # Monophasic pulse
    seq.append(mx.DAC(0, 512 + amplitude))
    seq.append(mx.DelaySamples(phase))
    seq.append(mx.DAC(0, 512))
    return seq
```

## Troubleshooting

### C++ Compilation Errors

**Error: Cannot find maxlab/maxlab.h**
- Check `MAXLAB_DIR` path in CMakeLists.txt
- Verify MaxLab installation: `ls ~/MaxLab/share/maxlab_lib`

**Error: Linking fails**
- Ensure libmaxlab.a exists in `$MAXLAB_DIR/lib`
- Try: `unzip ~/MaxLab/share/libmaxlab-*.zip -d ~/MaxLab/share/`

### Python Errors

**Error: No module named 'maxlab'**
- Install: `pip install maxlab`
- Or activate your conda/pyenv environment

**Error: Cannot connect to server**
- Ensure mxwserver is running
- Check device connection
- Verify permissions

### Runtime Issues

**No stimulation triggered**
- Verify detection channel is receiving spikes
- Lower detection threshold in C++ code
- Check electrode configuration

**"No active license found" error**
- Start the MaxLab GUI application first to activate license
- Ensure you have a valid license from MaxWell Biosystems
- Contact support@mxwbio.com if license issues persist
- Check license file location: `~/.mxw/` or `/usr/local/etc/mxw/`

**Multiple rapid stimulations**
- Increase blanking period (currently 8000 frames)
- Add additional conditions to detection logic

**Data not saving**
- Check recording directory exists and has write permissions
- Verify disk space available
- Check recording duration hasn't expired

## Advanced Features

### Custom Condition Function

Create reusable condition checking:

```cpp
bool should_stimulate(const maxlab::RawFrameData& frameData, int channel) {
    // Complex logic here
    return frameData.amplitudes[channel] > threshold &&
           other_conditions_met();
}

// In main loop:
if (should_stimulate(frameData, detection_channel)) {
    maxlab::sendSequence("closed_loop_sequence");
}
```

### Multi-Channel Detection

Monitor multiple channels:

```cpp
const int channels[] = {13248, 13378, 13200};
const int num_channels = 3;

for (int i = 0; i < num_channels; i++) {
    if (frameData.amplitudes[channels[i]] > threshold) {
        // Stimulate based on which channel fired
        send_channel_specific_sequence(channels[i]);
        break;
    }
}
```

### Temporal Pattern Detection

Implement spike pattern recognition:

```cpp
struct SpikeHistory {
    uint64_t timestamp;
    int channel;
};

std::vector<SpikeHistory> recent_spikes;

// Check for pattern (e.g., burst)
bool is_burst_pattern(const std::vector<SpikeHistory>& spikes) {
    // Implement pattern detection logic
    return /* pattern found */;
}
```

## References

- [MaxLab API Documentation](../mea1kdocumentation/MaxwellDocumentation/index.html)
- [Closed-Loop Tutorial](../mea1kdocumentation/MaxwellDocumentation/tutorial/closed_loop_tutorial.html)
- [Python API Tutorial](../mea1kdocumentation/MaxwellDocumentation/tutorial/python_api_tutorial.html)

## Notes

- The trigger sequence in this example is for simulation/demonstration
- In real experiments, triggers come from actual neural activity
- Always start the C++ script before the Python script
- Stop the C++ script with Ctrl+C when experiment is complete
- Data is saved in HDF5 format (.raw.h5)

## Support

For issues or questions:
1. Check the official MaxLab documentation
2. Verify all prerequisites are met
3. Review console output for error messages
4. Ensure all components are properly synchronized

---

Created following the MaxLab Closed-Loop API Tutorial
