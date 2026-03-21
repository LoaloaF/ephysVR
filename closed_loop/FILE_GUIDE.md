# Closed-Loop Experiment Files

## File Overview

This directory contains a complete implementation of closed-loop experiments for MaxOne/MaxTwo MEA devices.

### Core Files

1. **closedLoopSetup.py**
   - Main Python setup script
   - Configures system, electrodes, and stimulation sequences
   - Starts data recording
   - Run this AFTER starting the C++ processor

2. **process_raw_data.cpp**
   - C++ script for raw data stream processing
   - Monitors voltage amplitudes on specified channel
   - Sends stimulation when threshold exceeded
   - Good for custom detection logic

3. **process_spike_events.cpp**
   - C++ script for filtered spike event processing
   - Only processes detected spikes (more efficient)
   - Recommended for most experiments
   - Lower false positive rate

### Build Files

4. **CMakeLists.txt**
   - CMake configuration for building C++ code
   - Modern, cross-platform build system
   - Usage: `cmake .. && make`

5. **Makefile**
   - GNU Make configuration (alternative to CMake)
   - Simpler for quick builds
   - Usage: `make`

### Configuration Files

6. **config_template.py**
   - Centralized configuration template
   - All parameters documented
   - Helper functions for conversions
   - Validation utilities

7. **example_custom_setup.py**
   - Example of using config_template
   - Shows how to override defaults
   - Demonstrates best practices

### Utility Files

8. **quickstart.sh**
   - Interactive menu-driven setup script
   - Helps with building and running
   - Good for beginners
   - Usage: `./quickstart.sh`

9. **README.md**
   - Comprehensive documentation
   - Step-by-step instructions
   - Troubleshooting guide
   - Advanced customization examples

## Quick Start

### Minimal Setup (3 steps)

```bash
# 1. Build
make

# 2. Run C++ processor (in terminal 1)
./build/process_spike_events 13248

# 3. Run Python setup (in terminal 2)
python3 closedLoopSetup.py
```

### Interactive Setup

```bash
./quickstart.sh
# Follow the menu options
```

## File Dependencies

```
Python Setup Scripts
├── closedLoopSetup.py (standalone)
├── example_custom_setup.py
│   └── requires: config_template.py
└── config_template.py (standalone)

C++ Processing Scripts
├── process_raw_data.cpp
│   └── requires: maxlab/maxlab.h
└── process_spike_events.cpp
    └── requires: maxlab/maxlab.h

Build Systems
├── CMakeLists.txt
└── Makefile
```

## Typical Workflow

### 1. First Time Setup
```bash
# Clone or navigate to closed_loop directory
cd closed_loop

# Build executables
make
# or
./quickstart.sh  # Choose option 1 or 2
```

### 2. Configure Experiment
```python
# Edit config_template.py or create your own config
# Set electrodes, amplitudes, recording parameters
```

### 3. Run Experiment
```bash
# Terminal 1: Start C++ processor
./build/process_spike_events YOUR_CHANNEL_ID

# Terminal 2: Start Python setup
python3 closedLoopSetup.py
# or with custom config:
python3 example_custom_setup.py
```

### 4. Monitor
- Watch Terminal 1 for spike detection
- Watch Terminal 2 for system status
- Data saves automatically to specified location

### 5. Stop
- Ctrl+C in both terminals
- Data file closed automatically

## File Sizes

| File | Lines | Purpose |
|------|-------|---------|
| closedLoopSetup.py | ~170 | Main setup |
| process_raw_data.cpp | ~120 | Raw processing |
| process_spike_events.cpp | ~130 | Spike processing |
| config_template.py | ~230 | Configuration |
| README.md | ~450 | Documentation |
| CMakeLists.txt | ~30 | CMake build |
| Makefile | ~80 | Make build |
| quickstart.sh | ~180 | Interactive setup |
| example_custom_setup.py | ~140 | Usage example |

## Customization Points

### Python Scripts
- Electrode selection
- Stimulation parameters
- Recording configuration
- Detection thresholds

### C++ Scripts
- Detection conditions
- Blanking period
- Amplitude thresholds
- Response timing

### Build Configuration
- Compiler flags
- MaxLab library path
- Optimization levels

## Best Practices

1. **Always build first**: Run `make` or `cmake` before your experiment
2. **Start C++ first**: The processor must be running before Python setup
3. **Check paths**: Verify MaxLab installation location
4. **Test configuration**: Run `python3 config_template.py` to validate
5. **Monitor output**: Watch both terminals for errors
6. **Keep backups**: Save configuration files with your data

## Troubleshooting

### Build Issues
```bash
# Check MaxLab installation
ls ~/MaxLab/share/maxlab_lib

# Clean and rebuild
make clean && make
```

### Runtime Issues
```bash
# Check mxwserver is running
# Verify device connection
# Review terminal output for errors
```

### Python Issues
```bash
# Check maxlab package
python3 -c "import maxlab; print(maxlab.__version__)"

# Verify paths in config
python3 config_template.py
```

## Version Information

Created: January 26, 2026
Based on: MaxLab API v1.0.0
Compatible with:
- MaxOne devices
- MaxTwo devices
- Python 3.6+
- C++20 (GNU)

## Support Resources

- Documentation: README.md
- Examples: example_custom_setup.py
- Configuration help: config_template.py
- Quick start: quickstart.sh

## License

Follow MaxLab API licensing terms for your installation.

---

For questions or issues, refer to the comprehensive README.md file.
