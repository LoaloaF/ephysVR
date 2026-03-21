# Makefile for Closed-Loop Experiment
# Alternative to CMake for simpler builds

# Compiler settings
CXX = g++
CXXFLAGS = -std=gnu++20 -Wall -Wextra -O2

# MaxLab library paths
# Adjust MAXLAB_DIR if your installation is in a different location
MAXLAB_DIR = $(HOME)/MaxLab/share/maxlab_lib
INCLUDE_DIR = $(MAXLAB_DIR)/maxlab/include
LIB_DIR = $(MAXLAB_DIR)/maxlab/lib

# Linker flags
LDFLAGS = -L$(LIB_DIR)
LIBS = -lmaxlab

# Build directory
BUILD_DIR = build

# Targets
TARGETS = $(BUILD_DIR)/process_raw_data $(BUILD_DIR)/process_spike_events

# Default target
all: $(BUILD_DIR) $(TARGETS)
	@echo "Build complete!"
	@echo "Executables are in the $(BUILD_DIR) directory"

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Build process_raw_data
$(BUILD_DIR)/process_raw_data: process_raw_data.cpp
	@echo "Compiling process_raw_data..."
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) $< -o $@ $(LDFLAGS) $(LIBS)

# Build process_spike_events
$(BUILD_DIR)/process_spike_events: process_spike_events.cpp
	@echo "Compiling process_spike_events..."
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) $< -o $@ $(LDFLAGS) $(LIBS)

# Clean build artifacts
clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILD_DIR)

# Rebuild everything
rebuild: clean all

# Install (copy to a more accessible location)
install: all
	@echo "Installing to /usr/local/bin (requires sudo)..."
	@sudo cp $(TARGETS) /usr/local/bin/

# Help message
help:
	@echo "Closed-Loop Experiment Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build both executables (default)"
	@echo "  clean     - Remove build directory"
	@echo "  rebuild   - Clean and rebuild"
	@echo "  install   - Install executables to /usr/local/bin"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  MAXLAB_DIR = $(MAXLAB_DIR)"
	@echo "  CXX        = $(CXX)"
	@echo ""
	@echo "To change MaxLab directory:"
	@echo "  make MAXLAB_DIR=/custom/path"

# Phony targets
.PHONY: all clean rebuild install help
