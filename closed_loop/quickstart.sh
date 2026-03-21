#!/bin/bash

# Quick Start Script for Closed-Loop Experiment
# This script helps set up and run the closed-loop experiment

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Closed-Loop Experiment Quick Start${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if MaxLab is installed
MAXLAB_DIR="$HOME/MaxLab/share/maxlab_lib"
if [ ! -d "$MAXLAB_DIR" ]; then
    print_warning "MaxLab directory not found at: $MAXLAB_DIR"
    
    # Check if zip file exists
    MAXLAB_ZIP=$(ls "$HOME/MaxLab/share/libmaxlab-"*.zip 2>/dev/null | head -n 1)
    if [ ! -z "$MAXLAB_ZIP" ]; then
        print_info "Found MaxLab zip file: $MAXLAB_ZIP"
        read -p "Extract it now? [Y/n]: " EXTRACT
        if [[ "$EXTRACT" =~ ^[Yy]?$ ]]; then
            print_info "Extracting MaxLab library..."
            unzip -q "$MAXLAB_ZIP" -d "$HOME/MaxLab/share/"
            if [ -d "$HOME/MaxLab/share/maxlab_lib" ]; then
                MAXLAB_DIR="$HOME/MaxLab/share/maxlab_lib"
                print_success "Extraction complete!"
            else
                print_error "Extraction failed or directory structure unexpected"
                exit 1
            fi
        else
            print_warning "Please extract manually: unzip $MAXLAB_ZIP -d ~/MaxLab/share/"
            exit 1
        fi
    else
        read -p "Enter custom MaxLab library path (or press Enter to exit): " CUSTOM_PATH
        if [ ! -z "$CUSTOM_PATH" ]; then
            MAXLAB_DIR="$CUSTOM_PATH"
        else
            print_error "Cannot continue without MaxLab library"
            exit 1
        fi
    fi
fi

print_info "Using MaxLab directory: $MAXLAB_DIR"
echo ""

# Menu
echo "What would you like to do?"
echo ""
echo "1) Build C++ executables (using CMake)"
echo "2) Build C++ executables (using Make)"
echo "3) Run raw data processor"
echo "4) Run spike event processor"
echo "5) Run Python setup script"
echo "6) Complete setup (build + instructions)"
echo "7) Clean build files"
echo "8) Exit"
echo ""
read -p "Enter choice [1-8]: " choice

case $choice in
    1)
        print_info "Building with CMake..."
        mkdir -p build
        cd build
        cmake .. -DMAXLAB_DIR="$MAXLAB_DIR"
        make
        cd ..
        print_success "Build complete! Executables are in build/ directory"
        ;;
    
    2)
        print_info "Building with Make..."
        make MAXLAB_DIR="$MAXLAB_DIR"
        print_success "Build complete! Executables are in build/ directory"
        ;;
    
    3)
        if [ ! -f "build/process_raw_data" ]; then
            print_error "Executable not found. Please build first (option 1 or 2)"
            exit 1
        fi
        read -p "Enter detection channel electrode ID (e.g., 13248): " CHANNEL
        print_info "Starting raw data processor on channel $CHANNEL..."
        print_warning "Press Ctrl+C to stop"
        ./build/process_raw_data "$CHANNEL"
        ;;
    
    4)
        if [ ! -f "build/process_spike_events" ]; then
            print_error "Executable not found. Please build first (option 1 or 2)"
            exit 1
        fi
        read -p "Enter detection channel electrode ID (e.g., 13248): " CHANNEL
        print_info "Starting spike event processor on channel $CHANNEL..."
        print_warning "Press Ctrl+C to stop"
        ./build/process_spike_events "$CHANNEL"
        ;;
    
    5)
        print_info "Running Python setup script..."
        print_warning "Make sure C++ processor is already running!"
        sleep 2
        python3 closedLoopSetup.py
        ;;
    
    6)
        print_info "Complete setup process"
        echo ""
        
        # Build
        print_info "Step 1: Building C++ executables..."
        if [ -f "Makefile" ]; then
            make MAXLAB_DIR="$MAXLAB_DIR"
        else
            mkdir -p build
            cd build
            cmake .. -DMAXLAB_DIR="$MAXLAB_DIR"
            make
            cd ..
        fi
        print_success "Build complete!"
        echo ""
        
        # Instructions
        print_info "Step 2: Ready to run experiment"
        echo ""
        echo -e "${GREEN}Next steps:${NC}"
        echo ""
        echo "1. In this terminal, run the C++ processor:"
        echo -e "   ${YELLOW}./build/process_spike_events 13248${NC}"
        echo "   (replace 13248 with your detection channel)"
        echo ""
        echo "2. In a NEW terminal, run the Python setup:"
        echo -e "   ${YELLOW}python3 closedLoopSetup.py${NC}"
        echo ""
        echo "3. Monitor both terminals for output"
        echo ""
        echo "4. Press Ctrl+C in both terminals when done"
        echo ""
        ;;
    
    7)
        print_info "Cleaning build files..."
        if [ -f "Makefile" ]; then
            make clean
        else
            rm -rf build
        fi
        print_success "Clean complete!"
        ;;
    
    8)
        print_info "Exiting..."
        exit 0
        ;;
    
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo ""
print_success "Done!"
