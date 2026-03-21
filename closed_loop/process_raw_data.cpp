/**
 * @file process_raw_data.cpp
 * @brief Closed-loop experiment processing raw data stream from MaxOne/MaxTwo
 * 
 * This script monitors the raw data stream from the device and sends a
 * stimulation sequence when a condition is met (e.g., spike detected on
 * a specific channel).
 * 
 * Usage: ./process_raw_data [detection_channel]
 * Example: ./process_raw_data 13248
 * 
 * The script will:
 * 1. Open a connection to the raw data stream
 * 2. Continuously receive frames
 * 3. Check if amplitude on detection_channel exceeds threshold
 * 4. Send stimulation sequence when condition is met
 * 5. Implement blanking period to avoid artifacts
 */

#include <stdlib.h>
#include <stdio.h>
#include "maxlab/maxlab.h"

int main(int argc, char * argv[])
{
    // Check command line arguments
    if (argc < 2)
    {
        fprintf(stderr, "Call with: %s [detection_channel]\n", argv[0]);
        fprintf(stderr, "Example: %s 13248\n", argv[0]);
        exit(1);
    }
    const int detection_channel = atoi(argv[1]);
    
    printf("Starting closed-loop experiment with raw data stream...\n");
    printf("Detection channel: %d\n", detection_channel);
    printf("Threshold: 500 (arbitrary units)\n");
    printf("Blanking period: 8000 frames (~0.4 seconds at 20kHz)\n\n");

    // Open raw data stream
    printf("Opening raw data stream...\n");
    maxlab::verifyStatus(maxlab::DataStreamerRaw_open());
    printf("Raw data stream opened successfully!\n");
    printf("Listening for spikes...\n\n");

    uint64_t blanking = 0;
    uint64_t frame_count = 0;
    uint64_t stim_count = 0;
    
    while (true)
    {
        // Receive next frame
        maxlab::RawFrameData frameData;
        maxlab::Status status = maxlab::DataStreamerRaw_receiveNextFrame(&frameData);
        
        // Check if frame is valid
        if (status == maxlab::Status::MAXLAB_NO_FRAME || frameData.frameInfo.corrupted)
            continue;
        
        frame_count++;
        
        // Handle blanking period (post-stimulation recovery)
        if (blanking > 0)
        {
            blanking--;
            if(blanking != 0)
                continue;
            else
                printf("Blanking period complete. Resuming detection.\n");
        }

        // Check detection condition: amplitude exceeds threshold
        if (frameData.amplitudes[detection_channel] > 500)
        {
            stim_count++;
            printf("Spike detected! Frame: %lu, Amplitude: %f\n", 
                   frame_count, frameData.amplitudes[detection_channel]);
            printf("Sending closed-loop stimulation sequence #%lu...\n", stim_count);
            
            // Send the stimulation sequence
            maxlab::Status send_status = maxlab::sendSequence("closed_loop_sequence");
            
            if (send_status == maxlab::Status::MAXLAB_OK)
            {
                printf("Stimulation sent successfully!\n");
            }
            else
            {
                fprintf(stderr, "Error sending stimulation sequence!\n");
                maxlab::Response response = maxlab::sendRaw("get_errors");
                fprintf(stderr, "Error details: %s\n", response.content);
                maxlab::freeResponse(&response);
            }
            
            // Start blanking period to avoid stimulation artifacts
            blanking = 8000;
            printf("Starting blanking period...\n\n");
        }
        
        // Optional: Print status every 100k frames (~5 seconds at 20kHz)
        if (frame_count % 100000 == 0)
        {
            printf("Status: %lu frames processed, %lu stimulations sent\n", 
                   frame_count, stim_count);
        }
    }
    
    // Close raw data stream (unreachable in infinite loop, but good practice)
    printf("Closing raw data stream...\n");
    maxlab::verifyStatus(maxlab::DataStreamerRaw_close());
    
    return 0;
}
