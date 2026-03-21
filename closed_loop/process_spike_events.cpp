/**
 * @file process_spike_events.cpp
 * @brief Closed-loop experiment processing filtered spike events from MaxOne/MaxTwo
 * 
 * This script monitors the filtered data stream (spike events) from the device
 * and sends a stimulation sequence when a spike is detected on a specific channel.
 * 
 * Unlike the raw data stream, this version only processes actual detected spike
 * events, reducing computational load and false positives.
 * 
 * Usage: ./process_spike_events [detection_channel]
 * Example: ./process_spike_events 13248
 * 
 * The script will:
 * 1. Open a connection to the filtered data stream
 * 2. Continuously receive frames with spike events
 * 3. Check if any spike occurred on detection_channel
 * 4. Send stimulation sequence when spike is detected
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
    
    printf("Starting closed-loop experiment with filtered spike events...\n");
    printf("Detection channel: %d\n", detection_channel);
    printf("Blanking period: 8000 frames (~0.4 seconds at 20kHz)\n");
    printf("Filter type: IIR\n\n");

    // Open filtered data stream with IIR filter
    printf("Opening filtered data stream...\n");
    maxlab::verifyStatus(maxlab::DataStreamerFiltered_open(maxlab::FilterType::IIR));
    printf("Filtered data stream opened successfully!\n");
    printf("Listening for spike events...\n\n");

    uint64_t blanking = 0;
    uint64_t frame_count = 0;
    uint64_t spike_count = 0;
    uint64_t stim_count = 0;
    maxlab::FilteredFrameData frameData;
    
    while (true)
    {
        // Receive next frame
        maxlab::Status status = maxlab::DataStreamerFiltered_receiveNextFrame(&frameData);
        
        // Check if frame is available
        if(status == maxlab::Status::MAXLAB_NO_FRAME)
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

        // Process all spike events in this frame
        for (uint64_t i = 0; i < frameData.spikeCount; ++i)
        {
            const maxlab::SpikeEvent &spike = frameData.spikeEvents[i];
            spike_count++;
            
            // Check if spike is on our detection channel
            if (spike.channel == detection_channel)
            {
                stim_count++;
                printf("Target spike detected! Channel: %d, Frame: %lu\n", 
                       spike.channel, frame_count);
                printf("Total spikes seen: %lu\n", 
                       spike_count);
                printf("Sending closed-loop stimulation sequence #%lu...\n", stim_count);
                
                // Send the stimulation sequence
                const maxlab::Status send_status = maxlab::sendSequence("closed_loop_sequence");

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
                
                // Break to avoid multiple stimulations from same frame
                break;
            }
        }
        
        // Optional: Print status every 100k frames (~5 seconds at 20kHz)
        if (frame_count % 100000 == 0)
        {
            printf("Status: %lu frames processed, %lu total spikes, %lu stimulations sent\n", 
                   frame_count, spike_count, stim_count);
        }
    }
    
    // Close filtered data stream (unreachable in infinite loop, but good practice)
    printf("Closing filtered data stream...\n");
    maxlab::verifyStatus(maxlab::DataStreamerFiltered_close());
    
    return 0;
}
