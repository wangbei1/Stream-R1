#!/bin/bash

echo "======================================================"
echo "Starting model download..."
echo "======================================================"

echo "Downloading Wan2.1-T2V-1.3B..."
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir checkpoints/Wan2.1-T2V-1.3B 

echo "Downloading Wan2.1-T2V-14B..."
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir checkpoints/Wan2.1-T2V-14B 

echo "Downloading VideoReward..."
huggingface-cli download KlingTeam/VideoReward --local-dir checkpoints/Videoreward 

echo "Downloading ODE Initialization..."
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .

# Stream-R1 trained weights will be released here once uploaded:
#   huggingface-cli download <Stream-R1-HF-repo> --local-dir checkpoints/Stream-R1-T2V-1.3B

echo "======================================================"
echo "Finished downloading models!"
ls -R checkpoints
echo "======================================================"