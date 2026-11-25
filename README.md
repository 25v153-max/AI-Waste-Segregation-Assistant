# EcoAssist AI

An edge-based multimodal waste classification and intelligent waste segregation system developed for the Arm AI Hackathon.

## Features
- Image classification using MobileNetV3-Small.
- Audio classification using a custom CNN.
- Python-based multimodal sensor fusion.
- Llama-based natural language explanations.
- Designed for Raspberry Pi 5 edge deployment.

## Waste Categories
Plastic, paper, cardboard, glass, metal and organic waste.

## Hardware
- Raspberry Pi 5 (8 GB)
- Logitech C270 webcam and microphone

## Software
https://ecoassist-ai.ai.studio/

## Technologies
Python, PyTorch, TorchVision, Librosa and Llama.

## Project Status
PyTorch model loading on Raspberry Pi has been tested. Camera integration and ExecuTorch deployment are ongoing.

The current `.pth` checkpoints are PyTorch models, not ExecuTorch `.pte` models.
