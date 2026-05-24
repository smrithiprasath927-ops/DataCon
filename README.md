# DataCon
Presentation link - https://canva.link/i6hh28vpn3aruol
Web app prototype of an AI-based data corruption and reconstruction system using CNNs and mathematical techniques..

The project combines:
- Mathematics
- Convolutional Neural Networks (CNNs)
- Data corruption algorithms
- Pattern reconstruction
- AI-based denoising systems


## Problem Statement

Digital data can become corrupted because of:
- Transmission errors
- Malware or viruses
- Storage failures
- Noise interference
- Incomplete information

We often turn to antiviruses to prevent such situations, or try to retrieve copies of the original when data is lost.
This project investigates whether AI can learn hidden structures in damaged data and reconstruct the original information from the corrupted data.

## Project Objective

The objective of DataCon is to:
1. Intentionally corrupt structured text/data
2. Train a CNN model on corrupted patterns
3. Reconstruct the original information
4. Measure reconstruction accuracy


## Technologies Used

- Python
- PyTorch
- NumPy
- OpenCV
- Streamlit
- CNN Architecture


## How It Works

### Step 1 — Data Corruption
Original text/data is mathematically degraded using:
- Random masking
- Noise addition

### Step 2 — CNN Training
The CNN model learns relationships between:
- Corrupted inputs
- Original outputs

### Step 3 — Reconstruction
The trained model predicts and reconstructs the damaged information.

## Features

- Interactive corruption simulator
- Customizable diffusion, noise and training levels
- CNN-based reconstruction
- Real-time visual comparison
- Streamlit interface
- Accuracy visualization

---

## Sample Output

**Original** - hello
**Reconstructed** - hello


## CNN Architecture

The project uses a lightweight Convolutional Neural Network designed for:
- Fast training
- Pattern recognition
- Noise reduction
- Character reconstruction


## Future Improvements

- Larger datasets
- Audio denoising
- Advanced diffusion-based corruption for cybersecurity applications



## Team

Developed by:
- Smrithi Prasath
- Mohith Gowda


## Installation

Requirements:

python
torch
numpy
opencv-python
streamlit
matplotlib
pillow
