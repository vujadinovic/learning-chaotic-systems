# Double Pendulum Dynamics and Neural Network Prediction

## Project Description
This repository contains a study on the chaotic dynamics of a double mathematical pendulum and the application of neural networks to predict its future states. The project bridges classical Lagrangian mechanics with deep learning by using high-fidelity numerical simulations as a data source for model training.

## Physical Model
The system consists of two masses $m_1$ and $m_2$ connected by two massless rods of lengths $l_1$ and $l_2$. The motion is governed by a set of coupled, non-linear ordinary differential equations (ODEs). 



Unlike the simple pendulum, the double pendulum exhibits deterministic chaos, making long-term prediction highly sensitive to initial conditions.

## Implementation Details

### Numerical Integration
To ensure data integrity and physical consistency, the project utilizes a **variational integrator**. This approach is superior to simpler methods (like Euler or RK45) as it maintains energy conservation over longer simulation intervals, providing a reliable dataset for the neural network.

### Data Preprocessing
TO DO

## Repository Structure
* `simulation.py`: Contains the physical model and integration logic.
* `model.py`: Definition of the neural network architecture.
* `train.py`: Script for training the model on generated datasets.
* `visualize.py`: Tools for animating the pendulum and plotting loss metrics.

## Requirements
* Python 3.8+
* NumPy
* SciPy
* Matplotlib
* PyTorch

## Usage
1. Generate the dataset and visualize the chaotic trajectory:
   ```bash
   python simulation.py
