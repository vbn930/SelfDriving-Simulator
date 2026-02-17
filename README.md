# End-to-End Learning for Self-Driving Cars in Unity

This project is a reproduction of the **NVIDIA End-to-End Learning for Self-Driving Cars** paper, implemented using **PyTorch** and tested in a **Unity 3D simulation environment**.

## Project Overview
The goal of this project is to train a Convolutional Neural Network (CNN) to map raw pixels from a front-facing camera directly to steering commands. The model takes image data and vehicle speed as inputs and predicts the steering angle and throttle value.

### Reference Paper
- **Title**: End to End Learning for Self-Driving Cars
- **Authors**: Mariusz Bojarski et al. (NVIDIA)
- **Link**: [arXiv:1604.07316](https://arxiv.org/abs/1604.07316)

---

## System Architecture

### Model Architecture (Modified Dave2)
I utilized a modified version of the **NVIDIA Dave2** architecture. The original architecture is enhanced to include **vehicle speed** as an additional input feature, allowing the model to make more informed decisions based on the current state of the vehicle.

1.  **Visual Encoder (CNN)**:
    -   Input: 200x66 RGB Image (Resized & Normalized)
    -   3 Convolutional Layers (5x5 kernel, stride 2)
    -   2 Convolutional Layers (3x3 kernel, non-strided)
    -   Activation: ELU (Exponential Linear Unit)
    -   Output: Flattened feature vector (1152 dimensions)

2.  **Speed Encoder (MLP)**:
    -   Input: Vehicle Speed (Normalized)
    -   1 Fully Connected Layer (16 units) with ELU activation

3.  **Control Policy (Fusion & Output)**:
    -   The visual features and speed features are concatenated.
    -   4 Fully Connected Layers (100 -> 50 -> 10 -> 2)
    -   **Outputs**:
        -   `Steering Angle` (float)
        -   `Throttle` (float)

---

## Results
The model was trained on data collected from manual driving sessions.
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (Learning Rate: 1e-4)
- **Best Model**: `models/best_model_v3.pth`

The trained model successfully navigates the track, adjusting steering based on road curvature and managing speed appropriately.

## Demonstration
[![Autonomous Driving Demo](https://img.youtube.com/vi/v07TapYTM1Y/0.jpg)](https://www.youtube.com/watch?v=v07TapYTM1Y)
*Click the image to watch the demo video.*
