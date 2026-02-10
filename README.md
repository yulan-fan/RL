## 🛠 Installation & Dependencies

To run this agent, you need to set up a virtual environment with the dependencies used in the [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit1/introduction).

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yulan-fan/ML.git

2. **Install the course dependencies: Follow the instructions in the Unit 1 notebook or install the core libraries:**
   ```bash
   pip install stable-baselines3[extra]
   pip install gymnasium[box2d]
   pip install huggingface_sb3
   
## 💻 System Requirements
This script was developed and tested on **Ubuntu 20.04**. 

Note: Due to specific dependencies like `swig` and the `Box2D` engine, you might encounter issues on other operating systems or different Ubuntu distributions without proper library mapping.

## 🚀 Future Improvements & Discussion

The current agent performs well, but the process of finding the right parameters (learning rate, gamma, and epsilon decay) often feels like "manual guessing." 

To improve the training efficiency and final performance:
* **Automated Hyperparameter Tuning:** It would be highly beneficial to optimize the hyperparameters using a framework like **Optuna**. This would allow for a systematic search (e.g., Bayesian Optimization) rather than manual trial-and-error.
* **Compute Constraints:** While Optuna is powerful, it is computationally expensive as it requires running multiple training trials. This approach should be implemented if computing resources and time allow.
* **Reward Shaping:** For more complex environments (like the 8x8 Frozen Lake), introducing intermediate rewards could help guide the agent faster than relying on a sparse +1.0 goal reward.

## Training and Computational Strategy

To handle the high-dimensional state space of Atari *Space Invaders* and manage a **larger sample size**, training is conducted via the **VS Code Colab Extension**.

* **Custom Neural Architecture:** Instead of using a generic "ready-made" network, the CNN is constructed within a **custom helper function/class**. This provides the flexibility to fine-tune the feature extraction layers and output heads specifically for the Atari frame-stacking logic.
* **Hardware Acceleration:** Utilizes Google's **NVIDIA T4 GPU** to accelerate the convolutional forward and backward passes.
* **Memory Management:** Implemented a `ReplayBuffer` with a capacity of **30,000 transitions**, optimized for the ~12GB RAM limit of the Colab Free Tier.
* **Batch Processing:** A `batch_size` of **64** is used to provide a more stable gradient signal for the Adam optimizer.
  
   
