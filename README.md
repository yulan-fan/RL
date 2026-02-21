# Unit 1-3
## 🚀 Future Improvements & Discussion

The current agent performs well, but the process of finding the right parameters (learning rate, gamma, and epsilon decay) often feels like "manual guessing." 

To improve the training efficiency and final performance:
* **Automated Hyperparameter Tuning:** It would be highly beneficial to optimize the hyperparameters using a framework like **Optuna**. This would allow for a systematic search (e.g., Bayesian Optimization) rather than manual trial-and-error.
* **Compute Constraints:** While Optuna is powerful, it is computationally expensive as it requires running multiple training trials. This approach should be implemented if computing resources and time allow.
* **Reward Shaping:** For more complex environments (like the 8x8 Frozen Lake), introducing intermediate rewards could help guide the agent faster than relying on a sparse +1.0 goal reward.

# Training and Computational Strategy

To handle the high-dimensional state space of Atari *Space Invaders* and manage a **larger sample size**, training is conducted via the **VS Code Colab Extension**.

* **Custom Neural Architecture:** Instead of using a generic "ready-made" network, the CNN is constructed within a **custom helper function/class**. This provides the flexibility to fine-tune the feature extraction layers and output heads specifically for the Atari frame-stacking logic.
* **Hardware Acceleration:** Utilizes Google's **NVIDIA T4 GPU** to accelerate the convolutional forward and backward passes.
* **Memory Management:** Implemented a `ReplayBuffer` with a capacity of **30,000 transitions**, optimized for the ~12GB RAM limit of the Colab Free Tier.
* **Batch Processing:** A `batch_size` of **64** is used to provide a more stable gradient signal for the Adam optimizer.


# Unit 4: REINFORCE Policy Gradient Analysis

## 1. CartPole-v1: The Baseline for Convergence
The CartPole environment serves as a benchmark for verifying the core logic of the REINFORCE algorithm.

### Training Observations
* **Exploration Stage (Episodes 0-100)**: The agent exhibits low scores as it randomly samples the state-action space to initialize its neural network weights.
* **Learning Slope (Episodes 100-300)**: A steep climb in scores indicates the agent has identified high-reward trajectories, successfully shifting action probabilities toward "balancing" behaviors.
* **The Variance Problem**: Sharp vertical drops (dips) are visible even late in training. This is a hallmark of the REINFORCE algorithm: because it relies on full-trajectory returns ($G_t$), a single random "bad" action can lead to a large negative gradient that temporarily destabilizes the policy.
* **Full Convergence**: The agent eventually reaches a stable plateau at the environment maximum of **500**, signifying a perfected policy

## 2. LunarLander

### Key Technical Insights

* **Continuous Control Logic**: While this version uses discrete actions, policy-based methods are the primary choice for continuous environments (like autonomous driving) because they learn a probability distribution (mean $\mu$ and variance $\sigma$) rather than just calculating Q-values.
* **Dynamic Constraints**: The agent must navigate **Dynamic Constraints**—variables like fuel consumption and descent velocity boundaries that change in importance depending on the lander's current state.
* **The "Noisy" Ascent**: The LunarLander curve is significantly more rugged than CartPole. This reflects the agent's struggle to balance landing precision with fuel efficiency under varying initial conditions.
  

## 3. Reflection: What Defines "Success"?
When analyzing these plots, it is important to consider which metric defines a superior agent:

1.  **Fast Convergence**: Reaching the score ceiling in the fewest episodes.
2.  **Robustness**: Maintaining a stable line without "dips" once the task is learned.
3.  **Recoverability**: The ability to quickly re-learn and return to high performance after a stochastic failure (rebounding from a "dip").

**The Insight:** In simple tasks, we value speed. However, for real-world applications with high risk and **Dynamic Constraints**, **Robustness** and **Recoverability** are far more critical than initial learning speed.
