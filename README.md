## 🛠 Installation & Dependencies

To run this agent, you need to set up a virtual environment with the dependencies used in the [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit1/introduction).

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yulan-fan/lunar-lander-ppo.git

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
  
   
