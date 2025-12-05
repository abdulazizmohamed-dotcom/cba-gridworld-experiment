**Coherence-Based Alignment (CBA): Gridworld Experiment**

*Reducing reward-loop entrapment through coherence regularization*

**Overview**

This repository contains the official implementation of the Coherence-Based Alignment (CBA) toy experiment.

*CBA introduces a simple idea:*

Agents should not only optimize reward — they should maintain internal policy coherence over time.

The goal of this experiment is to test whether a simple coherence penalty can reduce reward-hacking, specifically reward-loop entrapment, in a reinforcement learning setting.

The experiment uses a 10×10 gridworld containing a deceptive, repeating reward loop that tempts standard Q-learning into suboptimal behavior.

**Key Result**

Across 5 random seeds (5000 episodes each):

Metric	Baseline Q-Learning	CBA (λ = 0.5)

Average Loop-Time Fraction	0.166	0.056

Reduction	—	≈ 66%

Goal Reaching Rate	99.85%	99.88%

CBA dramatically reduces loop entrapment while preserving goal performance.

*Technical Note*

A full explanation of the environment, algorithm, equations, and results is available here:

[CBA Technical Note (PDF)](https://github.com/abdulazizmohamed-dotcom/cba-gridworld-experiment/blob/main/cba_technical_note.pdf)

**Algorithm Summary**

**1. Baseline: Standard Q-Learning**

Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') − Q(s, a) ]


*Hyperparameters:*

α = 0.1

γ = 0.95

ε-decay: 1.0 → 0.01 over 5000 episodes

Max steps per episode: 1000

5000 episodes × 5 random seeds

**2. CBA-Regularized Update**

CBA introduces two instantaneous penalties:

Loop Penalty (Lₜ)
−1 if the agent enters the loop region, else 0

Incoherence Penalty (Iₜ)
−1 if the action is non-greedy under current Q(s), else 0

*TD-error:*

TD_error = r + γ max_a' Q(s', a') − Q(s, a)

*Regularized update:*

Q(s, a) ← Q(s, a) + α [ TD_error + λ (Lₜ + Iₜ) ]

λ = 0.5 in this experiment.

**How to Run**

Clone the repository:
git [clone](https://github.com/abdulazizmohamed-dotcom/cba-gridworld-experiment.git)


*Run the experiment:*
[python cba_gridworld.py](https://github.com/abdulazizmohamed-dotcom/cba-gridworld-experiment/blob/main/cba_experiment.py)


The script prints:

- Goal hits
- Average returns
- Loop-time fractions

Comparison summary between baseline and CBA

**Repository Structure**

- cba_gridworld.py           # Full implementation (environment + Q-learning + CBA)
- README.md                  # Project documentation
- CBA_Technical_Note.pdf     # Detailed technical note
- (more files as needed)

**Citation**

*If you use this work, please cite:*

Abdi, Abdulaziz (2025). Coherence-Based Alignment (CBA):

A Toy Experiment on Coherence-Regularized Reinforcement Learning.

GitHub repository.

**Limitations**

This experiment is a proof-of-concept in a simple gridworld.

*Further testing is needed in:*
- continuous control
- larger deceptive reward structures
- partial observability
- neural function approximators

**Future Directions**
- Scaling CBA to Deep Q-Networks (DQN)
- Testing on multiple forms of reward hacking
- Comparing coherence penalties to entropy regularization
- Investigating theoretical properties of coherence-based alignment
