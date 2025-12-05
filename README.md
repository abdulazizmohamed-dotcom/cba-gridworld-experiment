Coherence-Based Alignment (CBA) – Gridworld Experiment

This repository contains the reference implementation of the Coherence-Based Alignment (CBA) toy experiment, demonstrating how coherence-regularization reduces reward-loop failures in reinforcement learning.

Overview

CBA introduces a simple but powerful idea:
Agents should not only maximize reward — they should maintain internal policy coherence over time.

This repository includes:

cba_gridworld.py — full environment, Q-learning agent, and CBA regularizer

A technical note summarizing goals, methodology, and findings (coming soon)

Key Result

CBA reduces reward-loop entrapment by ~66%, while maintaining high global goal success.

Metric	Baseline Q-Learning	CBA (λ=0.5)
Average Loop-Time Fraction	0.166	0.056
Reduction	—	66%
How to Run
python cba_gridworld.py

Citation
Abdi, Abdulaziz (2025). Coherence-Based Alignment:  
Toy Experiment Demonstrating Coherence-Regularization  
in Reinforcement Learning. GitHub repository.

License

MIT License (to be added)
