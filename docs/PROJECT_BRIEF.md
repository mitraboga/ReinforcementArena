# Reinforcement Arena Project Brief

## Positioning

Reinforcement Arena is a reinforcement learning lab for business operations and finance optimization. It translates the CS50AI Nim Q-learning assignment into a portfolio-grade system focused on sequential decision-making in inventory management, cash allocation, and dynamic pricing.

## Business Problem

Business decisions are sequential and uncertain. Ordering too little creates stockouts and emergency procurement costs. Ordering too much creates holding costs and ties up cash. Cash allocation decisions affect debt, liquidity, and growth. Pricing decisions affect both demand and margin. Reinforcement learning is a natural fit because each action changes future state and future opportunity.

## Technical Scope

- Custom inventory environment with stochastic seasonal demand
- Custom cash-flow environment for finance allocation
- Custom pricing environment with elasticity-driven demand
- Supplier lead-time pipeline and pending-order state
- Tabular Q-learning and SARSA agents
- PyTorch DQN agent with replay memory and target network
- Random, fixed-order, and greedy reorder baselines
- Business-rule baselines for finance and pricing
- Reproducible YAML configuration
- Evaluation metrics aligned to business outcomes
- Streamlit analytics dashboard
- Unit tests and CI workflow

## Core Metrics

- Average reward
- Final cash balance
- Stockout rate
- Service level
- Units sold
- Emergency procurement cost
- Holding cost
- Ending inventory
- Net worth
- Debt balance
- Revenue
- Average price

## Portfolio Narrative

This project connects business and technical skills: operations management, finance-aware reward modeling, Python engineering, reinforcement learning, and analytics dashboarding. It demonstrates the ability to turn an academic RL assignment into a practical decision-optimization system.
