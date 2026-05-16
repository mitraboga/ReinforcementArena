# Reinforcement Arena

### Self-learning AI agents for inventory, cash flow, and business operations optimization

Reinforcement Arena is a production-style reinforcement learning lab for sequential business decisions. It trains agents to optimize inventory, cash allocation, and pricing decisions while comparing learned policies against practical business baselines.

This project extends the CS50AI Nim reinforcement learning idea into an operations and finance portfolio project.

## CS50AI Concepts Applied

This project is built directly on the reinforcement learning ideas from CS50AI's Nim project, then extends them into business optimization environments.

### State

In CS50AI Nim, the state is the current pile configuration, such as:

```python
[1, 3, 5, 7]
```

In Reinforcement Arena, each environment defines a business state:

```python
# Inventory
(inventory_bucket, cash_bucket, demand_forecast_bucket, pipeline_bucket)

# Cash Flow
(cash_bucket, debt_bucket, emergency_fund_bucket, month)

# Pricing
(inventory_bucket, demand_bucket, price_bucket, day)
```

The idea is the same: the agent observes the current situation before choosing an action.

### Actions

In Nim, an action means choosing a pile and removing objects:

```python
(pile_index, objects_to_remove)
```

In this project, actions are business decisions:

```python
# Inventory
reorder_quantity = 0, 10, 20, 30, 40, 50, 60

# Cash Flow
action = "pay_debt", "invest_marketing", "save_cash", "hold_cash"

# Pricing
price = 18, 22, 25, 28, 32, 36
```

### Rewards

In Nim, the reward is simple: winning is good and losing is bad.

In Reinforcement Arena, rewards are business outcomes:

```python
# Inventory
reward = revenue - procurement_cost - holding_cost - stockout_penalty - emergency_procurement_cost

# Cash Flow
reward = net_worth_growth - liquidity_penalty

# Pricing
reward = revenue - variable_cost - holding_cost - stockout_penalty
```

This is reward shaping: designing the reward signal so the AI learns behavior aligned with the real objective.

### Q-Learning

The core CS50AI algorithm used here is Q-learning. The agent learns a value for each `(state, action)` pair:

```python
Q(state, action)
```

The update rule follows the same idea from CS50AI:

```python
Q(s, a) <- Q(s, a) + alpha * (new_estimate - old_estimate)
```

where:

- `alpha` controls how quickly the agent learns
- `reward` reflects the immediate business result
- `future_reward` estimates long-term value
- `epsilon` controls exploration

### Exploration vs. Exploitation

The project uses epsilon-greedy action selection:

- Sometimes the agent explores a random action
- Otherwise it exploits the best known action

This mirrors the CS50AI Nim approach, where the AI must try different moves before it can learn which ones are best.

### SARSA

SARSA is added as an on-policy reinforcement learning comparison. While Q-learning updates toward the best possible next action, SARSA updates using the next action the current policy actually takes.

This shows the difference between:

- Off-policy learning: Q-learning
- On-policy learning: SARSA

### Deep Q-Network

The project also includes a PyTorch DQN agent. DQN extends Q-learning by replacing the table of Q-values with a neural network.

The DQN implementation includes:

- Experience replay
- Mini-batch training
- Target network updates
- Huber loss

This demonstrates how the CS50AI tabular Q-learning idea can scale toward larger state spaces.

### Training Episodes

Like the CS50AI Nim AI that learns by playing many games against itself, Reinforcement Arena trains agents over many simulated episodes.

Each episode represents a full business scenario:

- Inventory over many days
- Cash allocation over many months
- Pricing decisions until the sales horizon ends

### Policy Evaluation

After training, learned policies are compared against baselines:

- Random policy
- Fixed-order policy
- Greedy reorder policy
- Business-rule policies

This makes the project more industry-oriented: the AI is not just trained, it is benchmarked against realistic decision rules.

## Final Project Scope

Reinforcement Arena is a complete reinforcement learning platform for business decision optimization. It combines custom simulation environments, trainable RL agents, baseline policies, evaluation workflows, and an interactive Streamlit decision-support dashboard.

### Business Environments

- `InventoryEnv`: Optimizes reorder decisions under stochastic demand, seasonality, forecast noise, supplier lead time, pending orders, inventory capacity, cash constraints, holding costs, stockout penalties, and emergency procurement costs.
- `CashFlowEnv`: Optimizes monthly cash allocation across debt repayment, marketing investment, emergency savings, and liquidity preservation while tracking net worth, debt, cash, and risk penalties.
- `PricingEnv`: Optimizes dynamic pricing decisions under price elasticity, finite inventory, demand uncertainty, revenue, margin, holding costs, and stockout penalties.

### Reinforcement Learning Agents

- Q-learning agent with tabular `(state, action)` value learning and epsilon-greedy exploration.
- SARSA agent for on-policy temporal-difference learning comparison.
- PyTorch Deep Q-Network agent with replay memory, mini-batch training, Huber loss, checkpointing, and target-network updates.
- Persistent trained checkpoints for tabular agents.

### Baseline Policies

- Random action policy for naive comparison.
- Fixed-order inventory policy.
- Greedy reorder-point inventory policy.
- Cash-flow business-rule policy.
- Pricing business-rule policy.

### Training And Evaluation

- Reproducible YAML configs for inventory, cash flow, pricing, SARSA, and DQN experiments.
- Training scripts for tabular agents and DQN.
- Evaluation scripts for inventory, business environments, and DQN inventory policies.
- Experiment runner for comparing Q-learning and SARSA.
- Business metrics including reward, final cash, service level, stockout rate, units sold, emergency cost, holding cost, net worth, debt, revenue, average price, and ending inventory.
- Matplotlib-generated training and policy comparison artifacts.

### Streamlit Application

- Production-style dark slate Streamlit dashboard with green and mint accents.
- Single-page scrollable report layout with sidebar navigation.
- Portfolio overview comparing best policies across environments.
- Environment sections for inventory, cash flow, and pricing analytics.
- Live Scenario Simulator where users can change assumptions, select an environment, choose a policy, run a scenario, and inspect KPIs, outcome traces, and action logs without retraining.
- Model Ops section showing artifact status and reproducible commands.
- Cloud-safe fallback demo data when generated artifacts are unavailable.
- Custom transparent PNG visual asset in `assets/`.

### Engineering And Deployment

- Unit tests for environment behavior, Q-learning, SARSA, and DQN availability.
- GitHub Actions CI workflow for compile and test checks.
- Lightweight `requirements.txt` for Streamlit deployment.
- Optional `requirements-ml.txt` for PyTorch/DQN training.
- Streamlit deployment documentation.
- Project brief and resume-ready positioning.

## Project Structure

```text
agents/          RL and baseline policy implementations
analytics/       Metrics and plotting helpers
app/             Streamlit dashboard
configs/         YAML experiment configuration
docs/            Project brief and portfolio documentation
environments/    Business simulation environments
training/        Training and evaluation entry points
artifacts/       Generated model, CSV, and plot outputs
```

## Setup

Use Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

For DQN training, install the optional ML requirements:

```bash
python -m pip install -r requirements-ml.txt
```

## Train

```bash
python -m training.train_inventory
```

This writes:

- `artifacts/q_learning_inventory.pkl`
- `artifacts/training_history.csv`
- `artifacts/training_rewards.png`

## Evaluate

```bash
python -m training.evaluate_inventory
```

This compares Q-learning against random and greedy policies and writes:

- `artifacts/evaluation_summary.csv`
- `artifacts/policy_comparison.png`

## Compare Algorithms

```bash
python -m training.run_experiments
```

This trains and evaluates both configured learning algorithms:

- Q-learning: `configs/inventory_config.yaml`
- SARSA: `configs/inventory_sarsa_config.yaml`

Combined outputs are written to `artifacts/experiments/`.

## Train Cash Flow and Pricing

```bash
python -m training.train_tabular --config configs/cashflow_config.yaml
python -m training.evaluate_business --config configs/cashflow_config.yaml

python -m training.train_tabular --config configs/pricing_config.yaml
python -m training.evaluate_business --config configs/pricing_config.yaml
```

## Train DQN

Install the optional ML requirements, including PyTorch, first:

```bash
python -m pip install -r requirements-ml.txt
```

Then run:

```bash
python -m training.train_dqn_inventory
python -m training.evaluate_dqn_inventory
```

## Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard includes a live **Scenario Simulator**. Users can choose an environment, select a policy, adjust business assumptions, run a simulation, and inspect KPIs, outcome traces, and action logs without retraining the agents.

## Business Framing

The inventory agent observes a discretized operational state:

```python
(inventory_bucket, cash_bucket, demand_forecast_bucket, pipeline_bucket)
```

It chooses a reorder quantity:

```python
0, 10, 20, 30, 40, 50
```

The reward function reflects business value:

```python
reward = revenue - procurement_cost - holding_cost - stockout_penalty - emergency_procurement_cost
```

In the finished environment, stockouts also trigger emergency procurement cost and supplier orders arrive after a configurable lead time. This makes the decision sequential: ordering too late creates stockouts, while ordering too early ties up cash and creates holding cost.

## Algorithms

### Q-learning

Q-learning is an off-policy temporal-difference method. It updates each state-action value toward the best estimated future action, regardless of the action the current policy actually takes next.

### SARSA

SARSA is an on-policy temporal-difference method. It updates each state-action value using the next action selected by the same epsilon-greedy policy. This is useful for comparing aggressive value maximization against a policy that internalizes its own exploration behavior.

### Deep Q-Network

DQN replaces the tabular Q-table with a neural network. The implementation uses replay memory, mini-batch updates, Huber loss, and a target network to stabilize training on larger state spaces.

## Resume Bullet

Engineered **Reinforcement Arena**, a production-style reinforcement learning platform for business operations optimization, implementing Q-learning, SARSA, and PyTorch DQN agents across custom inventory, cash-flow, and pricing environments with reward shaping, policy evaluation, business baselines, CI-tested workflows, and Streamlit analytics dashboards.

## Quality Checks

```bash
python -m unittest discover
python -m compileall agents environments training analytics app tests
```

## Roadmap

- Add multi-product inventory scenarios
- Add cloud training profiles for larger DQN runs
- Add experiment tracking with run metadata and model registry support
