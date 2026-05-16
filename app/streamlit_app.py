"""Streamlit dashboard for Reinforcement Arena."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.cashflow_baseline_agent import CashFlowRuleAgent
from agents.fixed_order_agent import FixedOrderAgent
from agents.greedy_agent import GreedyAgent
from agents.pricing_baseline_agent import PricingRuleAgent
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from environments.cashflow_env import CashFlowConfig, CashFlowEnv
from environments.inventory_env import InventoryConfig, InventoryEnv
from environments.pricing_env import PricingConfig, PricingEnv


ARTIFACTS = ROOT / "artifacts"
ASSETS = ROOT / "assets"


@dataclass(frozen=True)
class EnvironmentView:
    """Dashboard metadata for a business environment."""

    key: str
    name: str
    summary_path: Path
    history_path: Path
    reward_column: str
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    summary_text: str


ENVIRONMENTS = {
    "inventory": EnvironmentView(
        key="inventory",
        name="Inventory Optimization",
        summary_path=ARTIFACTS / "evaluation_summary.csv",
        history_path=ARTIFACTS / "training_history.csv",
        reward_column="average_reward",
        primary_metric="average_reward",
        secondary_metrics=(
            "average_final_cash",
            "average_service_level",
            "average_stockout_rate",
            "average_emergency_cost",
            "average_holding_cost",
        ),
        summary_text="Lead-time-aware replenishment under uncertain seasonal demand.",
    ),
    "cashflow": EnvironmentView(
        key="cashflow",
        name="Cash Flow Optimization",
        summary_path=ARTIFACTS / "cashflow_evaluation_summary.csv",
        history_path=ARTIFACTS / "cashflow_training_history.csv",
        reward_column="average_total_reward",
        primary_metric="average_final_net_worth",
        secondary_metrics=(
            "average_total_reward",
            "average_final_cash",
            "average_final_debt",
            "average_final_emergency_fund",
            "average_liquidity_penalty",
        ),
        summary_text="Monthly capital allocation across debt, liquidity, savings, and growth.",
    ),
    "pricing": EnvironmentView(
        key="pricing",
        name="Pricing Optimization",
        summary_path=ARTIFACTS / "pricing_evaluation_summary.csv",
        history_path=ARTIFACTS / "pricing_training_history.csv",
        reward_column="average_total_reward",
        primary_metric="average_total_reward",
        secondary_metrics=(
            "average_revenue",
            "average_units_sold",
            "average_average_price",
            "average_service_level",
            "average_ending_inventory",
        ),
        summary_text="Dynamic price selection with finite inventory and elastic demand.",
    ),
}


def configure_page() -> None:
    st.set_page_config(
        page_title="Reinforcement Arena",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --ra-bg: #1f2937;
            --ra-bg-deep: #17202f;
            --ra-panel: #202d3f;
            --ra-panel-strong: #233247;
            --ra-ink: #f8fafc;
            --ra-muted: #b7c4d4;
            --ra-soft: #d7e1ec;
            --ra-border: rgba(148, 163, 184, 0.24);
            --ra-border-strong: rgba(148, 163, 184, 0.38);
            --ra-accent: #10b981;
            --ra-accent-strong: #059669;
            --ra-mint: #8df2cd;
            --ra-mint-soft: rgba(16, 185, 129, 0.18);
        }
        html {
            scroll-behavior: smooth;
        }
        .stApp {
            background: var(--ra-bg);
            color: var(--ra-ink);
        }
        .block-container {
            padding-top: 3.25rem;
            padding-bottom: 4.5rem;
            max-width: 96rem;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ra-ink);
        }
        p, li, span, label {
            color: var(--ra-muted);
        }
        [data-testid="stSidebar"] {
            background: #182231;
            border-right: 1px solid var(--ra-border);
        }
        [data-testid="stSidebar"] * {
            color: var(--ra-muted);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--ra-ink);
        }
        [data-testid="stMetric"] {
            background: var(--ra-panel);
            border: 1px solid var(--ra-border);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            box-shadow: none;
        }
        [data-testid="stMetricLabel"] {
            color: var(--ra-muted);
        }
        [data-testid="stMetricValue"] {
            color: #f8fafc;
        }
        [data-testid="stMetricDelta"] {
            background: var(--ra-mint-soft);
            border-radius: 999px;
            padding: 0.08rem 0.5rem;
            width: fit-content;
        }
        .ra-hero {
            padding: 0 0 1.3rem;
            margin-bottom: 0.8rem;
            border: 0;
            background: transparent;
            box-shadow: none;
        }
        .ra-hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.08fr) minmax(17rem, 0.72fr);
            gap: 2rem;
            align-items: center;
        }
        .ra-hero-visual {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            min-height: 13rem;
        }
        .ra-hero-visual img {
            width: min(22rem, 100%);
            height: auto;
            filter: drop-shadow(0 24px 42px rgba(2, 6, 23, 0.32));
        }
        .ra-hero-rule {
            height: 1px;
            background: linear-gradient(90deg, rgba(141, 242, 205, 0.74), rgba(148, 163, 184, 0.30), transparent);
            margin-top: 0.2rem;
        }
        .ra-title {
            font-size: 4.05rem;
            font-weight: 460;
            color: #f8fafc;
            margin: 0;
            line-height: 0.98;
        }
        .ra-title::before {
            content: "↗";
            color: var(--ra-mint);
            font-weight: 700;
            margin-right: 0.65rem;
        }
        .ra-subtitle {
            color: #dce6f2;
            font-size: 1.28rem;
            margin-top: 1.25rem;
            max-width: 56rem;
        }
        .ra-badge {
            display: inline-block;
            border: 1px solid var(--ra-border);
            border-radius: 999px;
            padding: 0.2rem 0.62rem;
            color: var(--ra-mint);
            background: rgba(16, 185, 129, 0.09);
            font-size: 0.78rem;
            margin-right: 0.35rem;
            margin-top: 1rem;
        }
        .ra-sidebar-nav {
            display: flex;
            flex-direction: column;
            gap: 0.48rem;
            margin-top: 0.7rem;
            width: 100%;
        }
        .ra-sidebar-nav a {
            display: block;
            color: #d9fff0;
            text-decoration: none;
            border: 1px solid rgba(141, 242, 205, 0.34);
            background: rgba(16, 185, 129, 0.12);
            border-radius: 999px;
            padding: 0.56rem 0.8rem;
            font-size: 0.9rem;
            width: 100%;
            box-sizing: border-box;
            text-align: center;
            line-height: 1.15;
        }
        .ra-sidebar-nav a:hover {
            background: rgba(16, 185, 129, 0.24);
            border-color: rgba(141, 242, 205, 0.66);
            color: #ffffff;
        }
        .ra-anchor {
            scroll-margin-top: 5.2rem;
        }
        .ra-section {
            margin-top: 2.4rem;
            padding: 0;
            border: 0;
            background: transparent;
            box-shadow: none;
        }
        .ra-section-title {
            color: #f8fafc;
            font-size: 1.6rem;
            font-weight: 620;
            margin: 0 0 0.25rem;
        }
        .ra-section-note {
            color: var(--ra-muted);
            margin-bottom: 1.05rem;
        }
        .ra-sim-note {
            border: 1px solid rgba(141, 242, 205, 0.28);
            border-radius: 8px;
            background: rgba(16, 185, 129, 0.10);
            padding: 0.72rem 0.9rem;
            color: #d9fff0;
            margin-bottom: 1rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--ra-border);
            border-radius: 8px;
            overflow: hidden;
            background: var(--ra-panel);
        }
        [data-testid="stVegaLiteChart"],
        [data-testid="stArrowVegaLiteChart"] {
            border: 1px solid var(--ra-border);
            border-radius: 8px;
            background: var(--ra-panel);
            padding: 0.75rem;
        }
        div[data-testid="stCodeBlock"] {
            border: 1px solid var(--ra-border);
            border-radius: 8px;
            background: var(--ra-panel);
        }
        div[data-baseweb="select"] > div {
            background-color: var(--ra-panel) !important;
            border-color: var(--ra-border-strong) !important;
            color: var(--ra-ink) !important;
        }
        button[kind="secondary"] {
            border-color: var(--ra-border) !important;
            background: var(--ra-panel) !important;
            color: var(--ra-ink) !important;
        }
        hr {
            border-color: var(--ra-border);
            margin: 2rem 0 0.5rem;
        }
        @media (max-width: 900px) {
            .ra-hero-grid {
                grid-template-columns: 1fr;
                gap: 0.75rem;
            }
            .ra-hero-visual {
                justify-content: flex-start;
                min-height: 0;
            }
            .ra-hero-visual img {
                width: min(14rem, 68vw);
            }
            .ra-title {
                font-size: 2.7rem;
            }
            .ra-subtitle {
                font-size: 1.05rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def demo_summary(environment: str) -> pd.DataFrame:
    """Return committed fallback metrics for Streamlit Cloud demos."""

    rows = {
        "inventory": [
            {
                "policy": "q_learning",
                "episodes": 200,
                "average_reward": 7592.95,
                "average_final_cash": 17592.95,
                "average_stockout_rate": 0.2351,
                "average_service_level": 0.8751,
                "average_units_sold": 1156.85,
                "average_emergency_cost": 5300.80,
                "average_holding_cost": 602.18,
                "average_ending_inventory": 25.01,
            },
            {
                "policy": "random",
                "episodes": 200,
                "average_reward": 6440.73,
                "average_final_cash": 16440.73,
                "average_stockout_rate": 0.0548,
                "average_service_level": 0.9681,
                "average_units_sold": 1280.22,
                "average_emergency_cost": 1352.96,
                "average_holding_cost": 2126.45,
                "average_ending_inventory": 90.02,
            },
            {
                "policy": "fixed_order_20",
                "episodes": 200,
                "average_reward": 7461.40,
                "average_final_cash": 17461.40,
                "average_stockout_rate": 0.3574,
                "average_service_level": 0.8758,
                "average_units_sold": 1157.34,
                "average_emergency_cost": 5284.96,
                "average_holding_cost": 465.90,
                "average_ending_inventory": 42.53,
            },
            {
                "policy": "greedy_reorder",
                "episodes": 200,
                "average_reward": 3247.78,
                "average_final_cash": 13247.78,
                "average_stockout_rate": 0.1874,
                "average_service_level": 0.8536,
                "average_units_sold": 1128.73,
                "average_emergency_cost": 6200.64,
                "average_holding_cost": 1336.94,
                "average_ending_inventory": 54.87,
            },
        ],
        "cashflow": [
            {
                "policy": "q_learning",
                "episodes": 200,
                "average_total_reward": 82204.16,
                "average_final_cash": 80398.26,
                "average_final_debt": 10961.94,
                "average_final_emergency_fund": 7767.84,
                "average_final_net_worth": 77204.16,
                "average_liquidity_penalty": 0.0,
            },
            {
                "policy": "random",
                "episodes": 200,
                "average_total_reward": 82172.20,
                "average_final_cash": 79338.51,
                "average_final_debt": 10282.84,
                "average_final_emergency_fund": 8116.53,
                "average_final_net_worth": 77172.20,
                "average_liquidity_penalty": 0.0,
            },
            {
                "policy": "business_rule",
                "episodes": 200,
                "average_total_reward": 86816.49,
                "average_final_cash": 85226.76,
                "average_final_debt": 5410.27,
                "average_final_emergency_fund": 2000.00,
                "average_final_net_worth": 81816.49,
                "average_liquidity_penalty": 0.0,
            },
        ],
        "pricing": [
            {
                "policy": "q_learning",
                "episodes": 200,
                "average_total_reward": 2239.69,
                "average_revenue": 4038.67,
                "average_units_sold": 160.0,
                "average_average_price": 26.93,
                "average_service_level": 0.9468,
                "average_ending_inventory": 0.0,
            },
            {
                "policy": "random",
                "episodes": 200,
                "average_total_reward": 2186.82,
                "average_revenue": 4024.06,
                "average_units_sold": 160.0,
                "average_average_price": 26.71,
                "average_service_level": 0.9227,
                "average_ending_inventory": 0.0,
            },
            {
                "policy": "business_rule",
                "episodes": 200,
                "average_total_reward": 2617.11,
                "average_revenue": 4407.97,
                "average_units_sold": 160.0,
                "average_average_price": 29.44,
                "average_service_level": 0.9542,
                "average_ending_inventory": 0.0,
            },
        ],
    }
    return pd.DataFrame(rows[environment])


def demo_history(environment: str) -> pd.DataFrame:
    """Return lightweight fallback training curves."""

    anchors = {
        "inventory": [7438, 9308, 6852, 4689, -1842, 5095, 5062, 4394, 7552, 8317, 5442],
        "cashflow": [76023, 76945, 87142, 75636, 75930, 78603, 84412],
        "pricing": [2034, 2885, 1912, 2322, 1891, 1798, 1925],
    }
    rewards = anchors[environment]
    max_episode = {"inventory": 2500, "cashflow": 3000, "pricing": 3000}[environment]
    step = max_episode // (len(rewards) - 1)
    rows = []
    for index, reward in enumerate(rewards):
        episode = 1 if index == 0 else index * step
        rows.append(
            {
                "episode": min(episode, max_episode),
                "total_reward": reward,
                "epsilon": max(0.04, 0.25 * (0.92**index)),
                "q_entries": 250 + index * 130,
            }
        )
    return pd.DataFrame(rows)


def dataset_for(view: EnvironmentView) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    summary = load_csv(view.summary_path)
    history = load_csv(view.history_path)
    using_demo = summary is None or history is None
    if summary is None:
        summary = demo_summary(view.key)
    if history is None:
        history = demo_history(view.key)
    return summary, history, using_demo


def format_metric(column: str, value: float) -> str:
    if "rate" in column or "level" in column:
        return f"{value:.1%}"
    if "price" in column:
        return f"${value:,.2f}"
    if any(token in column for token in ("cash", "debt", "fund", "worth", "reward", "cost", "revenue")):
        return f"${value:,.0f}"
    return f"{value:,.1f}"


def clean_label(column: str) -> str:
    label = column.replace("average_", "").replace("_", " ")
    return label.title()


def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_hero(using_demo_count: int) -> None:
    mode = "Demo fallback data" if using_demo_count else "Local experiment artifacts"
    visual_uri = image_data_uri(ASSETS / "reinforcement_arena_system.png")
    st.markdown(
        f"""
        <div class="ra-hero">
            <div class="ra-hero-grid">
                <div>
                    <div class="ra-title">Reinforcement Arena</div>
                    <div class="ra-subtitle">
                        Reinforcement learning for inventory, cash-flow, and pricing decisions.
                    </div>
                    <span class="ra-badge">Q-learning</span>
                    <span class="ra-badge">SARSA</span>
                    <span class="ra-badge">Deep Q-Network</span>
                    <span class="ra-badge">{mode}</span>
                </div>
                <div class="ra-hero-visual">
                    <img src="{visual_uri}" alt="Reinforcement learning system diagram" />
                </div>
            </div>
            <div class="ra-hero-rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_policy_snapshot(all_summaries: dict[str, pd.DataFrame]) -> None:
    rows = []
    for key, summary in all_summaries.items():
        view = ENVIRONMENTS[key]
        best = summary.sort_values(view.primary_metric, ascending=False).iloc[0]
        learned_rows = summary[summary["policy"].isin(["q_learning", "sarsa", "dqn"])]
        learned = learned_rows.iloc[0] if not learned_rows.empty else best
        rows.append(
            {
                "Environment": view.name,
                "Best Policy": best["policy"],
                "Best Metric": format_metric(view.primary_metric, float(best[view.primary_metric])),
                "Learned Policy": learned["policy"],
                "Learned Metric": format_metric(view.primary_metric, float(learned[view.primary_metric])),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def open_section(anchor: str, title: str, note: str) -> None:
    st.markdown(
        f"""
        <div id="{anchor}" class="ra-anchor"></div>
        <div class="ra-section">
            <div class="ra-section-title">{title}</div>
            <div class="ra-section-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_environment(view: EnvironmentView, summary: pd.DataFrame, history: pd.DataFrame) -> None:
    sorted_summary = summary.sort_values(view.primary_metric, ascending=False)
    best = sorted_summary.iloc[0]
    learned_rows = summary[summary["policy"].isin(["q_learning", "sarsa", "dqn"])]
    learned = learned_rows.iloc[0] if not learned_rows.empty else best
    delta = float(learned[view.primary_metric] - best[view.primary_metric])

    columns = st.columns(4)
    columns[0].metric("Best Policy", str(best["policy"]))
    columns[1].metric(clean_label(view.primary_metric), format_metric(view.primary_metric, float(best[view.primary_metric])))
    columns[2].metric("Learned Policy", str(learned["policy"]))
    columns[3].metric(
        "Gap To Best",
        format_metric(view.primary_metric, delta),
    )

    left, right = st.columns([1.45, 1])
    with left:
        st.markdown("#### Training Curve")
        reward_chart = history[["episode", "total_reward"]].set_index("episode")
        st.line_chart(reward_chart, height=300)

    with right:
        st.markdown("#### Policy Leaderboard")
        display_columns = ["policy", view.primary_metric, *view.secondary_metrics]
        display_columns = [column for column in display_columns if column in sorted_summary.columns]
        st.dataframe(
            sorted_summary[display_columns],
            hide_index=True,
            width="stretch",
            column_config={
                column: st.column_config.NumberColumn(clean_label(column), format="%.2f")
                for column in display_columns
                if column != "policy"
            },
        )

    metric_columns = [column for column in view.secondary_metrics if column in summary.columns]
    if metric_columns:
        st.markdown("#### Operating Metrics")
        selected_metric = st.selectbox(
            "Metric",
            options=metric_columns,
            format_func=clean_label,
            key=f"{view.key}_metric",
            label_visibility="collapsed",
        )
        st.bar_chart(summary.set_index("policy")[[selected_metric]], height=280)


def render_model_ops(using_demo: dict[str, bool]) -> None:
    st.subheader("Model Operations")
    status_rows = []
    for key, view in ENVIRONMENTS.items():
        status_rows.append(
            {
                "Environment": view.name,
                "Summary Artifact": str(view.summary_path),
                "Training Artifact": str(view.history_path),
                "Data Source": "Demo fallback" if using_demo[key] else "Generated artifact",
            }
        )
    st.dataframe(pd.DataFrame(status_rows), hide_index=True, width="stretch")

    st.markdown("#### Reproducible Commands")
    st.code(
        "\n".join(
            [
                "python -m training.train_inventory",
                "python -m training.evaluate_inventory",
                "python -m training.train_tabular --config configs/cashflow_config.yaml",
                "python -m training.evaluate_business --config configs/cashflow_config.yaml",
                "python -m training.train_tabular --config configs/pricing_config.yaml",
                "python -m training.evaluate_business --config configs/pricing_config.yaml",
                "python -m training.train_dqn_inventory",
                "python -m training.evaluate_dqn_inventory",
            ]
        ),
        language="bash",
    )


def load_policy(path: Path) -> QLearningAgent | None:
    if not path.exists():
        return None
    try:
        return QLearningAgent.load(path)
    except (OSError, TypeError, ValueError):
        return None


def build_simulator_env(environment: str, params: dict[str, Any]):
    if environment == "Inventory":
        config = InventoryConfig(
            initial_inventory=params["initial_inventory"],
            initial_cash=params["initial_cash"],
            mean_demand=params["mean_demand"],
            demand_std=params["demand_std"],
            holding_cost=params["holding_cost"],
            stockout_penalty=params["stockout_penalty"],
            supplier_lead_time=params["supplier_lead_time"],
            episode_length=params["horizon"],
            seed=params["seed"],
        )
        return InventoryEnv(config)
    if environment == "Cash Flow":
        config = CashFlowConfig(
            initial_cash=params["initial_cash"],
            initial_debt=params["initial_debt"],
            emergency_fund=params["emergency_fund"],
            monthly_revenue=params["monthly_revenue"],
            operating_expenses=params["operating_expenses"],
            marketing_roi=params["marketing_roi"],
            allocation_amount=params["allocation_amount"],
            episode_length=params["horizon"],
            seed=params["seed"],
        )
        return CashFlowEnv(config)

    config = PricingConfig(
        initial_inventory=params["initial_inventory"],
        base_demand=params["base_demand"],
        demand_std=params["demand_std"],
        price_elasticity=params["price_elasticity"],
        holding_cost=params["holding_cost"],
        stockout_penalty=params["stockout_penalty"],
        episode_length=params["horizon"],
        seed=params["seed"],
    )
    return PricingEnv(config)


def build_simulator_policy(environment: str, policy_name: str, env):
    if policy_name == "Random":
        return RandomAgent(env.action_space, seed=101)
    if environment == "Inventory":
        if policy_name == "Q-learning":
            return load_policy(ARTIFACTS / "q_learning_inventory.pkl") or GreedyAgent(
                env.action_space, inventory_bucket_size=env.config.inventory_bucket_size
            )
        if policy_name == "Fixed order":
            return FixedOrderAgent(env.action_space, order_quantity=20)
        return GreedyAgent(env.action_space, inventory_bucket_size=env.config.inventory_bucket_size)
    if environment == "Cash Flow":
        if policy_name == "Q-learning":
            return load_policy(ARTIFACTS / "q_learning_cashflow.pkl") or CashFlowRuleAgent()
        return CashFlowRuleAgent()
    if policy_name == "Q-learning":
        return load_policy(ARTIFACTS / "q_learning_pricing.pkl") or PricingRuleAgent()
    return PricingRuleAgent()


def run_simulation(env, policy, seed: int) -> pd.DataFrame:
    state, _ = env.reset(seed=seed)
    done = False
    rows = []
    while not done:
        action = policy.choose_action(state, explore=False)
        next_state, reward, done, info = env.step(action)
        rows.append({"state": state, "next_state": next_state, **info})
        state = next_state
    return pd.DataFrame(rows)


def render_inventory_controls() -> dict[str, Any]:
    cols = st.columns(4)
    return {
        "initial_inventory": cols[0].slider("Initial inventory", 0, 120, 40, 5),
        "initial_cash": float(cols[1].slider("Initial cash", 1000, 25000, 10000, 500)),
        "mean_demand": float(cols[2].slider("Mean demand", 5, 45, 22, 1)),
        "demand_std": float(cols[3].slider("Demand volatility", 0, 15, 6, 1)),
        "holding_cost": float(cols[0].slider("Holding cost", 0.0, 3.0, 0.5, 0.1)),
        "stockout_penalty": float(cols[1].slider("Stockout penalty", 0.0, 25.0, 8.0, 0.5)),
        "supplier_lead_time": cols[2].slider("Supplier lead time", 0, 6, 2, 1),
        "horizon": cols[3].slider("Simulation days", 15, 120, 60, 5),
    }


def render_cashflow_controls() -> dict[str, Any]:
    cols = st.columns(4)
    return {
        "initial_cash": float(cols[0].slider("Initial cash", 0, 50000, 5000, 500)),
        "initial_debt": float(cols[1].slider("Initial debt", 0, 50000, 12000, 500)),
        "emergency_fund": float(cols[2].slider("Emergency fund", 0, 25000, 2000, 500)),
        "monthly_revenue": float(cols[3].slider("Monthly revenue", 1000, 25000, 8000, 500)),
        "operating_expenses": float(cols[0].slider("Operating expenses", 500, 20000, 4500, 500)),
        "marketing_roi": float(cols[1].slider("Marketing ROI", 0.5, 3.0, 1.35, 0.05)),
        "allocation_amount": float(cols[2].slider("Allocation amount", 250, 5000, 1000, 250)),
        "horizon": cols[3].slider("Simulation months", 6, 60, 24, 3),
    }


def render_pricing_controls() -> dict[str, Any]:
    cols = st.columns(4)
    return {
        "initial_inventory": cols[0].slider("Initial inventory", 20, 500, 160, 10),
        "base_demand": float(cols[1].slider("Base demand", 5, 80, 26, 1)),
        "demand_std": float(cols[2].slider("Demand volatility", 0, 20, 4, 1)),
        "price_elasticity": float(cols[3].slider("Price elasticity", 0.2, 3.0, 1.4, 0.1)),
        "holding_cost": float(cols[0].slider("Holding cost", 0.0, 3.0, 0.35, 0.05)),
        "stockout_penalty": float(cols[1].slider("Stockout penalty", 0.0, 25.0, 6.0, 0.5)),
        "horizon": cols[2].slider("Simulation days", 10, 90, 45, 5),
    }


def simulation_kpis(environment: str, trace: pd.DataFrame) -> list[tuple[str, str]]:
    if environment == "Inventory":
        total_demand = trace["demand"].sum()
        service_level = 1 if total_demand == 0 else 1 - trace["unmet_demand"].sum() / total_demand
        return [
            ("Reward", f"${trace['reward'].sum():,.0f}"),
            ("Final Cash", f"${trace['cash'].iloc[-1]:,.0f}"),
            ("Service Level", f"{service_level:.1%}"),
            ("Ending Inventory", f"{trace['inventory'].iloc[-1]:,.0f}"),
        ]
    if environment == "Cash Flow":
        return [
            ("Reward", f"${trace['reward'].sum():,.0f}"),
            ("Net Worth", f"${trace['net_worth'].iloc[-1]:,.0f}"),
            ("Final Cash", f"${trace['cash'].iloc[-1]:,.0f}"),
            ("Debt", f"${trace['debt'].iloc[-1]:,.0f}"),
        ]
    return [
        ("Reward", f"${trace['reward'].sum():,.0f}"),
        ("Revenue", f"${trace['revenue'].sum():,.0f}"),
        ("Units Sold", f"{trace['units_sold'].sum():,.0f}"),
        ("Average Price", f"${trace['price'].mean():,.2f}"),
    ]


def render_trace_chart(environment: str, trace: pd.DataFrame) -> None:
    if environment == "Inventory":
        chart = trace[["day", "inventory", "pending_units", "demand"]].set_index("day")
    elif environment == "Cash Flow":
        chart = trace[["month", "cash", "debt", "emergency_fund", "net_worth"]].set_index("month")
    else:
        chart = trace[["day", "inventory", "price", "demand", "units_sold"]].set_index("day")
    st.line_chart(chart, height=320)


def render_scenario_simulator() -> None:
    st.markdown(
        "<div class='ra-sim-note'>Change assumptions, choose a policy, and run a fresh simulated scenario without retraining the model.</div>",
        unsafe_allow_html=True,
    )
    top_cols = st.columns([1, 1, 1])
    environment = top_cols[0].selectbox("Environment", ["Inventory", "Cash Flow", "Pricing"])
    policy_options = {
        "Inventory": ["Q-learning", "Greedy reorder", "Fixed order", "Random"],
        "Cash Flow": ["Q-learning", "Business rule", "Random"],
        "Pricing": ["Q-learning", "Business rule", "Random"],
    }
    policy_name = top_cols[1].selectbox("Policy", policy_options[environment])
    seed = top_cols[2].number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    if environment == "Inventory":
        params = render_inventory_controls()
    elif environment == "Cash Flow":
        params = render_cashflow_controls()
    else:
        params = render_pricing_controls()
    params["seed"] = int(seed)

    if st.button("Run Scenario", type="primary", width="stretch"):
        env = build_simulator_env(environment, params)
        policy = build_simulator_policy(environment, policy_name, env)
        trace = run_simulation(env, policy, int(seed))
        st.session_state["scenario_trace"] = trace
        st.session_state["scenario_environment"] = environment
        st.session_state["scenario_policy"] = policy_name

    trace = st.session_state.get("scenario_trace")
    if trace is None:
        return

    environment = st.session_state["scenario_environment"]
    policy_name = st.session_state["scenario_policy"]
    st.markdown(f"#### Scenario Results: {environment} / {policy_name}")
    cols = st.columns(4)
    for col, (label, value) in zip(cols, simulation_kpis(environment, trace)):
        col.metric(label, value)

    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("#### Outcome Trace")
        render_trace_chart(environment, trace)
    with right:
        st.markdown("#### Action Log")
        action_cols = [column for column in ("day", "month", "action", "reward") if column in trace.columns]
        st.dataframe(trace[action_cols].tail(15), hide_index=True, width="stretch")


def main() -> None:
    configure_page()

    summaries: dict[str, pd.DataFrame] = {}
    histories: dict[str, pd.DataFrame] = {}
    using_demo: dict[str, bool] = {}
    for key, view in ENVIRONMENTS.items():
        summary, history, demo = dataset_for(view)
        summaries[key] = summary
        histories[key] = history
        using_demo[key] = demo

    render_hero(sum(using_demo.values()))

    with st.sidebar:
        st.header("Navigation")
        st.caption("Artifacts are loaded from the local `artifacts/` directory when available.")
        for key, view in ENVIRONMENTS.items():
            status = "Demo fallback" if using_demo[key] else "Generated artifact"
            st.write(f"{view.name}: {status}")
        st.markdown("---")
        st.markdown(
            """
            <div class="ra-sidebar-nav">
                <a href="#overview">Overview</a>
                <a href="#simulator">Simulator</a>
                <a href="#inventory">Inventory</a>
                <a href="#cash-flow">Cash Flow</a>
                <a href="#pricing">Pricing</a>
                <a href="#model-ops">Model Ops</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    open_section(
        "overview",
        "Portfolio Dashboard",
        "Policy quality across operational and financial decision environments.",
    )
    render_policy_snapshot(summaries)
    overview_cols = st.columns(3)
    for column, key in zip(overview_cols, ENVIRONMENTS):
        view = ENVIRONMENTS[key]
        summary = summaries[key]
        best = summary.sort_values(view.primary_metric, ascending=False).iloc[0]
        column.metric(
            view.name,
            format_metric(view.primary_metric, float(best[view.primary_metric])),
            delta=str(best["policy"]),
        )

    open_section(
        "simulator",
        "Scenario Simulator",
        "Interactive decision support using the same reinforcement-learning environments.",
    )
    render_scenario_simulator()

    open_section("inventory", "Inventory Optimization", ENVIRONMENTS["inventory"].summary_text)
    render_environment(ENVIRONMENTS["inventory"], summaries["inventory"], histories["inventory"])

    open_section("cash-flow", "Cash Flow Optimization", ENVIRONMENTS["cashflow"].summary_text)
    render_environment(ENVIRONMENTS["cashflow"], summaries["cashflow"], histories["cashflow"])

    open_section("pricing", "Pricing Optimization", ENVIRONMENTS["pricing"].summary_text)
    render_environment(ENVIRONMENTS["pricing"], summaries["pricing"], histories["pricing"])

    open_section(
        "model-ops",
        "Model Operations",
        "Artifact status, training commands, and reproducibility checks.",
    )
    render_model_ops(using_demo)


if __name__ == "__main__":
    main()
