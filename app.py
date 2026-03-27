"""
GPU Cloud Business Model Calculator
Interactive tool for modeling GPU procurement mix, profitability, and risk.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Dict

st.set_page_config(
    page_title="GPU Cloud Business Model",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GPU Catalog
# ─────────────────────────────────────────────

@dataclass
class GPUSpec:
    purchase_price: float          # USD per GPU (for owned depreciation + capital at risk)
    rental_price_hr: float         # $/GPU/hr all-in blended rate (for rented)
    on_demand_cost_hr: float       # $/hr to rent from cloud provider (for on-demand)
    power_watts: float             # GPU TDP (for owned power cost)
    suggested_billing_rate: float  # $/hr suggested customer billing rate


GPU_CATALOG: Dict[str, GPUSpec] = {
    "NVIDIA H100 SXM (80GB)":  GPUSpec(32000, 2.74, 3.50, 700,  4.50),
    "NVIDIA H100 PCIe (80GB)": GPUSpec(26000, 2.19, 2.80, 350,  3.50),
    "NVIDIA A100 SXM (80GB)":  GPUSpec(16000, 1.37, 2.00, 400,  2.60),
    "NVIDIA A100 PCIe (80GB)": GPUSpec(13000, 1.10, 1.80, 250,  2.30),
    "NVIDIA L40S (48GB)":      GPUSpec(12000, 1.10, 1.80, 350,  2.30),
    "NVIDIA A10G (24GB)":      GPUSpec( 4000, 0.34, 0.90, 150,  1.20),
    "Custom GPU":               GPUSpec(10000, 0.68, 1.50, 300,  2.00),
}

HOURS_PER_MONTH = 730
PUE = 1.3  # Power Usage Effectiveness (data center overhead)

# ─────────────────────────────────────────────
# Financial Model
# ─────────────────────────────────────────────

def calc_metrics(
    total_gpus: int,
    pct_owned: float,
    pct_rented: float,
    pct_on_demand: float,
    # Owned inputs
    purchase_price: float,
    depreciation_months: int,
    colo_cost_per_gpu: float,
    electricity_rate: float,
    power_watts: float,
    # Rented inputs
    rental_price_hr: float,
    # On-demand inputs
    on_demand_cost_hr: float,
    # Revenue
    customer_billing_rate: float,
    utilization: float,
) -> dict:
    owned   = total_gpus * pct_owned
    rented  = total_gpus * pct_rented
    on_dem  = total_gpus * pct_on_demand

    # Owned: depreciation + colo + power (with PUE overhead) — always fixed
    power_cost   = (power_watts / 1000) * 24 * 30 * electricity_rate * PUE
    opex_per_gpu = colo_cost_per_gpu + power_cost
    owned_cost   = owned * (purchase_price / depreciation_months + opex_per_gpu)

    # Rented: flat blended hourly rate × hours in month — always fixed
    rented_cost  = rented * rental_price_hr * HOURS_PER_MONTH

    # Priority utilization: fill owned first, then rented, then on-demand
    # On-demand only activates once owned + rented are fully saturated
    demanded_gpus  = total_gpus * utilization
    od_active_gpus = max(0.0, demanded_gpus - owned - rented)
    od_cost        = od_active_gpus * HOURS_PER_MONTH * on_demand_cost_hr

    # Effective utilization per tier (for display)
    owned_util_eff  = min(1.0, demanded_gpus / owned)           if owned  > 0 else 0.0
    rented_util_eff = min(1.0, max(0.0, demanded_gpus - owned) / rented)  if rented > 0 else 0.0
    od_util_eff     = (od_active_gpus / on_dem)                 if on_dem > 0 else 0.0

    fixed_costs    = owned_cost + rented_cost
    variable_costs = od_cost
    total_cost     = fixed_costs + variable_costs

    # Revenue — based on total GPU-hours actually served
    gpu_hours = demanded_gpus * HOURS_PER_MONTH
    revenue   = gpu_hours * customer_billing_rate

    # Profit
    profit = revenue - total_cost
    margin = (profit / revenue * 100) if revenue > 0 else float("-inf")

    # Break-even utilization (piecewise: on-demand only kicks in above owned+rented threshold)
    # Zone A (util ≤ owned+rented fraction): no on-demand cost
    #   break-even_A = fixed_costs / (total_gpus × HOURS × rate)
    # Zone B (util > threshold): on-demand active
    #   break-even_B = (fixed_costs − HOURS×(owned+rented)×od_rate) / (HOURS×total_gpus×(rate−od_rate))
    od_threshold = (owned + rented) / total_gpus if total_gpus > 0 else 1.0
    if customer_billing_rate > 0 and total_gpus > 0:
        be_A = fixed_costs / (total_gpus * HOURS_PER_MONTH * customer_billing_rate)
        if be_A <= od_threshold:
            breakeven = be_A
        else:
            net_rate = customer_billing_rate - on_demand_cost_hr
            if net_rate > 0:
                breakeven = (fixed_costs - HOURS_PER_MONTH * (owned + rented) * on_demand_cost_hr) \
                            / (HOURS_PER_MONTH * total_gpus * net_rate)
            else:
                breakeven = float("inf")
    else:
        breakeven = float("inf")

    # Capital & risk
    capital      = owned * purchase_price
    payback      = (capital / profit) if profit > 0 else float("inf")
    max_exposure = fixed_costs  # total cost at 0% utilization (on-demand = $0)

    return dict(
        owned=owned, rented=rented, on_dem=on_dem,
        owned_util_eff=owned_util_eff, rented_util_eff=rented_util_eff, od_util_eff=od_util_eff,
        od_active_gpus=od_active_gpus,
        revenue=revenue, gpu_hours=gpu_hours,
        owned_cost=owned_cost, rented_cost=rented_cost, od_cost=od_cost,
        fixed_costs=fixed_costs, variable_costs=variable_costs, total_cost=total_cost,
        profit=profit, margin=margin, annual_profit=profit * 12,
        breakeven=breakeven, capital=capital,
        max_exposure=max_exposure,
        payback=payback, opex_per_gpu=opex_per_gpu,
        rev_per_gpu=revenue / total_gpus if total_gpus else 0,
        cost_per_gpu=total_cost / total_gpus if total_gpus else 0,
        profit_per_gpu=profit / total_gpus if total_gpus else 0,
    )


@st.cache_data(show_spinner=False)
def util_curve(params_tuple, util_array):
    params = dict(params_tuple)
    rows = []
    for u in util_array:
        mm = calc_metrics(**{**params, "utilization": u})
        rows.append(dict(u=u * 100, revenue=mm["revenue"],
                         total_cost=mm["total_cost"], fixed=mm["fixed_costs"],
                         profit=mm["profit"]))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def profit_heatmap(params_tuple, util_vals, owned_vals, pct_rented_base):
    params = dict(params_tuple)
    Z = np.zeros((len(util_vals), len(owned_vals)))
    for i, u in enumerate(util_vals):
        for j, own in enumerate(owned_vals):
            rent = min(100 - own, int(pct_rented_base * 100))
            od   = 100 - own - rent
            mm = calc_metrics(**{**params, "utilization": u / 100,
                                 "pct_owned": own / 100, "pct_rented": rent / 100,
                                 "pct_on_demand": od / 100})
            Z[i, j] = mm["profit"]
    return Z


# ─────────────────────────────────────────────
# Linked slider + number input helper
# ─────────────────────────────────────────────

def linked_slider(label, min_val, max_val, default, step, key,
                  fmt="%d", help=None, max_val_dynamic=None):
    """Render a slider and number input that stay in sync via session state."""
    state_key = f"_v_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = default

    # Clamp stored value if dynamic max changed (e.g. pct_rented after pct_owned moves)
    effective_max = max_val_dynamic if max_val_dynamic is not None else max_val
    if st.session_state[state_key] > effective_max:
        st.session_state[state_key] = effective_max

    def from_slider():
        st.session_state[state_key] = st.session_state[f"_sl_{key}"]

    def from_input():
        v = st.session_state[f"_ni_{key}"]
        st.session_state[state_key] = int(max(min_val, min(effective_max, v)))

    col_s, col_n = st.columns([3, 1])
    with col_s:
        st.slider(label, min_val, effective_max,
                  value=st.session_state[state_key],
                  step=step, key=f"_sl_{key}",
                  on_change=from_slider, help=help)
    with col_n:
        st.number_input("value", min_val, effective_max,
                        value=st.session_state[state_key],
                        step=step, key=f"_ni_{key}",
                        on_change=from_input,
                        label_visibility="collapsed",
                        format=fmt)
    return st.session_state[state_key]


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

def main():
    st.title("⚡ GPU Cloud Business Model Calculator")
    st.caption("Model procurement mix (owned / rented / on-demand) to maximise profit and quantify risk.")

    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        st.header("Inputs")

        # GPU type
        st.subheader("GPU Type")
        gpu_name = st.selectbox("Select GPU", list(GPU_CATALOG.keys()))
        spec = GPU_CATALOG[gpu_name]

        with st.expander("Edit GPU Base Specs", expanded=(gpu_name == "Custom GPU")):
            purchase_price = st.number_input("Purchase Price ($/GPU)", 500, 200_000,
                                             int(spec.purchase_price), step=500,
                                             help="Used for owned depreciation and capital at risk")
            power_watts    = st.number_input("TDP (Watts)", 50, 1500,
                                             int(spec.power_watts), step=10,
                                             help="GPU thermal design power (used to calculate owned power cost)")

        # Fleet & Mix
        st.subheader("Fleet Size & Mix")
        total_gpus = linked_slider("Total GPUs", 1, 5000, 100, 1, "fleet")
        st.caption("Owned + Rented + On-Demand = 100%")
        pct_owned  = linked_slider("% Owned",  0, 100, 50, 1, "owned")
        pct_rented = linked_slider("% Rented", 0, 100, 30, 1, "rented",
                                   max_val_dynamic=100 - pct_owned)
        pct_on_demand = 100 - pct_owned - pct_rented
        st.info(f"On-Demand: **{pct_on_demand}%** ({int(total_gpus * pct_on_demand / 100)} GPUs)")

        # ── Owned GPU Costs ───────────────────
        st.subheader("Owned GPU Costs")
        colo_cost_per_gpu   = st.number_input("Colo ($/GPU/mo)", 0, 1000, 120, step=10,
                                               help="Rack space + networking per GPU per month")
        electricity_rate    = st.number_input("Electricity ($/kWh)", 0.01, 0.50, 0.08,
                                               step=0.01, format="%.2f")
        depreciation_months = st.number_input("Depreciation (months)", 12, 84, 36, step=6)

        # ── Rented GPU Costs ──────────────────
        st.subheader("Rented GPU Costs")
        rental_price_hr = st.number_input(
            "Blended Rental Price ($/GPU/hr)", 0.10, 50.0,
            float(spec.rental_price_hr), step=0.01, format="%.2f",
            help="All-in hourly cost per rented GPU — hardware + colo + power bundled",
        )

        # ── On-Demand GPU Costs ───────────────
        st.subheader("On-Demand GPU Costs")
        on_demand_cost_hr = st.number_input(
            "On-Demand Price ($/GPU/hr)", 0.10, 50.0,
            float(spec.on_demand_cost_hr), step=0.10, format="%.2f",
            help="Hourly cost to rent from a cloud provider (AWS, GCP, CoreWeave, etc.)",
        )

        # ── Revenue ──────────────────────────
        st.subheader("Revenue")
        customer_billing_rate = st.number_input("Customer Billing Rate ($/GPU/hr)",
                                                0.10, 100.0,
                                                float(spec.suggested_billing_rate),
                                                step=0.10, format="%.2f",
                                                help="Hourly rate you charge your customers")

        if customer_billing_rate <= on_demand_cost_hr:
            st.warning("⚠️ Billing rate ≤ on-demand cost — on-demand GPUs lose money per hour!")

        # ── Demand ───────────────────────────
        st.subheader("Demand")
        utilization  = linked_slider("Expected Utilization (%)", 0, 100, 70, 1, "util",
                                     help="% of time GPUs are actively serving paying customers") / 100
        time_horizon = linked_slider("Analysis Horizon (months)", 6, 60, 24, 1, "horizon")

    # ── Build params ─────────────────────────
    params = dict(
        total_gpus=total_gpus,
        pct_owned=pct_owned / 100,
        pct_rented=pct_rented / 100,
        pct_on_demand=pct_on_demand / 100,
        purchase_price=purchase_price,
        depreciation_months=depreciation_months,
        colo_cost_per_gpu=colo_cost_per_gpu,
        electricity_rate=electricity_rate,
        power_watts=power_watts,
        rental_price_hr=rental_price_hr,
        on_demand_cost_hr=on_demand_cost_hr,
        customer_billing_rate=customer_billing_rate,
        utilization=utilization,
    )
    m = calc_metrics(**params)
    params_tuple = tuple(sorted(params.items()))

    be_pct     = m["breakeven"] * 100
    be_display = f"{be_pct:.1f}%" if be_pct <= 100 else "Never"
    cushion    = utilization * 100 - be_pct

    # ── Tabs ─────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Sensitivity", "⚠️ Risk", "🔀 Scenarios"])

    # ════════════════════════════════════════
    # TAB 1 — DASHBOARD
    # ════════════════════════════════════════
    with tab1:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Monthly Revenue",    f"${m['revenue']:,.0f}")
        c2.metric("Monthly Profit",     f"${m['profit']:,.0f}",
                  delta=f"{m['margin']:.1f}% margin")
        c3.metric("Annual Profit",      f"${m['annual_profit']:,.0f}")
        c4.metric("Break-even Util.",   be_display,
                  delta=f"{cushion:+.1f}pp cushion" if be_pct <= 100 else None)
        c5.metric("Capital Invested",   f"${m['capital']:,.0f}")
        c6.metric(f"Max Exposure ({time_horizon}mo)", f"${m['max_exposure'] * time_horizon:,.0f}",
                  help=f"Total fixed cost over {time_horizon} months at 0% utilization — "
                       "owned depreciation/opex + rented fees. On-demand incurs no cost at zero demand.")

        st.divider()
        col_left, col_right = st.columns([1.3, 1])

        with col_left:
            util_arr = np.linspace(0, 1.0, 101)
            df_u = util_curve(params_tuple, util_arr)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_u["u"], y=df_u["revenue"],
                                     name="Revenue", line=dict(color="#2196F3", width=2)))
            fig.add_trace(go.Scatter(x=df_u["u"], y=df_u["total_cost"],
                                     name="Total Cost", line=dict(color="#F44336", width=2)))
            fig.add_trace(go.Scatter(x=df_u["u"], y=df_u["fixed"],
                                     name="Fixed Cost (Max Exposure)",
                                     line=dict(color="#FF9800", width=2, dash="dash")))
            fig.add_trace(go.Scatter(x=df_u["u"], y=df_u["profit"],
                                     name="Profit", line=dict(color="#4CAF50", width=2),
                                     fill="tozeroy", fillcolor="rgba(76,175,80,0.08)"))
            fig.add_vline(x=utilization * 100, line_dash="dot", line_color="white",
                          annotation_text=f"Now: {utilization*100:.0f}%")
            if be_pct <= 100:
                fig.add_vline(x=be_pct, line_dash="dash", line_color="#FF9800",
                              annotation_text=f"Break-even: {be_pct:.1f}%")
            fig.update_layout(title="Revenue, Cost & Profit vs. Utilization",
                              xaxis_title="Utilization (%)", yaxis_title="Monthly ($)",
                              template="plotly_dark", height=400,
                              legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            cost_items = [
                ("Owned (Depr+Opex)",        m["owned_cost"],  "#2196F3"),
                ("Rented (Blended Monthly)", m["rented_cost"], "#9C27B0"),
                ("On-Demand (Variable)",     m["od_cost"],     "#FF9800"),
            ]
            filtered = [(l, v, c) for l, v, c in cost_items if v > 0]
            if filtered:
                labels, vals, colors = zip(*filtered)
                fig_pie = go.Figure(go.Pie(labels=labels, values=vals, hole=0.4,
                                           marker_colors=colors))
                fig_pie.update_layout(title="Monthly Cost Breakdown",
                                      template="plotly_dark", height=400,
                                      legend=dict(orientation="h", y=-0.15))
                st.plotly_chart(fig_pie, use_container_width=True)

        # P&L table
        st.subheader("Monthly P&L Summary")
        pnl = [
            ("Revenue",                                m["revenue"],       m["revenue"] * 12,
             f"{m['gpu_hours']:,.0f} GPU-hrs/mo @ ${customer_billing_rate:.2f}/hr"),
            (f"  Owned  ({int(m['owned'])} GPUs)",    -m["owned_cost"],   -m["owned_cost"] * 12,
             f"Depr ${purchase_price/depreciation_months:,.0f} + Opex ${m['opex_per_gpu']:,.0f} /GPU/mo — {m['owned_util_eff']*100:.0f}% util (fixed cost)"),
            (f"  Rented ({int(m['rented'])} GPUs)",   -m["rented_cost"],  -m["rented_cost"] * 12,
             f"${rental_price_hr:.2f}/GPU/hr blended — {m['rented_util_eff']*100:.0f}% util (fixed cost)"),
            (f"  On-Dem ({int(m['on_dem'])} GPUs)",   -m["od_cost"],      -m["od_cost"] * 12,
             f"${on_demand_cost_hr:.2f}/hr — {m['od_util_eff']*100:.0f}% util, {m['od_active_gpus']:.0f} active GPUs (variable)"),
            ("Net Profit",                             m["profit"],        m["annual_profit"],
             f"{m['margin']:.1f}% gross margin"),
        ]

        def fmt(v):
            if v is None: return ""
            if abs(v) >= 1e6: return f"${v/1e6:.2f}M"
            return f"${v:,.0f}"

        df_pnl = pd.DataFrame(pnl, columns=["Line Item", "Monthly", "Annual", "Note"])
        df_pnl["Monthly"] = df_pnl["Monthly"].apply(fmt)
        df_pnl["Annual"]  = df_pnl["Annual"].apply(fmt)
        st.dataframe(df_pnl, use_container_width=True, hide_index=True)

        # Per-GPU metrics
        st.subheader("Per-GPU Economics (Monthly)")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Revenue / GPU", f"${m['rev_per_gpu']:,.0f}")
        g2.metric("Cost / GPU",    f"${m['cost_per_gpu']:,.0f}")
        g3.metric("Profit / GPU",  f"${m['profit_per_gpu']:,.0f}")
        g4.metric("Owned Opex / GPU", f"${m['opex_per_gpu']:,.0f}",
                  help="Colo + power cost per owned GPU per month")

    # ════════════════════════════════════════
    # TAB 2 — SENSITIVITY
    # ════════════════════════════════════════
    with tab2:
        st.subheader("Sensitivity Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Profit & Break-even vs. % Owned**")
            owned_range = np.arange(0, 105, 5)
            mix_rows = []
            for own in owned_range:
                rent = min(100 - own, pct_rented)
                od   = 100 - own - rent
                mm = calc_metrics(**{**params, "pct_owned": own/100,
                                    "pct_rented": rent/100, "pct_on_demand": od/100})
                mix_rows.append({"own": own, "profit": mm["profit"],
                                 "be": mm["breakeven"] * 100,
                                 "exposure": mm["max_exposure"]})
            df_mix = pd.DataFrame(mix_rows)

            fig_mix = make_subplots(specs=[[{"secondary_y": True}]])
            fig_mix.add_trace(go.Scatter(x=df_mix["own"], y=df_mix["profit"],
                                         name="Monthly Profit ($)",
                                         line=dict(color="#4CAF50")), secondary_y=False)
            fig_mix.add_trace(go.Scatter(x=df_mix["own"],
                                         y=df_mix["be"].clip(upper=100),
                                         name="Break-even Util. (%)",
                                         line=dict(color="#FF9800", dash="dash")), secondary_y=True)
            fig_mix.add_vline(x=pct_owned, line_dash="dot", line_color="white",
                              annotation_text=f"Now: {pct_owned}%")
            fig_mix.update_yaxes(title_text="Monthly Profit ($)", secondary_y=False)
            fig_mix.update_yaxes(title_text="Break-even (%)", secondary_y=True)
            fig_mix.update_xaxes(title_text="% Owned")
            fig_mix.update_layout(template="plotly_dark", height=360,
                                   legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig_mix, use_container_width=True)

        with col2:
            st.markdown("**Profit & Margin vs. Customer Billing Rate**")
            rate_min  = max(0.10, on_demand_cost_hr * 0.5)
            rate_max  = customer_billing_rate * 2.5
            rate_range = np.linspace(rate_min, rate_max, 60)
            rate_rows = []
            for r in rate_range:
                mm = calc_metrics(**{**params, "customer_billing_rate": r})
                rate_rows.append({"rate": r, "profit": mm["profit"], "margin": mm["margin"]})
            df_rate = pd.DataFrame(rate_rows)

            fig_rate = make_subplots(specs=[[{"secondary_y": True}]])
            fig_rate.add_trace(go.Scatter(x=df_rate["rate"], y=df_rate["profit"],
                                          name="Monthly Profit ($)",
                                          line=dict(color="#4CAF50")), secondary_y=False)
            fig_rate.add_trace(go.Scatter(x=df_rate["rate"], y=df_rate["margin"],
                                          name="Gross Margin (%)",
                                          line=dict(color="#2196F3", dash="dash")), secondary_y=True)
            fig_rate.add_vline(x=customer_billing_rate, line_dash="dot", line_color="white",
                               annotation_text=f"Now: ${customer_billing_rate:.2f}")
            fig_rate.add_vline(x=on_demand_cost_hr, line_dash="dash", line_color="#F44336",
                               annotation_text=f"OD Cost: ${on_demand_cost_hr:.2f}")
            fig_rate.update_yaxes(title_text="Monthly Profit ($)", secondary_y=False)
            fig_rate.update_yaxes(title_text="Gross Margin (%)", secondary_y=True)
            fig_rate.update_xaxes(title_text="Customer Billing Rate ($/hr)")
            fig_rate.update_layout(template="plotly_dark", height=360,
                                    legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig_rate, use_container_width=True)

        # Heatmap
        st.markdown("**Profit Heatmap: Utilization × % Owned** *(rented % held constant)*")
        util_vals  = np.arange(10, 105, 5)
        owned_vals = np.arange(0, 105, 10)
        Z = profit_heatmap(params_tuple, util_vals, owned_vals, pct_rented / 100)

        text_Z = [[f"${Z[i,j]/1000:.0f}K" for j in range(len(owned_vals))]
                  for i in range(len(util_vals))]
        fig_heat = go.Figure(go.Heatmap(
            z=Z, x=[f"{o}%" for o in owned_vals], y=[f"{u}%" for u in util_vals],
            colorscale="RdYlGn", zmid=0,
            text=text_Z, texttemplate="%{text}", textfont={"size": 8},
            colorbar=dict(title="Monthly Profit ($)"),
        ))
        fig_heat.add_scatter(x=[f"{pct_owned}%"], y=[f"{int(utilization*100)}%"],
                             mode="markers",
                             marker=dict(size=14, color="white", symbol="x", line=dict(width=2)),
                             name="Current", showlegend=True)
        fig_heat.update_layout(
            title="Monthly Profit — white X marks current settings",
            xaxis_title="% Owned", yaxis_title="Utilization",
            template="plotly_dark", height=480,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Tornado
        st.markdown("**Tornado Chart — Impact of ±20% Change in Key Inputs**")
        base_profit = m["profit"]
        tornado_inputs = {
            "Customer Billing Rate":  "customer_billing_rate",
            "Utilization":            "utilization",
            "On-Demand Price/hr":     "on_demand_cost_hr",
            "Blended Rental Price/hr":"rental_price_hr",
            "Purchase Price":         "purchase_price",
            "Colo Cost/GPU":          "colo_cost_per_gpu",
            "Electricity Rate":       "electricity_rate",
        }
        tornado_rows = []
        for label, key in tornado_inputs.items():
            base_val = params[key]
            lo = calc_metrics(**{**params, key: base_val * 0.8})["profit"] - base_profit
            hi = calc_metrics(**{**params, key: base_val * 1.2})["profit"] - base_profit
            tornado_rows.append({"label": label, "lo": lo, "hi": hi,
                                  "spread": max(abs(lo), abs(hi))})
        tornado_rows.sort(key=lambda x: x["spread"])

        fig_torn = go.Figure()
        for row in tornado_rows:
            fig_torn.add_trace(go.Bar(y=[row["label"]], x=[row["lo"]], orientation="h",
                                      marker_color="#F44336", showlegend=False))
            fig_torn.add_trace(go.Bar(y=[row["label"]], x=[row["hi"]], orientation="h",
                                      marker_color="#4CAF50", showlegend=False))
        fig_torn.add_vline(x=0, line_color="white", line_width=1)
        fig_torn.update_layout(
            title="Change in Monthly Profit from ±20% Input Variation  (red = −20%, green = +20%)",
            xaxis_title="Δ Monthly Profit ($)", template="plotly_dark",
            barmode="overlay", height=400,
        )
        st.plotly_chart(fig_torn, use_container_width=True)

    # ════════════════════════════════════════
    # TAB 3 — RISK
    # ════════════════════════════════════════
    with tab3:
        st.subheader("Risk Assessment")

        r1, r2, r3, r4, r5, r6 = st.columns(6)
        r1.metric("Break-even Util.", be_display)
        r2.metric("Cushion vs. Break-even",
                  f"{cushion:+.1f}pp" if be_pct <= 100 else "N/A")
        r3.metric(f"Max Exposure ({time_horizon}mo)", f"${m['max_exposure'] * time_horizon:,.0f}",
                  help=f"Total fixed cost over {time_horizon} months at 0% utilization — owned (depr+opex) + rented fees")
        r4.metric("Capital at Risk",  f"${m['capital']:,.0f}",
                  help="Total purchase price of owned GPUs")
        payback_display = f"{m['payback']:.1f} mo" if m["payback"] != float("inf") else "N/A"
        r5.metric("Payback Period", payback_display)
        roi = (m["annual_profit"] / m["capital"] * 100) if m["capital"] > 0 else None
        r6.metric("Annual ROI (owned)",
                  f"{roi:.1f}%" if roi is not None else "N/A")

        st.divider()

        # Maximum Exposure breakdown
        st.markdown("**Maximum Exposure Breakdown (at 0% Utilization)**")
        exp_col1, exp_col2 = st.columns([1, 2])
        with exp_col1:
            exp_items = []
            if m["owned_cost"] > 0:
                exp_items.append(("Owned (Depr+Opex)", m["owned_cost"], "#2196F3"))
            if m["rented_cost"] > 0:
                exp_items.append(("Rented (Blended Monthly)", m["rented_cost"], "#9C27B0"))
            if exp_items:
                el, ev, ec = zip(*exp_items)
                fig_exp = go.Figure(go.Pie(labels=el, values=ev, hole=0.5,
                                           marker_colors=ec))
                fig_exp.update_layout(
                    title=f"Lifetime Total: ${m['max_exposure'] * time_horizon:,.0f}  (${m['max_exposure']:,.0f}/mo × {time_horizon}mo)",
                    template="plotly_dark", height=300,
                    legend=dict(orientation="h", y=-0.2),
                    margin=dict(t=40, b=60),
                )
                st.plotly_chart(fig_exp, use_container_width=True)

        with exp_col2:
            # Max exposure vs % owned (rented constant)
            me_rows = []
            for own in np.arange(0, 105, 5):
                rent = min(100 - own, pct_rented)
                od   = 100 - own - rent
                mm = calc_metrics(**{**params, "pct_owned": own/100,
                                    "pct_rented": rent/100, "pct_on_demand": od/100})
                me_rows.append({"own": own, "exposure": mm["max_exposure"],
                                "owned_part": mm["owned_cost"],
                                "rented_part": mm["rented_cost"]})
            df_me = pd.DataFrame(me_rows)

            fig_me = go.Figure()
            fig_me.add_trace(go.Bar(x=df_me["own"], y=df_me["owned_part"],
                                    name="Owned Cost", marker_color="#2196F3"))
            fig_me.add_trace(go.Bar(x=df_me["own"], y=df_me["rented_part"],
                                    name="Rented Cost", marker_color="#9C27B0"))
            fig_me.add_vline(x=pct_owned, line_dash="dot", line_color="white",
                             annotation_text=f"Now: {pct_owned}%")
            fig_me.update_layout(
                title="Maximum Exposure vs. % Owned",
                xaxis_title="% Owned", yaxis_title="Monthly Exposure ($)",
                barmode="stack", template="plotly_dark", height=300,
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_me, use_container_width=True)

        st.divider()

        # Downside scenarios
        st.subheader("Downside Scenario Analysis")
        scenarios = [
            ("Bull Case (+20% demand, +10% price)",  min(utilization*1.2, 1.0), customer_billing_rate*1.10),
            ("Base Case",                             utilization,               customer_billing_rate),
            ("Bear — Demand −20%",                   utilization*0.8,           customer_billing_rate),
            ("Bear — Price −20%",                    utilization,               customer_billing_rate*0.80),
            ("Stress — Demand −30%, Price −15%",     utilization*0.7,           customer_billing_rate*0.85),
            ("Worst Case — Demand −50%, Price −30%", utilization*0.5,           customer_billing_rate*0.70),
        ]
        rows = []
        for name, u, r in scenarios:
            mm = calc_metrics(**{**params, "utilization": u, "customer_billing_rate": r})
            be = mm["breakeven"] * 100
            rows.append({
                "Scenario":        name,
                "Utilization":     f"{u*100:.0f}%",
                "Billing Rate":    f"${r:.2f}/hr",
                "Monthly Revenue": f"${mm['revenue']:,.0f}",
                "Monthly Profit":  f"${mm['profit']:,.0f}",
                "Margin":          f"{mm['margin']:.1f}%",
                f"Max Exposure ({time_horizon}mo)": f"${mm['max_exposure'] * time_horizon:,.0f}",
                "Break-even":      f"{be:.1f}%" if be <= 100 else "Never",
                "✓":               "✅" if mm["profit"] > 0 else "❌",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()

        # Cumulative P&L
        st.subheader(f"Cumulative P&L Over {time_horizon} Months")
        months        = np.arange(1, time_horizon + 1)
        cum_profit    = months * m["profit"]
        cum_net_capex = cum_profit - m["capital"]

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=months, y=cum_profit,
                                     name="Cumulative Profit (ex-CapEx)",
                                     line=dict(color="#4CAF50", width=2),
                                     fill="tozeroy", fillcolor="rgba(76,175,80,0.08)"))
        fig_cum.add_trace(go.Scatter(x=months, y=cum_net_capex,
                                     name="Cumulative Profit (incl. CapEx outlay)",
                                     line=dict(color="#2196F3", width=2, dash="dash")))
        fig_cum.add_hline(y=0, line_color="white", line_dash="dot")
        if m["profit"] > 0 and m["capital"] > 0:
            pb = m["capital"] / m["profit"]
            if pb <= time_horizon:
                fig_cum.add_vline(x=pb, line_dash="dash", line_color="#FF9800",
                                   annotation_text=f"Payback: mo {pb:.1f}")
        fig_cum.update_layout(
            xaxis_title="Month", yaxis_title="Cumulative Profit ($)",
            template="plotly_dark", height=420,
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    # ════════════════════════════════════════
    # TAB 4 — SCENARIOS
    # ════════════════════════════════════════
    with tab4:
        st.subheader("Scenario Comparison")
        st.caption("Compare procurement strategies across profitability, risk, and exposure.")

        PRESETS = {
            "All Owned (CapEx-heavy)":       (100,  0,  0),
            "All Rented (OpEx)":             (  0,100,  0),
            "Asset-Light (all on-demand)":   (  0,  0,100),
            "Balanced Mix (50/30/20)":       ( 50, 30, 20),
            "Base-load + Burst (60/20/20)":  ( 60, 20, 20),
            "Rental + Overflow (0/80/20)":   (  0, 80, 20),
            "Majority On-Demand (20/20/60)": ( 20, 20, 60),
            "Current Settings":              (pct_owned, pct_rented, pct_on_demand),
        }

        selected = st.multiselect(
            "Select scenarios",
            list(PRESETS.keys()),
            default=["All Owned (CapEx-heavy)", "Balanced Mix (50/30/20)",
                     "Asset-Light (all on-demand)", "Rental + Overflow (0/80/20)", "Current Settings"],
        )

        if not selected:
            st.info("Select at least one scenario above.")
        else:
            comp_rows = []
            for name in selected:
                own, rnt, od = PRESETS[name]
                mm = calc_metrics(**{**params, "pct_owned": own/100,
                                    "pct_rented": rnt/100, "pct_on_demand": od/100})
                be = mm["breakeven"] * 100
                comp_rows.append({
                    "Scenario":        name,
                    "Mix (O/R/OD)":    f"{own}/{rnt}/{od}",
                    "Monthly Profit":  mm["profit"],
                    "Gross Margin":    mm["margin"],
                    "Break-even Util.":be,
                    "Max Exposure":    mm["max_exposure"] * time_horizon,
                    "Capital at Risk": mm["capital"],
                    "Payback (mo)":    mm["payback"],
                })
            df_comp = pd.DataFrame(comp_rows)

            df_display = df_comp.copy()
            df_display["Monthly Profit"]   = df_display["Monthly Profit"].apply(lambda x: f"${x:,.0f}")
            df_display["Gross Margin"]     = df_display["Gross Margin"].apply(lambda x: f"{x:.1f}%")
            df_display["Break-even Util."] = df_display["Break-even Util."].apply(
                lambda x: f"{x:.1f}%" if x <= 100 else "Never")
            df_display = df_display.rename(columns={"Max Exposure": f"Max Exposure ({time_horizon}mo)"})
            df_display[f"Max Exposure ({time_horizon}mo)"] = df_display[f"Max Exposure ({time_horizon}mo)"].apply(lambda x: f"${x:,.0f}")
            df_display["Capital at Risk"]  = df_display["Capital at Risk"].apply(lambda x: f"${x:,.0f}")
            df_display["Payback (mo)"]     = df_display["Payback (mo)"].apply(
                lambda x: f"{x:.1f}" if x != float("inf") else "N/A")
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)

            with col1:
                fig_bar = go.Figure(go.Bar(
                    x=df_comp["Scenario"],
                    y=df_comp["Monthly Profit"],
                    marker_color=["#4CAF50" if p >= 0 else "#F44336"
                                  for p in df_comp["Monthly Profit"]],
                ))
                fig_bar.add_hline(y=0, line_color="white", line_dash="dot")
                fig_bar.update_layout(title="Monthly Profit by Scenario",
                                       yaxis_title="Monthly Profit ($)",
                                       template="plotly_dark", height=380,
                                       xaxis_tickangle=-25)
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                fig_exp = go.Figure()
                fig_exp.add_trace(go.Bar(
                    x=df_comp["Scenario"],
                    y=df_comp["Max Exposure"],
                    marker_color="#FF9800", name=f"Max Exposure ({time_horizon}mo)",
                ))
                fig_exp.add_trace(go.Bar(
                    x=df_comp["Scenario"],
                    y=df_comp["Capital at Risk"],
                    marker_color="#F44336", name="Capital at Risk",
                    visible="legendonly",
                ))
                fig_exp.update_layout(title=f"Maximum Exposure ({time_horizon}mo) by Scenario",
                                       yaxis_title="Lifetime Exposure ($)",
                                       template="plotly_dark", height=380,
                                       xaxis_tickangle=-25,
                                       legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_exp, use_container_width=True)

            # Utilization curves
            st.markdown("**Monthly Profit vs. Utilization — All Selected Scenarios**")
            util_arr = np.linspace(0, 1.0, 101)
            colors   = px.colors.qualitative.Set2
            fig_comp = go.Figure()

            for idx, name in enumerate(selected):
                own, rnt, od = PRESETS[name]
                p_tuple = tuple(sorted({
                    **params,
                    "pct_owned": own/100, "pct_rented": rnt/100, "pct_on_demand": od/100,
                }.items()))
                df_u = util_curve(p_tuple, util_arr)
                fig_comp.add_trace(go.Scatter(
                    x=df_u["u"], y=df_u["profit"],
                    name=name, line=dict(color=colors[idx % len(colors)], width=2),
                ))

            fig_comp.add_hline(y=0, line_color="white", line_dash="dot")
            fig_comp.add_vline(x=utilization * 100, line_dash="dot", line_color="gray",
                                annotation_text=f"Expected: {utilization*100:.0f}%")
            fig_comp.update_layout(
                xaxis_title="Utilization (%)", yaxis_title="Monthly Profit ($)",
                template="plotly_dark", height=460,
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_comp, use_container_width=True)


main()
