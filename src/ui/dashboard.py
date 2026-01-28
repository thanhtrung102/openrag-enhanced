"""
OpenRAG Enhanced - Evaluation Dashboard

Streamlit dashboard for monitoring RAG system performance:
- Real-time metrics visualization
- Hallucination rate tracking
- Model comparison charts
- Query analysis
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import httpx

# Page configuration
st.set_page_config(
    page_title="OpenRAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stMetric {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# Database connection
@st.cache_resource
def get_db_connection():
    """Get database connection."""
    import psycopg2

    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            database=os.getenv("POSTGRES_DB", "openrag_eval"),
            user=os.getenv("POSTGRES_USER", "openrag"),
            password=os.getenv("POSTGRES_PASSWORD", "openrag_secret"),
        )
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


def fetch_data(query: str) -> pd.DataFrame:
    """Execute query and return DataFrame."""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()

    try:
        return pd.read_sql(query, conn)
    except Exception as e:
        st.warning(f"Query failed: {e}")
        return pd.DataFrame()


def generate_sample_data():
    """Generate sample data for demo when no DB available."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')

    data = {
        'date': dates,
        'faithfulness': np.random.uniform(3.5, 4.8, 30),
        'relevance': np.random.uniform(3.2, 4.5, 30),
        'hallucination_rate': np.random.uniform(0.05, 0.25, 30),
        'hit_rate_5': np.random.uniform(0.7, 0.92, 30),
        'mrr': np.random.uniform(0.55, 0.75, 30),
        'latency_p95': np.random.uniform(80, 200, 30),
        'queries': np.random.randint(100, 500, 30),
    }

    return pd.DataFrame(data)


# Sidebar
with st.sidebar:
    st.title("📊 OpenRAG Dashboard")
    st.markdown("---")

    # Date range filter
    st.subheader("Filters")
    date_range = st.date_input(
        "Date Range",
        value=(datetime.now() - timedelta(days=7), datetime.now()),
        max_value=datetime.now(),
    )

    # Model filter
    models = ["All", "mistral-7b", "gpt-4", "claude-3-sonnet", "LUCIE-7B"]
    selected_model = st.selectbox("Model", models)

    # Retrieval method filter
    methods = ["All", "dense", "hybrid", "hybrid_rerank", "graph"]
    selected_method = st.selectbox("Retrieval Method", methods)

    st.markdown("---")

    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("OpenRAG Enhanced v0.1.0")
    st.caption("Linagora AI Department")


# Main content
st.title("📊 OpenRAG Evaluation Dashboard")
st.markdown("Real-time monitoring of RAG system performance and hallucination detection.")

# Try to fetch real data, fall back to sample
try:
    metrics_df = fetch_data("""
        SELECT * FROM daily_evaluation_summary
        ORDER BY date DESC
        LIMIT 30
    """)
    if metrics_df.empty:
        metrics_df = generate_sample_data()
        st.info("📌 Showing sample data. Connect to database for live metrics.")
except Exception:
    metrics_df = generate_sample_data()
    st.info("📌 Showing sample data. Connect to database for live metrics.")


# Top metrics row
st.header("Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    current_faith = metrics_df['faithfulness'].iloc[-1] if not metrics_df.empty else 4.2
    prev_faith = metrics_df['faithfulness'].iloc[-2] if len(metrics_df) > 1 else current_faith
    delta = current_faith - prev_faith
    st.metric(
        "Avg Faithfulness",
        f"{current_faith:.2f}/5",
        delta=f"{delta:+.2f}",
        delta_color="normal"
    )

with col2:
    current_hall = metrics_df['hallucination_rate'].iloc[-1] if not metrics_df.empty else 0.12
    prev_hall = metrics_df['hallucination_rate'].iloc[-2] if len(metrics_df) > 1 else current_hall
    delta = current_hall - prev_hall
    st.metric(
        "Hallucination Rate",
        f"{current_hall:.1%}",
        delta=f"{delta:+.1%}",
        delta_color="inverse"
    )

with col3:
    current_hit = metrics_df['hit_rate_5'].iloc[-1] if not metrics_df.empty else 0.85
    prev_hit = metrics_df['hit_rate_5'].iloc[-2] if len(metrics_df) > 1 else current_hit
    delta = current_hit - prev_hit
    st.metric(
        "Hit Rate @5",
        f"{current_hit:.1%}",
        delta=f"{delta:+.1%}"
    )

with col4:
    current_mrr = metrics_df['mrr'].iloc[-1] if not metrics_df.empty else 0.68
    prev_mrr = metrics_df['mrr'].iloc[-2] if len(metrics_df) > 1 else current_mrr
    delta = current_mrr - prev_mrr
    st.metric(
        "MRR",
        f"{current_mrr:.2f}",
        delta=f"{delta:+.2f}"
    )

with col5:
    current_lat = metrics_df['latency_p95'].iloc[-1] if not metrics_df.empty else 125
    prev_lat = metrics_df['latency_p95'].iloc[-2] if len(metrics_df) > 1 else current_lat
    delta = current_lat - prev_lat
    st.metric(
        "Latency P95",
        f"{current_lat:.0f}ms",
        delta=f"{delta:+.0f}ms",
        delta_color="inverse"
    )

st.markdown("---")

# Charts row 1
st.header("Performance Trends")

col1, col2 = st.columns(2)

with col1:
    # Quality metrics over time
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=metrics_df['date'],
        y=metrics_df['faithfulness'],
        name='Faithfulness',
        mode='lines+markers',
        line=dict(color='#2ecc71', width=2),
    ))

    fig.add_trace(go.Scatter(
        x=metrics_df['date'],
        y=metrics_df['relevance'],
        name='Relevance',
        mode='lines+markers',
        line=dict(color='#3498db', width=2),
    ))

    fig.update_layout(
        title='Quality Metrics Over Time',
        xaxis_title='Date',
        yaxis_title='Score (0-5)',
        yaxis=dict(range=[0, 5]),
        legend=dict(x=0, y=1.1, orientation='h'),
        hovermode='x unified',
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Hallucination rate over time
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=metrics_df['date'],
        y=metrics_df['hallucination_rate'] * 100,
        name='Hallucination Rate',
        mode='lines+markers',
        fill='tozeroy',
        line=dict(color='#e74c3c', width=2),
        fillcolor='rgba(231, 76, 60, 0.2)',
    ))

    # Add threshold line
    fig.add_hline(
        y=20,
        line_dash="dash",
        line_color="orange",
        annotation_text="Warning Threshold (20%)",
    )

    fig.update_layout(
        title='Hallucination Rate Over Time',
        xaxis_title='Date',
        yaxis_title='Rate (%)',
        yaxis=dict(range=[0, 40]),
        hovermode='x unified',
    )

    st.plotly_chart(fig, use_container_width=True)

# Charts row 2
col1, col2 = st.columns(2)

with col1:
    # Retrieval metrics
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=metrics_df['date'],
        y=metrics_df['hit_rate_5'] * 100,
        name='Hit Rate @5',
        mode='lines+markers',
        line=dict(color='#9b59b6', width=2),
    ))

    fig.add_trace(go.Scatter(
        x=metrics_df['date'],
        y=metrics_df['mrr'] * 100,
        name='MRR',
        mode='lines+markers',
        line=dict(color='#f39c12', width=2),
    ))

    fig.update_layout(
        title='Retrieval Metrics Over Time',
        xaxis_title='Date',
        yaxis_title='Score (%)',
        yaxis=dict(range=[0, 100]),
        legend=dict(x=0, y=1.1, orientation='h'),
        hovermode='x unified',
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Latency distribution
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=metrics_df['date'],
        y=metrics_df['latency_p95'],
        name='P95 Latency',
        mode='lines+markers',
        line=dict(color='#1abc9c', width=2),
    ))

    # Add SLA line
    fig.add_hline(
        y=150,
        line_dash="dash",
        line_color="red",
        annotation_text="SLA Target (150ms)",
    )

    fig.update_layout(
        title='Latency P95 Over Time',
        xaxis_title='Date',
        yaxis_title='Latency (ms)',
        hovermode='x unified',
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Model comparison section
st.header("Model Comparison")

# Sample model comparison data
model_data = pd.DataFrame({
    'Model': ['mistral-7b', 'gpt-4', 'claude-3-sonnet', 'LUCIE-7B'],
    'Faithfulness': [4.1, 4.6, 4.5, 3.9],
    'Relevance': [4.0, 4.5, 4.4, 3.8],
    'Hallucination Rate': [0.15, 0.08, 0.09, 0.18],
    'Latency (ms)': [85, 450, 320, 95],
    'Cost ($/1K)': [0.02, 0.15, 0.08, 0.01],
})

col1, col2 = st.columns(2)

with col1:
    # Radar chart for quality comparison
    categories = ['Faithfulness', 'Relevance', 'Completeness', 'Coherence', 'Speed']

    fig = go.Figure()

    for i, model in enumerate(model_data['Model']):
        fig.add_trace(go.Scatterpolar(
            r=[
                model_data.loc[i, 'Faithfulness'],
                model_data.loc[i, 'Relevance'],
                4.0 + np.random.uniform(-0.3, 0.3),
                4.2 + np.random.uniform(-0.3, 0.3),
                5 - (model_data.loc[i, 'Latency (ms)'] / 100),
            ],
            theta=categories,
            fill='toself',
            name=model,
        ))

    fig.update_layout(
        title='Model Quality Comparison',
        polar=dict(radialaxis=dict(range=[0, 5])),
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Bar chart for hallucination rates
    fig = px.bar(
        model_data,
        x='Model',
        y='Hallucination Rate',
        color='Model',
        title='Hallucination Rate by Model',
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.update_layout(
        yaxis_tickformat='.0%',
        showlegend=False,
    )

    # Add threshold line
    fig.add_hline(y=0.15, line_dash="dash", line_color="red",
                  annotation_text="Acceptable Threshold")

    st.plotly_chart(fig, use_container_width=True)

# Model comparison table
st.subheader("Detailed Model Comparison")
st.dataframe(
    model_data.style.format({
        'Faithfulness': '{:.2f}',
        'Relevance': '{:.2f}',
        'Hallucination Rate': '{:.1%}',
        'Latency (ms)': '{:.0f}',
        'Cost ($/1K)': '${:.3f}',
    }).background_gradient(
        subset=['Faithfulness', 'Relevance'],
        cmap='Greens',
    ).background_gradient(
        subset=['Hallucination Rate', 'Latency (ms)', 'Cost ($/1K)'],
        cmap='Reds_r',
    ),
    use_container_width=True,
)

st.markdown("---")

# Recent queries with issues
st.header("Queries Requiring Review")

# Sample problematic queries
problem_queries = pd.DataFrame({
    'Timestamp': pd.date_range(end=datetime.now(), periods=5, freq='H'),
    'Query': [
        "How do I implement async transformers?",
        "What is the best GPU for training LLMs?",
        "Explain the difference between RAG and fine-tuning",
        "How to reduce hallucinations in GPT models?",
        "What are MCP servers in Claude?",
    ],
    'Hallucination Score': [0.72, 0.65, 0.58, 0.55, 0.52],
    'Confidence': [0.28, 0.35, 0.42, 0.45, 0.48],
    'Model': ['mistral-7b', 'gpt-4', 'claude-3-sonnet', 'mistral-7b', 'LUCIE-7B'],
    'Status': ['🔴 High Risk', '🟡 Medium Risk', '🟡 Medium Risk', '🟢 Reviewed', '🟡 Medium Risk'],
})

st.dataframe(
    problem_queries.style.format({
        'Hallucination Score': '{:.2f}',
        'Confidence': '{:.2f}',
    }).background_gradient(
        subset=['Hallucination Score'],
        cmap='Reds',
    ),
    use_container_width=True,
)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>OpenRAG Enhanced Evaluation Dashboard | Linagora AI Department</p>
    <p>Last updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
