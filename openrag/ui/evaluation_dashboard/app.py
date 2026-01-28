"""
OpenRAG Evaluation Dashboard

Streamlit dashboard for monitoring RAG evaluation metrics:
- Hallucination rate tracking
- Retrieval performance (Hit Rate, MRR, NDCG)
- Model comparison
- Historical trends

Usage:
    streamlit run openrag/ui/evaluation_dashboard/app.py

Or with Docker:
    docker-compose up dashboard
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import asyncio

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
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #007bff;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .danger-card {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .success-card {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# Data Loading Functions
# ============================================

@st.cache_resource
def get_db_connection():
    """Get database connection if available."""
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            database=os.getenv("POSTGRES_DB", "openrag_eval"),
            user=os.getenv("POSTGRES_USER", "openrag"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
        )
        return conn
    except Exception as e:
        st.warning(f"Database connection unavailable: {e}")
        return None


def load_data_from_db(query: str) -> pd.DataFrame:
    """Load data from PostgreSQL."""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()

    try:
        return pd.read_sql(query, conn)
    except Exception as e:
        st.error(f"Query failed: {e}")
        return pd.DataFrame()


def generate_sample_data() -> dict:
    """Generate sample data for demo mode."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')

    return {
        "daily_metrics": pd.DataFrame({
            'date': dates,
            'hit_rate': np.random.uniform(0.75, 0.92, 30),
            'mrr': np.random.uniform(0.60, 0.78, 30),
            'ndcg': np.random.uniform(0.65, 0.82, 30),
            'hallucination_rate': np.random.uniform(0.05, 0.20, 30),
            'avg_hallucination_score': np.random.uniform(0.15, 0.35, 30),
            'queries': np.random.randint(100, 500, 30),
        }),
        "model_comparison": pd.DataFrame({
            'model': ['mistral-7b', 'gpt-4', 'claude-3-sonnet', 'LUCIE-7B'],
            'hit_rate': [0.82, 0.89, 0.87, 0.79],
            'mrr': [0.68, 0.76, 0.74, 0.65],
            'hallucination_rate': [0.15, 0.08, 0.10, 0.18],
            'avg_latency_ms': [85, 450, 320, 95],
            'total_queries': [1500, 800, 1200, 600],
        }),
        "recent_flagged": pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.now(), periods=10, freq='H'),
            'query': [
                "How do transformers handle attention?",
                "What is the capital of France in 2050?",
                "Explain quantum computing basics",
                "Who invented Python programming?",
                "What are the best AI models?",
                "How does RAG work internally?",
                "Compare GPT-4 and Claude",
                "What is machine learning?",
                "Explain neural network layers",
                "How to implement embeddings?",
            ],
            'hallucination_score': np.random.uniform(0.5, 0.9, 10),
            'model': np.random.choice(['mistral-7b', 'gpt-4', 'claude-3-sonnet'], 10),
            'flagged_claims': [
                ["Transformers were invented in 2020"],
                ["Paris will be renamed in 2050"],
                [],
                ["Python was created by Microsoft"],
                ["GPT-5 is currently available"],
                [],
                ["Claude is made by OpenAI"],
                [],
                [],
                ["Embeddings require GPUs"],
            ],
        }),
    }


# ============================================
# Dashboard Components
# ============================================

def render_sidebar():
    """Render sidebar with filters."""
    with st.sidebar:
        st.title("📊 OpenRAG Dashboard")
        st.markdown("---")

        st.subheader("Filters")

        # Date range
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=7), datetime.now()),
            max_value=datetime.now(),
        )

        # Model filter
        models = ["All", "mistral-7b", "gpt-4", "claude-3-sonnet", "LUCIE-7B"]
        selected_model = st.selectbox("Model", models)

        # Retrieval method
        methods = ["All", "dense", "hybrid", "hybrid_rerank"]
        selected_method = st.selectbox("Retrieval Method", methods)

        # Hallucination threshold
        threshold = st.slider(
            "Hallucination Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )

        st.markdown("---")

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.caption("OpenRAG Enhanced v0.1.0")

        return {
            "date_range": date_range,
            "model": selected_model,
            "method": selected_method,
            "threshold": threshold,
        }


def render_metrics_row(data: dict):
    """Render top metrics row."""
    st.header("Key Metrics")

    df = data["daily_metrics"]
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        delta = latest['hit_rate'] - previous['hit_rate']
        st.metric(
            "Hit Rate @5",
            f"{latest['hit_rate']:.1%}",
            delta=f"{delta:+.1%}",
        )

    with col2:
        delta = latest['mrr'] - previous['mrr']
        st.metric(
            "MRR",
            f"{latest['mrr']:.3f}",
            delta=f"{delta:+.3f}",
        )

    with col3:
        delta = latest['ndcg'] - previous['ndcg']
        st.metric(
            "NDCG @10",
            f"{latest['ndcg']:.3f}",
            delta=f"{delta:+.3f}",
        )

    with col4:
        delta = latest['hallucination_rate'] - previous['hallucination_rate']
        st.metric(
            "Hallucination Rate",
            f"{latest['hallucination_rate']:.1%}",
            delta=f"{delta:+.1%}",
            delta_color="inverse",  # Red for increase
        )

    with col5:
        st.metric(
            "Queries Today",
            f"{int(latest['queries']):,}",
        )


def render_trends_charts(data: dict):
    """Render trend charts."""
    st.header("Performance Trends")

    df = data["daily_metrics"]

    col1, col2 = st.columns(2)

    with col1:
        # Retrieval metrics over time
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['date'], y=df['hit_rate'] * 100,
            name='Hit Rate',
            mode='lines+markers',
            line=dict(color='#2ecc71', width=2),
        ))

        fig.add_trace(go.Scatter(
            x=df['date'], y=df['mrr'] * 100,
            name='MRR',
            mode='lines+markers',
            line=dict(color='#3498db', width=2),
        ))

        fig.add_trace(go.Scatter(
            x=df['date'], y=df['ndcg'] * 100,
            name='NDCG',
            mode='lines+markers',
            line=dict(color='#9b59b6', width=2),
        ))

        fig.update_layout(
            title='Retrieval Metrics Over Time',
            xaxis_title='Date',
            yaxis_title='Score (%)',
            yaxis=dict(range=[0, 100]),
            hovermode='x unified',
            legend=dict(x=0, y=1.1, orientation='h'),
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Hallucination rate over time
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['date'], y=df['hallucination_rate'] * 100,
            name='Hallucination Rate',
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color='#e74c3c', width=2),
            fillcolor='rgba(231, 76, 60, 0.2)',
        ))

        # Warning threshold
        fig.add_hline(
            y=15,
            line_dash="dash",
            line_color="orange",
            annotation_text="Warning (15%)",
        )

        # Critical threshold
        fig.add_hline(
            y=25,
            line_dash="dash",
            line_color="red",
            annotation_text="Critical (25%)",
        )

        fig.update_layout(
            title='Hallucination Rate Over Time',
            xaxis_title='Date',
            yaxis_title='Rate (%)',
            yaxis=dict(range=[0, 40]),
            hovermode='x unified',
        )

        st.plotly_chart(fig, use_container_width=True)


def render_model_comparison(data: dict):
    """Render model comparison section."""
    st.header("Model Comparison")

    df = data["model_comparison"]

    col1, col2 = st.columns(2)

    with col1:
        # Radar chart
        categories = ['Hit Rate', 'MRR', 'Low Hallucination', 'Speed']

        fig = go.Figure()

        for _, row in df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[
                    row['hit_rate'] * 5,
                    row['mrr'] * 5,
                    (1 - row['hallucination_rate']) * 5,
                    max(0, 5 - row['avg_latency_ms'] / 100),
                ],
                theta=categories,
                fill='toself',
                name=row['model'],
            ))

        fig.update_layout(
            title='Model Quality Comparison',
            polar=dict(radialaxis=dict(range=[0, 5])),
            showlegend=True,
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Bar chart - hallucination rates
        fig = px.bar(
            df,
            x='model',
            y='hallucination_rate',
            color='model',
            title='Hallucination Rate by Model',
            color_discrete_sequence=px.colors.qualitative.Set2,
        )

        fig.update_layout(
            yaxis_tickformat='.0%',
            showlegend=False,
        )

        fig.add_hline(
            y=0.15,
            line_dash="dash",
            line_color="red",
            annotation_text="Threshold",
        )

        st.plotly_chart(fig, use_container_width=True)

    # Data table
    st.subheader("Detailed Comparison")
    st.dataframe(
        df.style.format({
            'hit_rate': '{:.1%}',
            'mrr': '{:.3f}',
            'hallucination_rate': '{:.1%}',
            'avg_latency_ms': '{:.0f}ms',
            'total_queries': '{:,}',
        }).background_gradient(
            subset=['hit_rate', 'mrr'],
            cmap='Greens',
        ).background_gradient(
            subset=['hallucination_rate'],
            cmap='Reds_r',
        ),
        use_container_width=True,
    )


def render_flagged_queries(data: dict):
    """Render flagged queries section."""
    st.header("🚨 Flagged Queries (Potential Hallucinations)")

    df = data["recent_flagged"]
    df_sorted = df.sort_values('hallucination_score', ascending=False)

    for _, row in df_sorted.head(5).iterrows():
        score = row['hallucination_score']

        if score > 0.7:
            card_class = "danger-card"
            icon = "🔴"
        elif score > 0.5:
            card_class = "warning-card"
            icon = "🟡"
        else:
            card_class = "metric-card"
            icon = "🟢"

        with st.container():
            st.markdown(f"""
            <div class="metric-card {card_class}">
                <strong>{icon} Score: {score:.2f}</strong> | Model: {row['model']}<br>
                <em>Query:</em> {row['query']}<br>
                <em>Flagged Claims:</em> {', '.join(row['flagged_claims']) if row['flagged_claims'] else 'None'}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")


# ============================================
# Main Application
# ============================================

def main():
    """Main dashboard application."""
    st.title("📊 OpenRAG Evaluation Dashboard")
    st.markdown("Real-time monitoring of RAG performance and hallucination detection.")

    # Render sidebar and get filters
    filters = render_sidebar()

    # Load data
    conn = get_db_connection()
    if conn is None:
        st.info("📌 Running in demo mode with sample data. Connect to PostgreSQL for live metrics.")
        data = generate_sample_data()
    else:
        # Load from database
        # TODO: Implement actual queries based on filters
        data = generate_sample_data()

    # Render dashboard sections
    render_metrics_row(data)
    st.markdown("---")

    render_trends_charts(data)
    st.markdown("---")

    render_model_comparison(data)
    st.markdown("---")

    render_flagged_queries(data)

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #666;'>"
        f"OpenRAG Evaluation Dashboard | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        f"</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
