import plotly.graph_objects as go
import pandas as pd
import streamlit as st


def plot_fraud_probability(prob):
    """
    Creates a gauge chart showing the risk level.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fraud Probability (%)"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if prob > 50 else "green"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "salmon"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}
        }
    ))
    return fig


def plot_feature_importance(model, feature_names):
    """
    Shows which features (Time, In-degree, etc.) were most important.
    """
    # Get feature importance from the model
    importances = model.feature_importances_

    # Create a simple dataframe
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)

    # Simple Bar Chart
    st.bar_chart(df.set_index('Feature'))