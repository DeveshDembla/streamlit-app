#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 13:13:29 2025

@author: deveshdembla
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from pypfopt import plotting

# Streamlit Title
st.title("Portfolio Optimization and Risk Analysis")

# File Upload Section
uploaded_file = st.file_uploader(r"/Users/deveshdembla/Documents/MVO_Assignment/MsciUS_Factors.xlsx"  , type=["xlsx"])

if uploaded_file:
   
    # Step 1: Load Data
    data = pd.read_excel(uploaded_file, parse_dates=True, index_col=0)

    # Data Cleaning
    for column in data.columns:
        data[column] = data[column].replace(",", "", regex=True).astype(float)
        data[column] = pd.to_numeric(data[column], errors='coerce')
    data = data.dropna()

    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    # Calculate Returns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#D3D3D3')
    ax.set_facecolor('#D3D3D3')
    returns = data.pct_change().dropna()
    st.subheader("Correlation Matrix")
    correlation_matrix = returns.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    #plt.xticks(rotation=45, ha="right", fontsize=10, color="white", weight="bold")
    #plt.yticks(rotation=0, fontsize=10, color="white", weight="bold")

    st.pyplot(fig)

    # Step 2: Mean-Variance Optimization
    expected_returns = mean_historical_return(data, frequency=12, compounding=True)
    cov_matrix = CovarianceShrinkage(data, frequency=12).ledoit_wolf()


    # User input for weight bounds
    st.sidebar.header("Customize Weight Bounds")
    lower_bound = st.sidebar.slider("Lower Bound", 0.0, 1.0, 0.2, 0.01)
    upper_bound = st.sidebar.slider("Upper Bound", 0.0, 1.0, 0.5, 0.01)
    
    st.sidebar.header("Set the risk-free rate")
    rfr = st.sidebar.slider("Risk Free Rate", 0.0, 0.06, 0.03, 0.01)
    
    if lower_bound >= upper_bound:
        st.error("Lower bound must be less than upper bound")   
    else:
        ef = EfficientFrontier(expected_returns, cov_matrix, weight_bounds=(lower_bound, upper_bound))
        weights = ef.max_sharpe(risk_free_rate=rfr)
        cleaned_weights = ef.clean_weights()
        expected_return, portfolio_volatility, sharpe_ratio = ef.portfolio_performance(risk_free_rate=rfr)

    st.subheader("Optimized Portfolio Weights")
    st.write(cleaned_weights)

    # Pie Chart for Portfolio Weights
    colors = plt.cm.tab20c(range(len(cleaned_weights)))

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Set a background color for the figure
    fig.patch.set_facecolor('#D3D3D3')  # Light grey background
    
    # Set a background color for the axes
    ax.set_facecolor('#eaeaea')  # Slightly darker grey for the chart area
    
    # Customize the pie chart
    wedges, texts, autotexts = ax.pie(
        cleaned_weights.values(),
        labels=cleaned_weights.keys(),
        autopct="%.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops={"edgecolor": "k", "linewidth": 1.5},  # Add borders
        textprops={"fontsize": 10, "weight": "bold"}  # Text size for better readability
    )
    
    # Style the percentage text
    plt.setp(autotexts, size=9, weight="bold", color="white")  
    plt.setp(texts, size=10)
    
    # Add a title with a contrasting color
    ax.set_title("Optimized Portfolio Allocation", fontsize=14, weight="bold", color="#333333")
    
    # Remove axes for a cleaner look
    ax.axis("equal")  # Ensure the pie chart is circular
    
    # Display the chart in Streamlit
    st.pyplot(fig)

    # Portfolio Metrics
    st.subheader("Portfolio Performance")
    st.write(f"Expected Annual Return: {expected_return:.2%}")
    st.write(f"Annual Volatility (Risk): {portfolio_volatility:.2%}")
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # Step 3: Drawdown and Additional Metrics
    st.subheader("Additional Metrics and Drawdown Analysis")
    portfolio_returns = returns[list(cleaned_weights.keys())].dot(list(cleaned_weights.values()))
    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    st.write(f"Maximum Drawdown: {max_drawdown:.2%}")

    # Drawdown Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(drawdowns, label="Portfolio Drawdown", color='red')
    plt.axhline(0, linestyle='--', color='black')
    plt.title("Portfolio Drawdown")
    plt.ylabel("Drawdown")
    plt.xlabel("Time")
    plt.legend()
    st.pyplot(fig)

    # Active Risk and Additional Metrics
    st.subheader("Risk Metrics")
    benchmark_file = st.file_uploader("Upload Benchmark Data (Excel)", type=["xlsx"])
    if benchmark_file:
        bmk_data = pd.read_excel(benchmark_file, parse_dates=True, index_col=0)
        for column in bmk_data.columns:
            bmk_data[column] = bmk_data[column].replace(",", "", regex=True).astype(float)
            bmk_data[column] = pd.to_numeric(bmk_data[column], errors='coerce')
        bmk_data = bmk_data.dropna()
        benchmark_returns = bmk_data['USA Standard (Large+Mid Cap)'].pct_change().dropna()

        active_returns = portfolio_returns - benchmark_returns
        active_risk = np.std(active_returns) * np.sqrt(12)

        st.write(f"Active Risk (Tracking Error): {active_risk:.2%}")

