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
    
    st.subheader("Objective")
    st.markdown(
    """
    
    >The investor has a portfolio of **$100 million** and seeks to optimize its allocation across 
    the following four identified asset classes:

    >- **US Large Cap Value**
    >- **US Large Cap Growth**
    >- **US Large Cap Quality**
    >- **US Large Cap Minimum Volatility**

    >The portfolio's performance is benchmarked against the **MSCI USA Index**, which serves as a 
    proxy for these asset classes. Historical performance data for the MSCI USA factor benchmarks 
    has been sourced from publicly available datasets.

    >The objective is to construct a portfolio that maximizes risk-adjusted returns using traditional 
    MVO, taking into account expected returns, volatility, and correlations among the assets. 

    >Additionally, the app will analyze the portfolio's active risk relative to the MSCI USA Index and 
    provide relevant risk analytics to better understand portfolio positioning and performance.
    """
    )
    
    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    # Calculate Returns
    fig, ax = plt.subplots(figsize=(8, 6))
    
    returns = data.pct_change().dropna()
    st.subheader("Correlation Matrix")
    correlation_matrix = returns.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="Blues", fmt=".2f")
    plt.gcf().set_facecolor('#D3D3D3')
    ax.set_xticks(np.arange(len(correlation_matrix.columns)) + 0.5)  # Position ticks in the center of cells
    ax.set_yticks(np.arange(len(correlation_matrix.index)) + 0.5)
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right", fontsize=10, color="black", weight="bold")
    ax.set_yticklabels(correlation_matrix.index, rotation=0, fontsize=10, color="black", weight="bold")
    
    

    st.pyplot(fig)

    # Step 2: Mean-Variance Optimization
    expected_returns = mean_historical_return(data, frequency=12, compounding=True)
    cov_matrix = CovarianceShrinkage(data, frequency=12).ledoit_wolf()
    
    st.sidebar.header("Select Optimization Method")
    optimization_method = st.sidebar.selectbox(
        "The traditional MVO minimizes risk for a certain target return - Efficient Return but we have a couple of other options available",
        ["Efficient Return", "Max Sharpe", "Minimum Volatility"]
    )

    # User input for weight bounds
    st.sidebar.header("Customize Weight Bounds")
    lower_bound = st.sidebar.slider("Lower Bound", 0.0, 0.25, 0.0, 0.01)
    upper_bound = st.sidebar.slider("Upper Bound", 0.0, 1.0, 0.5, 0.01)
    
    st.sidebar.header("Set the risk-free rate")
    rfr = st.sidebar.slider("Risk Free Rate", 0.0, 0.06, 0.03, 0.01)
    
    st.sidebar.header("Set the target return")
    target_return = st.sidebar.slider("Target Return", 0.05, 0.11, 0.08, 0.01)
    
    # Add dropdown for selecting optimization method
    

    
    if lower_bound >= upper_bound:
        st.error("Lower bound must be less than upper bound")   
    else:
        ef = EfficientFrontier(expected_returns, cov_matrix, weight_bounds=(lower_bound, upper_bound))
        
        if optimization_method == "Max Sharpe":
            weights = ef.max_sharpe(risk_free_rate=rfr)
            optimization_label = "Max Sharpe"
        elif optimization_method == "Efficient Return":
            weights = ef.efficient_return(target_return)
            optimization_label = f"Efficient Return ({target_return*100}%)"
        else:  # Minimum Volatility
            weights = ef.min_volatility()
            optimization_label = "Minimum Volatility"
        
        
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
        labels=[label if weight > 0 else "" for label, weight in cleaned_weights.items()],
        autopct="%.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops={"edgecolor": "k", "linewidth": 1.5},  # Add borders
        textprops={"fontsize": 10, "weight": "bold"}  # Text size for better readability
    )
    
    # Style the percentage text
    plt.setp(autotexts, size=9, weight="bold", color="black")  
    plt.setp(texts, size=10)
    
    # Add a title with a contrasting color
    ax.set_title("Optimized Portfolio Allocation", fontsize=14, weight="bold", color="#333333")
    
    # Remove axes for a cleaner look
    ax.axis("equal")  # Ensure the pie chart is circular
    plt.show()
    
    # Display the chart in Streamlit
    st.pyplot(fig)
    
    
    # Efficient Frontier Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#D3D3D3')  # Light grey background for the figure
    ax.set_facecolor('#eaeaea')         # Slightly darker grey for the chart area

    # Plot the efficient frontier
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)

    # Highlight the selected portfolio on the frontier
    ax.scatter(portfolio_volatility, expected_return, marker="*", s=100, c="red", label=f"{optimization_label} Portfolio")

    # Chart styling
    ax.set_title("Efficient Frontier", fontsize=14, weight="bold", color="#333333")
    ax.set_xlabel("Volatility (Standard Deviation)", fontsize=12, weight="bold", color="#333333")
    ax.set_ylabel("Expected Return", fontsize=12, weight="bold", color="#333333")
    ax.legend()
    plt.tight_layout()

    # Display the plot in Streamlit
    st.subheader("Efficient Frontier")
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
    st.subheader("Benchmark relative metrics")
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
        
        #Additional Analytics
        
        returns['Portfolio'] = portfolio_returns
        returns['MSCI USA'] = benchmark_returns
        
        mrfr = rfr / 12  # Monthly risk-free rate (3% annualized)
        excess_returns = returns['Portfolio'] - mrfr

        #Sortino Ratio
        downside_returns = excess_returns[excess_returns < 0]
        downside_risk = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(12)  # Annualized
        sortino_ratio = (excess_returns.mean() * 12) / downside_risk

        #Beta
        covariance = np.cov(returns['Portfolio'], returns['MSCI USA'])[0, 1]
        benchmark_variance = np.var(returns['MSCI USA'])
        beta = covariance / benchmark_variance

        #Portfolio Alpha
        portfolio_return = returns['Portfolio'].mean() * 12  # Annualized portfolio return
        benchmark_return = returns['MSCI USA'].mean() * 12  # Annualized benchmark return
        alpha = portfolio_return - (rfr + beta * (benchmark_return - rfr))

        #Information Ratio
        information_ratio = (portfolio_return - benchmark_return) / active_risk
        

        st.write(f"Active Risk (Tracking Error): {active_risk:.2%}")
        st.write(f"Sortino Ratio: {sortino_ratio:.2f}")
        st.write(f"Beta: {beta:.2f}")
        st.write(f"Alpha: {alpha:.2%}")        
        st.write(f"Information Ratio: {information_ratio:.2f}")
        
        
        
        
        
        
        