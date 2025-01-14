#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 11:51:40 2025

@author: deveshdembla
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from pypfopt import plotting
import seaborn as sns

# Import Data
file_path = r"/Users/deveshdembla/Documents/MVO_Assignment/MsciUS_Factors.xlsx"  
data = pd.read_excel(file_path, parse_dates=True, index_col=0)

# Convert price data (read as strings) to float
for column in data.columns:
    data[column] = data[column].replace(",", "", regex=True).astype(float)
    data[column] = pd.to_numeric(data[column], errors='coerce')

# drop missing values
data = data.dropna()


rfr = 0.03 # Assumption - 3% per annum risk free rate
 
# Compute Inputs for MVO
# Expected returns (mean historical return), uses geometric mean
expected_returns = mean_historical_return(data, frequency=12, compounding=True)  # Annualized

# Covariance matrix
cov_matrix = CovarianceShrinkage(data, frequency=12).ledoit_wolf()
asset_std_devs = (pd.Series(np.diag(cov_matrix)**0.5, index=expected_returns.index))

asset_assumptions = pd.DataFrame({"Expected Return": expected_returns, "Standard Deviation": asset_std_devs})
print("Asset class assumptions")
print(asset_assumptions.head())
print('\n\n')


target_return = 0.08

# Mean-Variance Optimization unbounded
# We do an unbounded (unconstrained) MVO to highlight one of the issues with traditional MVO.
# Perform unbounded MVO for demonstration

ef = EfficientFrontier(expected_returns, cov_matrix, weight_bounds=(0,1))

#plotting the efficient frontier
fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)

# Find the tangency portfolio
ef.efficient_return(target_return=target_return)
ret_tangent, std_tangent, _ = ef.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Efficient Portfolio")
cleaned_weights = ef.clean_weights()


# Output
ax.set_title("Efficient Frontier")
ax.legend()
plt.tight_layout()
plt.show()


#Bounds are variable
lowerB = 0.1
upperB = 1
#bounded optimization because unbounded results in a single asset portfolio.

ef = EfficientFrontier(expected_returns, cov_matrix, weight_bounds=(lowerB,upperB))


#We can use several different optimisation functions within PyPortfolioOpt.
#The Streamlit App allows users to switch between different functions
#Here the other functions are commented out but can be instead used for optimisation.


# Max Sharpe Ratio portfolio (risk-adjusted return maximization)
#weights = ef.max_sharpe(risk_free_rate=rfr)  # Set an appropriate risk-free rate
#cleaned_weights = ef.clean_weights()

# Minimize total risk
#weights = ef.min_volatility()  
#cleaned_weights = ef.clean_weights()

#Minimize volatility for a target return - Traditional MVO
weights = ef.efficient_return(target_return=target_return)  # Minimize risk for target return
cleaned_weights = ef.clean_weights()



# Portfolio performance
expected_return, portfolio_volatility, sharpe_ratio = ef.portfolio_performance(verbose=True, risk_free_rate = rfr)
print('\n')

# Visualization
# Allocation Pie Chart
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



# Results Summary
print("\nOptimized Portfolio Weights:")
for asset, weight in cleaned_weights.items():
    print(f"{asset}: {weight:.2%}")

print("\nPortfolio Statistics:")
print(f"Expected Annual Return: {expected_return:.2%}")
print(f"Annual Volatility (Risk): {portfolio_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")



# Correlation Matrix Heatmap

# Calculate returns (log or simple returns)
returns = data.pct_change().dropna()
# Correlation matrix
correlation_matrix = returns.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


#Portfolio metrics

#Load Benchmark Data
file_path = r"/Users/deveshdembla/Documents/MVO_Assignment/MsciUSA_Bmk.xlsx"  
bmk_data = pd.read_excel(file_path, parse_dates=True, index_col=0)

#Price data read as strings, convert to float
for column in bmk_data.columns:
    bmk_data[column] = bmk_data[column].replace(",", "", regex=True).astype(float)
    bmk_data[column] = pd.to_numeric(bmk_data[column], errors='coerce')

# Ensure data is cleaned (drop missing values)
bmk_data = bmk_data.dropna()
returns['MSCI USA'] = bmk_data['USA Standard (Large+Mid Cap)'].pct_change().dropna()


#Historic Portfolio Returns 
portfolio_returns = (
    returns[['USA LARGE VALUE', 'USA LARGE GROWTH', 'USA MINIMUM VOLATILITY', 'USA QUALITY']]
    .dot(list(cleaned_weights.values()))
)
returns['Portfolio'] = portfolio_returns


#Calculate active returns (portfolio - benchmark)
active_returns = returns['Portfolio'] - returns['MSCI USA']
active_returns = active_returns.dropna()

active_risk = np.std(active_returns)*np.sqrt(12)

#Drawdown

def calculate_drawdown(return_series):
    cumulative_returns = (1 + return_series).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    return drawdowns, max_drawdown

# Calculate drawdowns and max drawdown
returns['Drawdown'], max_drawdown = calculate_drawdown(portfolio_returns)




#Additional Analytics
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

# Output Results
print('\n')
print("Metrics Summary:")
print(f"Active Risk: {active_risk:.2%}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Sortino Ratio: {sortino_ratio:.2f}")
print(f"Beta: {beta:.2f}")
print(f"Alpha: {alpha:.2%}")
print(f"Information Ratio: {information_ratio:.2f}")



# Visualize Drawdowns
plt.figure(figsize=(10, 6))
plt.plot(returns['Drawdown'], label="Portfolio Drawdown", color='red')
plt.axhline(0, linestyle='--', color='black')
plt.title("Portfolio Drawdown")
plt.ylabel("Drawdown (%)")
plt.xlabel("Time")
plt.legend()
plt.show()


