#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:17:57 2025

@author: deveshdembla
"""

import pandas as pd
import matplotlib.pyplot as plt


file_path = r"/Users/deveshdembla/Documents/MVO_Assignment/CPI_allitemsv1.xlsx"   
data = pd.read_excel(file_path, parse_dates=True, index_col=0)


# Define high CPI threshold (e.g., top quartile)
high_cpi_threshold = data['All items'].quantile(0.75)

# Classify periods as High CPI or Normal CPI
data['CPI Period'] = ['High CPI' if cpi > high_cpi_threshold else 'Normal CPI' for cpi in data['All items']]

# Calculate average returns in each CPI period
avg_returns = data.groupby('CPI Period')[['USA LARGE GROWTH', 'USA MINIMUM VOLATILITY', 'USA  VALUE', 'USA QUALITY']].mean()

# Plot the results
avg_returns.plot(kind='bar', figsize=(8, 6), color=['#1f77b4', '#ff7f0e', 'pink', 'grey'])
plt.title('Performance During High CPI vs Normal CPI Periods', fontsize=14)
plt.ylabel('Average Monthly Return', fontsize=12)
plt.xlabel('CPI Period', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Asset Class', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
