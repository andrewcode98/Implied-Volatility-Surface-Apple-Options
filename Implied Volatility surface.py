# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:05:03 2024

@author: andre
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from scipy.optimize import minimize
import plotly.graph_objects as go


df = pd.read_csv(r"C:\Users\andre\OneDrive\Desktop\Apple_options_ 2016_to_2020.csv")
df['C_IV'].replace(r'^\s*$', np.nan, regex=True, inplace=True)
df = df.dropna()


# Select necessary columns

df = df[['C_IV', 'QUOTE_DATE', 'EXPIRE_DATE', 'UNDERLYING_LAST', 'STRIKE']]
# Drop rows where 'C_IV' is zero
df['C_IV'] = pd.to_numeric(df['C_IV'], errors='coerce')
df = df[df['C_IV'] > 0]

# Rename columns for easier reference
df.columns = ['Implied Volatility', 'Quote Date', 'Expiry Date', 'Underlying Asset', 'Strike Price']




# Convert date columns to datetime
df['Quote Date'] = pd.to_datetime(df['Quote Date'])
df['Expiry Date'] = pd.to_datetime(df['Expiry Date'])
df = df[df['Quote Date'].isin(pd.to_datetime(['2016-01-04']))]
#df = df[df['Quote Date'].isin(pd.to_datetime(['2016-01-05']))]
#df = df[df['Quote Date'].isin(pd.to_datetime(['2016-01-06']))]
#df = df[df['Quote Date'].isin(pd.to_datetime(['2016-01-07']))]

# Calculatelog-strike
df['Log-Moneyness'] = np.log(df['Strike Price']/df['Underlying Asset'])
df['Log-Strike'] = np.log(df['Strike Price'])


def calculate_trading_days(start_date, end_date):
    # Generate a range of business days between start_date and end_date
    return np.busday_count(start_date.date(), end_date.date())
# Apply the function to calculate trading days passed for each expiry date / 252 trading days
df['Time to Expiry'] = df.apply(lambda row: calculate_trading_days(row['Quote Date'], row['Expiry Date']) / 252, axis=1)

implied_vol_surf_df = df[['Log-Moneyness', 'Time to Expiry', 'Log-Strike', 'Implied Volatility']]


implied_vol_surf_df = implied_vol_surf_df[
    (implied_vol_surf_df['Time to Expiry'] <= 1)]

implied_vol_surf_df = implied_vol_surf_df[
    (implied_vol_surf_df['Implied Volatility'] >= 0.2)]


def model_function(k,t, params):
    a, b, rho, mu, sigma = params
    return (a + b * (rho*(k - mu) + np.sqrt((k-mu)**2 + sigma**2)))/t

def objective_function(params, k_data, t_data, y_data):
    predictions = model_function(k_data, t_data, params)
    residuals = np.abs(y_data - predictions) / y_data
    mape = np.mean(residuals)   
    return mape

def symbolic_regression_function_Q1(M,T):
    return ((M - np.sqrt(M * (M / T))) * -0.26628348) + 0.24132709

initial_guess = [10, 0, 0, 0, 0]  # Initial guess for a, b, rho, mu, sigma
bounds = [(None, None), (0, None), (-1, 1), (None,None) , (0.001,None) ]  
# Example constraint: 
def constraint(params):
    return params[0] + (params[1] * params[4]) * np.sqrt(1 - params[2]**2)
constraints = [{'type': 'ineq', 'fun': constraint}]

k_data = df['Log-Moneyness'].values
k1_data = df['Log-Strike'].values
t_data = df['Time to Expiry'].values
y_data = df['Implied Volatility'].values

result = minimize(objective_function, initial_guess, args=(k1_data, t_data, y_data), method='SLSQP', bounds=bounds)
# Print the result
print("Optimal parameters:", result.x)
print("Objective function value at the optimal solution:", result.fun)


# # 3D Plot
implied_vol_surf_df['Implied Volatility'] = pd.to_numeric(implied_vol_surf_df['Implied Volatility'], errors='coerce')




opt_params = result.x
# Create a grid of points
k1_range = np.linspace(implied_vol_surf_df['Log-Strike'].min(), implied_vol_surf_df['Log-Strike'].max(), 100)
k_range = np.linspace(implied_vol_surf_df['Log-Moneyness'].min(), implied_vol_surf_df['Log-Moneyness'].max(), 100)
t_range = np.linspace(implied_vol_surf_df['Time to Expiry'].min(), implied_vol_surf_df['Time to Expiry'].max(), 100)
K, T = np.meshgrid(k_range, t_range)
K1, T = np.meshgrid(k1_range,t_range)

# Evaluate the surface function
Z = model_function(K1, T, opt_params)
Z_symbolic = symbolic_regression_function_Q1(K, T)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the scatter plot
scatter = ax.scatter(
    implied_vol_surf_df['Log-Moneyness'],
    implied_vol_surf_df['Time to Expiry'],
    implied_vol_surf_df['Implied Volatility'],
    c=implied_vol_surf_df['Implied Volatility'],  # Color by 'Implied Volatility'
    cmap='viridis',  # Choose a colormap
    s=30  # Size of the points
)

# Plot the surface
#SIV_surface = ax.plot_surface(K, T, Z, cmap='viridis', alpha=0.6)
surface_symbolic = ax.plot_surface(K, T, Z_symbolic, cmap='plasma', alpha=0.6)

# Label the axes
ax.set_xlabel('Log-Moneyness')
ax.set_ylabel('Time to Expiry')
ax.set_zlabel('Implied Volatility')

# Set plot title
ax.set_title('3D Scatter Plot with Surface of Implied Volatility')

ax.view_init(elev=30, azim=45)
# Show plot
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the scatter plot
scatter = ax.scatter(
    implied_vol_surf_df['Log-Strike'],
    implied_vol_surf_df['Time to Expiry'],
    implied_vol_surf_df['Implied Volatility'],
    c=implied_vol_surf_df['Implied Volatility'],  # Color by 'Implied Volatility'
    cmap='viridis',  # Choose a colormap
    s=30  # Size of the points
)

# Plot the surface
SIV_surface = ax.plot_surface(K1, T, Z, cmap='viridis', alpha=0.6)


# Label the axes
ax.set_xlabel('Log-Strike')
ax.set_ylabel('Time to Expiry')
ax.set_zlabel('Implied Volatility')

# Set plot title
ax.set_title('3D Scatter Plot with Surface of Implied Volatility')


ax.view_init(elev=30, azim=45)
# Show plot
plt.show()



# Create a Plotly figure
fig = go.Figure()

# Add the scatter plot for actual data points
fig.add_trace(go.Scatter3d(
    x=implied_vol_surf_df['Log-Moneyness'],
    y=implied_vol_surf_df['Time to Expiry'],
    z=implied_vol_surf_df['Implied Volatility'],
    mode='markers',
    marker=dict(
        size=3,
        color=implied_vol_surf_df['Implied Volatility'],  # Color by 'Implied Volatility'
        colorscale='Viridis',
        opacity=0.8
    ),
    name='Actual Data'
))




# Add the surface plot for the symbolic regression function
fig.add_trace(go.Surface(
    x=K,
    y=T,
    z=Z_symbolic,
    colorscale='Plasma',
    opacity=0.6,
    showscale=False,
    name='Symbolic Surface'
))

# Update layout for better visualization
fig.update_layout(
    scene=dict(
        xaxis_title='Log-Moneyness',
        yaxis_title='Time to Expiry',
        zaxis_title='Implied Volatility'
    ),
    title='Implied Volatility Surface',
    autosize=True
)

# Save the plot as an HTML file
fig_html_path = "volatility_surface_symbolic.html"
fig.write_html(fig_html_path)

fig_html_path

# Create a Plotly figure
fig = go.Figure()

# Add the scatter plot for actual data points
fig.add_trace(go.Scatter3d(
    x=implied_vol_surf_df['Log-Strike'],
    y=implied_vol_surf_df['Time to Expiry'],
    z=implied_vol_surf_df['Implied Volatility'],
    mode='markers',
    marker=dict(
        size=3,
        color=implied_vol_surf_df['Implied Volatility'],  # Color by 'Implied Volatility'
        colorscale='Viridis',
        opacity=0.8
    ),
    name='Actual Data'
))




# Add the surface plot for the symbolic regression function
fig.add_trace(go.Surface(
    x=K1,
    y=T,
    z=Z,
    colorscale='Viridis',
    opacity=0.6,
    showscale=False,
    name='Symbolic Surface'
))

# Update layout for better visualization
fig.update_layout(
    scene=dict(
        xaxis_title='Log-Strike',
        yaxis_title='Time to Expiry',
        zaxis_title='Implied Volatility'
    ),
    title='Implied Volatility Surface',
    autosize=True
)

# Save the plot as an HTML file
fig_html_path = "volatility_surface_SIV.html"
fig.write_html(fig_html_path)

fig_html_path
