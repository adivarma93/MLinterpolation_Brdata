#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import matplotlib.cm as cm

plt.rcParams.update({'font.size': 12, 'text.usetex': False})

# Grid setup
Nt = int(144 * 3 / 2.0)  # sample  numbers.
Np = int(88 * 3)
t = np.linspace(-np.pi / 2, np.pi / 2, Nt)
p = np.linspace(-np.pi, np.pi, Np)
lon2 = p * 180. / np.pi
lat2 = t * 180. / np.pi
lon_grid, lat_grid = np.meshgrid(lon2, lat2)

# Load data
n_time_steps = 2000
time_steps = np.arange(1, n_time_steps + 1)
data_time_series = []

for ii in range(1, n_time_steps + 1):
    jj = f'{ii:04d}'
    filename = f'br_surf12x_{jj}.dat'
    if os.path.exists(filename):
        S = np.loadtxt(filename)
        data = S.reshape(Nt, Np, order='F').copy()
        A1 = data[:, :Np // 7]
        A2 = data[:, Np // 7:Np]  # Rearrangement of matrices. this should be done based on data input
        data = np.concatenate((A2, A1), axis=1)
        data_time_series.append(data)
    else:
        print(f"File {filename} not found, skipping.")
        data_time_series.append(np.zeros((Nt, Np)))

data_time_series = np.stack(data_time_series, axis=0)
print(f"Loaded data shape: {data_time_series.shape}")

# Sample for RF
sample_interval =200 ## this can be customized for comparison between the models 
sampled_time_steps = time_steps[::sample_interval]
sampled_data = data_time_series[::sample_interval, :, :]

time_grid_sampled = np.repeat(sampled_time_steps[:, np.newaxis, np.newaxis], Nt, axis=1).repeat(Np, axis=2)
X_sampled = np.stack((time_grid_sampled, 
                      lat_grid[np.newaxis, :, :].repeat(len(sampled_time_steps), axis=0), 
                      lon_grid[np.newaxis, :, :].repeat(len(sampled_time_steps), axis=0)), axis=-1).reshape(-1, 3)
y_sampled = sampled_data.reshape(-1)

time_grid_full = np.repeat(time_steps[:, np.newaxis, np.newaxis], Nt, axis=1).repeat(Np, axis=2)
X_full = np.stack((time_grid_full, 
                   lat_grid[np.newaxis, :, :].repeat(n_time_steps, axis=0), 
                   lon_grid[np.newaxis, :, :].repeat(n_time_steps, axis=0)), axis=-1).reshape(-1, 3)

# Random Forest Interpolation
model = RandomForestRegressor(n_estimators=50, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_sampled, y_sampled)
interpolated_rf = model.predict(X_full).reshape(n_time_steps, Nt, Np)
for t in range(n_time_steps):
    interpolated_rf[t] = gaussian_filter(interpolated_rf[t], sigma=1)

# Linear Interpolation
linear_interp = np.zeros((n_time_steps, Nt, Np))
for i in range(Nt):
    for j in range(Np):
        f = interp1d(sampled_time_steps, sampled_data[:, i, j], kind='linear', fill_value="extrapolate")
        linear_interp[:, i, j] = f(time_steps)

# Cubic Spline Interpolation
cubic_interp = np.zeros((n_time_steps, Nt, Np))
for i in range(Nt):
    for j in range(Np):
        f = interp1d(sampled_time_steps, sampled_data[:, i, j], kind='cubic', fill_value="extrapolate")
        cubic_interp[:, i, j] = f(time_steps)
        
# RMSE for each time step and average
rmse_rf_per_step = np.array([np.sqrt(mean_squared_error(data_time_series[t], interpolated_rf[t])) for t in range(n_time_steps)])
rmse_linear_per_step = np.array([np.sqrt(mean_squared_error(data_time_series[t], linear_interp[t])) for t in range(n_time_steps)])
rmse_cubic_per_step = np.array([np.sqrt(mean_squared_error(data_time_series[t], cubic_interp[t])) for t in range(n_time_steps)])

avg_rmse_rf = np.mean(rmse_rf_per_step)
avg_rmse_linear = np.mean(rmse_linear_per_step)
avg_rmse_cubic = np.mean(rmse_cubic_per_step)

# Quantitative Comparison (Global and Average)
mse_rf = mean_squared_error(data_time_series.ravel(), interpolated_rf.ravel())
rmse_rf = np.sqrt(mse_rf)
mse_linear = mean_squared_error(data_time_series.ravel(), linear_interp.ravel())
rmse_linear = np.sqrt(mse_linear)
mse_cubic = mean_squared_error(data_time_series.ravel(), cubic_interp.ravel())
rmse_cubic = np.sqrt(mse_cubic)

print(f"Random Forest - Global MSE: {mse_rf:.4f}, RMSE: {rmse_rf:.4f}, Average RMSE: {avg_rmse_rf:.4f}")
print(f"Linear Interpolation - Global MSE: {mse_linear:.4f}, RMSE: {rmse_linear:.4f}, Average RMSE: {avg_rmse_linear:.4f}")
print(f"Cubic Spline Interpolation - Global MSE: {mse_cubic:.4f}, RMSE: {rmse_cubic:.4f}, Average RMSE: {avg_rmse_cubic:.4f}")        


# Visualize
time_indices_to_plot = [0+sample_interval//2, n_time_steps // 2+ sample_interval//2, int(n_time_steps-sample_interval*1.5)]  # t=1, t=5, t=9
fig, axes = plt.subplots(4, len(time_indices_to_plot), figsize=(15, 12))  # 4 rows: Original, RF, Linear, Cubic

vmin, vmax = -np.max(np.abs(data_time_series)), np.max(np.abs(data_time_series))
ppp = 0.3
cmap = cm.RdBu_r

for idx, t_idx in enumerate(time_indices_to_plot):
    # Original
    ax1 = axes[0, idx]
    map1 = Basemap(projection='hammer', lon_0=0, resolution='l', ax=ax1)
    map1.drawcoastlines(linewidth=0.5)
    map1.drawparallels([70, 0, -70], dashes=[1, 1], linewidth=0.5, color='k')
    x, y = map1(lon_grid, lat_grid)
    im1 = map1.pcolormesh(x, y, data_time_series[t_idx], cmap=cmap, vmin=ppp * vmin, vmax=ppp * vmax)
    ax1.set_title(f"Original t={time_steps[t_idx]}")

    # Random Forest
    ax2 = axes[1, idx]
    map2 = Basemap(projection='hammer', lon_0=0, resolution='l', ax=ax2)
    map2.drawcoastlines(linewidth=0.5)
    map2.drawparallels([70, 0, -70], dashes=[1, 1], linewidth=0.5, color='k')
    x, y = map2(lon_grid, lat_grid)
    im2 = map2.pcolormesh(x, y, interpolated_rf[t_idx], cmap=cmap, vmin=ppp * vmin, vmax=ppp * vmax)
    ax2.set_title(f"RF Interpolated t={time_steps[t_idx]}")
    
    

    # Linear Interpolation
    ax3 = axes[2, idx]
    map3 = Basemap(projection='hammer', lon_0=0, resolution='l', ax=ax3)
    map3.drawcoastlines(linewidth=0.5)
    map3.drawparallels([70, 0, -70], dashes=[1, 1], linewidth=0.5, color='k')
    x, y = map3(lon_grid, lat_grid)
    im3 = map3.pcolormesh(x, y, linear_interp[t_idx], cmap=cmap, vmin=ppp * vmin, vmax=ppp * vmax)
    ax3.set_title(f"Linear Interpolated t={time_steps[t_idx]}")

    # Cubic Spline Interpolation
    ax4 = axes[3, idx]
    map4 = Basemap(projection='hammer', lon_0=0, resolution='l', ax=ax4)
    map4.drawcoastlines(linewidth=0.5)
    map4.drawparallels([70, 0, -70], dashes=[1, 1], linewidth=0.5, color='k')
    x, y = map4(lon_grid, lat_grid)
    im4 = map4.pcolormesh(x, y, cubic_interp[t_idx], cmap=cmap, vmin=ppp * vmin, vmax=ppp * vmax)
    ax4.set_title(f"Cubic Interpolated t={time_steps[t_idx]}")

#fig.colorbar(im1, ax=axes.ravel().tolist(), orientation='vertical', shrink=0.55)
plt.tight_layout()
plt.show()

# Per-time-step metrics
print("\nPer-Time-Step Metrics:")
for t_idx in time_indices_to_plot:
    mse_rf_t = mean_squared_error(data_time_series[t_idx], interpolated_rf[t_idx])
    mse_linear_t = mean_squared_error(data_time_series[t_idx], linear_interp[t_idx])
    mse_cubic_t = mean_squared_error(data_time_series[t_idx], cubic_interp[t_idx])
    print(f"t={time_steps[t_idx]}: RF MSE={mse_rf_t:.4f}, Linear MSE={mse_linear_t:.4f}, Cubic MSE={mse_cubic_t:.4f}")
