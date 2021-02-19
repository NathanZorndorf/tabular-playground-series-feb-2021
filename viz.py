import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_histograms(data, cols, ncols=5):
    nrows = math.ceil(len(cols) / ncols)
    colors = plt.rcParams["axes.prop_cycle"]()
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows*3))
    for i, col in enumerate(cols):
        ax = axes.flatten()[i]
        c = next(colors)["color"]
        sns.histplot(data[col], ax=ax, color=c)
        ax.set_xlabel(col)
        
def cor_plot(data):
    # Compute the correlation matrix
    corr = data.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})