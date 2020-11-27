import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import os

def perform_and_render_analysis(df):
    if not os.path.exists('renders'):
        os.makedirs('renders')
    
    # Correlation diagram
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]]
    fig_corr = plt.figure(num=None,figsize=(14,14), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(df.corr(), fignum=fig_corr.number)
    plt.xticks(range(len(df.corr().columns)),df.corr().columns, fontsize=6, rotation=90)
    plt.yticks(range(len(df.corr().columns)),df.corr().columns, fontsize=6)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    title = "Correlation Matrix"
    plt.title(title, fontsize=16)
    plt.savefig('./renders/'+"our_dataset_correlation")

    # Scatter and density plots
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=(14,14), diagonal='kde')
    plt.suptitle('Scatter and Density Plot')
    plt.savefig('./renders/'+"our_scatter_plots")
