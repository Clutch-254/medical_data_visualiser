import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def draw_cat_plot():
    """
    Cleans and visualizes medical data using a Catplot (Categorical Plot)
    for various health metrics.
    """
    # 1. Import the data
    df = pd.read_csv('medical_examination.csv')

    # 2. Add 'overweight' column
    # Calculate BMI: weight (kg) / [height (m)]^2
    # Height is in cm, so convert to meters (divide by 100)
    bmi = df['weight'] / (df['height'] / 100)**2
    df['overweight'] = (bmi > 25).astype(int) # 1 if overweight (>25), 0 otherwise

    # 3. Normalize data by making 0 always good and 1 always bad
    # Cholesterol: 1 is normal, 2 is above normal, 3 is well above normal.
    # Set 1 (normal) to 0, and 2, 3 (bad) to 1.
    df['cholesterol'] = (df['cholesterol'] > 1).astype(int)

    # Glucose: 1 is normal, 2 is above normal, 3 is well above normal.
    # Set 1 (normal) to 0, and 2, 3 (bad) to 1.
    df['gluc'] = (df['gluc'] > 1).astype(int)

    # 4. Convert data to long format for easier plotting (Melt the DataFrame)
    # The 'value' column will contain the normalized values (0 or 1)
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 5. Group and reformat the data to draw the catplot
    # Calculate the count of each 'value' (0 or 1) for each 'variable', grouped by 'cardio'
    df_cat = (
        df_cat.groupby(['cardio', 'variable', 'value'])['value']
        .count()
        .reset_index(name='total') # name the count column 'total'
    )

    # 6. Draw the Catplot
    fig = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='bar',
        height=5,
        aspect=1.2 # Make the plots slightly wider
    )

    # Get the figure object
    fig = fig.fig

    # Save image and return fig (don't modify this part)
    fig.savefig('catplot.png')
    return fig

# ---

def draw_heat_map():
    """
    Cleans the data and visualizes the correlation matrix using a Heatmap.
    """
    # 1. Import the data (or reuse the initial cleaning steps)
    df = pd.read_csv('medical_examination.csv')
    
    # Apply initial cleaning/normalization as done in draw_cat_plot()
    df['overweight'] = (df['weight'] / (df['height'] / 100)**2 > 25).astype(int)
    df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
    df['gluc'] = (df['gluc'] > 1).astype(int)

    # 2. Filter out bad data points (Data Cleaning)
    # Diastolic pressure (ap_lo) should be less than Systolic pressure (ap_hi)
    # Height should be in the bottom 2.5% and top 97.5%
    # Weight should be in the bottom 2.5% and top 97.5%
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 3. Calculate the correlation matrix
    corr = df_heat.corr()

    # 4. Create a mask for the upper triangle
    # We only need to show half the matrix since it's symmetric
    mask = np.triu(corr)

    # 5. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # 6. Draw the heatmap with seaborn
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,        # Show the correlation values
        fmt='.1f',         # Format to one decimal place
        linewidths=.5,     # Lines between cells
        cbar_kws={'shrink': 0.5}, # Shrink the color bar
        center=0,          # Center the color map at 0
        vmax=0.32,         # Set max color limit for better visualization
        square=True,       # Ensure cells are square
        cbar=True,         # Display the color bar
        ax=ax
    )

    # Save image and return fig (don't modify this part)
    fig.savefig('heatmap.png')
    return fig
