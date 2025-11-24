# Medical Data Visualizer

A Python script that cleans and visualizes medical examination data to explore relationships between cardiovascular disease and various health factors like body measurements, blood markers, and lifestyle choices.

## What It Does

This visualizer creates two main plots:

1. **Categorical Plot (Catplot)** - Shows the distribution of health metrics (cholesterol, glucose, smoking, alcohol, activity, and weight) split between patients with and without cardiovascular disease

2. **Correlation Heatmap** - Displays how different health variables correlate with each other, helping identify which factors tend to appear together

## Requirements

```bash
pip install pandas seaborn matplotlib numpy
```

## Data Format

The script expects `medical_examination.csv` with columns like:
- age
- height (cm)
- weight (kg)
- gender
- ap_hi (systolic blood pressure)
- ap_lo (diastolic blood pressure)
- cholesterol (1: normal, 2: above normal, 3: well above normal)
- gluc (glucose levels, same scale as cholesterol)
- smoke (0: no, 1: yes)
- alco (alcohol intake, 0: no, 1: yes)
- active (physical activity, 0: no, 1: yes)
- cardio (cardiovascular disease, 0: no, 1: yes)

## Usage

```python
from medical_data_visualizer import draw_cat_plot, draw_heat_map

# Create the categorical plot
draw_cat_plot()  # Saves as 'catplot.png'

# Create the correlation heatmap
draw_heat_map()  # Saves as 'heatmap.png'
```

## What Happens Behind the Scenes

### Categorical Plot
1. **Calculates BMI** and adds an `overweight` column (1 if BMI > 25, 0 otherwise)
2. **Normalizes data** so 0 = good and 1 = bad for all metrics:
   - Cholesterol and glucose: 1 (normal) becomes 0, 2-3 (elevated) becomes 1
3. **Melts the data** into long format for easier visualization
4. **Groups and counts** occurrences of each health metric
5. **Creates side-by-side bar plots** comparing people with and without cardiovascular disease

### Correlation Heatmap
1. **Applies the same cleaning** as the catplot
2. **Filters out questionable data**:
   - Removes cases where diastolic > systolic pressure (physically impossible)
   - Excludes extreme outliers in height and weight (bottom/top 2.5%)
3. **Calculates correlations** between all variables
4. **Shows only the lower triangle** (since the matrix is symmetric)
5. **Displays values** with color-coding to make patterns obvious

## Output

Both functions save PNG images and return the figure objects:
- `catplot.png` - Side-by-side comparison of health factors
- `heatmap.png` - Correlation matrix with values

## Insights You Can Find

- Which health factors are most associated with cardiovascular disease?
- Do lifestyle choices (smoking, alcohol, activity) correlate with blood markers?
- How do different health metrics relate to each other?
- Are certain factors clustered together?

## Data Cleaning Notes

The heatmap function is pretty strict about data quality. It removes:
- Impossible blood pressure readings
- Extremely tall/short people (outliers)
- Extremely heavy/light people (outliers)

This ensures the correlations reflect real patterns rather than data entry errors.

## Customization

Feel free to tweak:
- The BMI threshold for overweight (currently 25)
- The quantile cutoffs for outliers (currently 2.5% and 97.5%)
- Plot aesthetics like colors, size, and formatting
- Which variables to include in the analysis

