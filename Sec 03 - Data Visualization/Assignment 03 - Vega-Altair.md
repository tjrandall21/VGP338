# Assignment 03: Declarative Visualization with Vega-Altair

**Course:** AI & Machine Learning Foundations  
**Section:** 03 - Data Visualization  
**Reference Notebook:** `04 - Vega-Altair.ipynb`  
**Estimated Time to Complete:** ~30 Minutes  
**Deliverables:** Submit a completed Jupyter Notebook (`assignment_03_altair_<your_name>.ipynb`) or Python script (`assignment_03_altair_<your_name>.py`).

---

## Overview & Objectives

Unlike imperative libraries like Matplotlib where you specify step-by-step drawing commands, **Vega-Altair** implements a **declarative grammar of graphics**. In Altair, you specify *what* relationships exist between data columns and visual encoding channels (position, color, shape, size), and the visualization compiler handles the rendering pipeline.

By completing this assignment, you will demonstrate your ability to:
1. Formulate charts using Altair's core syntax: `alt.Chart(data).mark_*().encode(...)`.
2. Apply explicit Altair data types (`:Q` Quantitative, `:N` Nominal, `:O` Ordinal, `:T` Temporal).
3. Compose multi-layered charts using the layering operator (`+`) and concatenation (`|`, `&`).
4. Implement aggregations, sorting, and binning directly within encoding channels.
5. Build interactive visualizations featuring tooltips, legend filtering, and dynamic interval selections (brushing).

---

## Setup

Run the following cell to import libraries and load the World Happiness Report 2024 dataset:

```python
import altair as alt
import pandas as pd

# Load dataset
df = pd.read_csv('data/WHR_2024.csv')

# Verify the first few rows
df.head()
```

---

## Task 1: Core Encodings & Interactive Tooltips (~7 mins)
*Topics: Declarative syntax, Data Types (`:Q`, `:N`), Tooltips, Color Encodings (`Part 1 & 2`)*

Explore the relationship between economic output and life expectancy globally.

### Instructions:
1. Initialize an Altair chart using `df`:
   - Use `mark_circle(size=80, opacity=0.75)`.
   - Map `GDP_per_Capita:Q` to `x`, setting a custom title `'GDP per Capita'`.
   - Map `Healthy_Life_Expectancy:Q` to `y`, setting a custom title `'Healthy Life Expectancy'`.
   - Map `Continent:N` to `color` with a qualitative color scale (e.g., `scale=alt.Scale(scheme='category10')`).
   - Add rich tooltips displaying `Country:N`, `Continent:N`, `Happiness_Score:Q`, `GDP_per_Capita:Q`, and `Rank:O`.
2. Set the chart title to `'Healthy Life Expectancy vs GDP per Capita (2024)'`.
3. Set the chart dimensions: `.properties(width=650, height=400)`.
4. Make the chart pannable and zoomable by appending `.interactive()`.

---

## Task 2: Ranked Bar Chart with In-Chart Sorting (~7 mins)
*Topics: `mark_bar()`, Sorting, Horizontal bars, Transformations (`Part 2`)*

Identify the top 15 happiest nations and visualize their scores.

### Instructions:
1. Filter the dataset to obtain the top 15 countries by rank:
   ```python
   top_15 = df.sort_values(by='Rank').head(15)
   ```
2. Build an Altair horizontal bar chart:
   - Use `mark_bar(cornerRadiusEnd=4)`.
   - Map `Happiness_Score:Q` to `x`, setting the title `'Happiness Score'` and domain scale starting from 6.0 (`scale=alt.Scale(domain=[6, 8])`).
   - Map `Country:N` to `y`, explicitly sorted descending by x-value (`sort='-x'`, `title='Country'`).
   - Map `Happiness_Score:Q` to `color` using a sequential color scheme (`scale=alt.Scale(scheme='blues')`).
3. Add text labels directly on each bar:
   - Create a secondary mark: `mark_text(align='left', baseline='middle', dx=3)`.
   - Encode `text=alt.Text('Happiness_Score:Q', format='.2f')`.
4. Combine the bars and text labels using the layering operator `+` (`bars + text`).
5. Set the title to `'Top 15 Happiest Countries (2024)'` with `.properties(width=500, height=350)`.

---

## Task 3: Layered Visualizations & Statistical Baselines (~8 mins)
*Topics: Layering (`+`), `mark_rule()`, `transform_regression()`, Mean aggregation (`Part 3`)*

Combine a scatter plot with statistical reference marks to highlight outliers.

### Instructions:
1. Create a scatter plot `points`:
   - Data: `df`.
   - Mark: `mark_point(filled=True, size=60, opacity=0.7)`.
   - X: `Social_Support:Q`.
   - Y: `Happiness_Score:Q`.
   - Color: `Continent:N`.
2. Create a statistical regression trendline `reg_line`:
   - Start from `alt.Chart(df)`.
   - Transform: `.transform_regression('Social_Support', 'Happiness_Score')`.
   - Mark: `mark_line(color='black', strokeDash=[4, 4], size=2)`.
   - Encode: `x='Social_Support:Q'`, `y='Happiness_Score:Q'`.
3. Create a global mean reference rule `mean_rule`:
   - Start from `alt.Chart(df)`.
   - Mark: `mark_rule(color='red', strokeWidth=1.5)`.
   - Encode: `y='mean(Happiness_Score):Q'`.
4. Layer all three components together:
   ```python
   layered_chart = (points + reg_line + mean_rule).properties(
       width=600,
       height=400,
       title='Happiness vs Social Support with Regression & Global Mean'
   )
   layered_chart
   ```

---

## Task 4: Interactive Multi-View & Linked Selections (~8 mins)
*Topics: `alt.selection_point()`, `alt.selection_interval()`, Linked views (`&` or `|`) (`Part 5 & 6`)*

Create a coordinated dashboard where selecting a region dynamically updates or highlights related plots.

### Instructions:
1. Define an interactive point selection bound to the chart's legend:
   ```python
   continent_selection = alt.selection_point(fields=['Continent'], bind='legend')
   ```
2. Build an interactive scatter plot (`scatter`):
   - X: `Freedom:Q`.
   - Y: `Happiness_Score:Q`.
   - Color condition: Use `alt.condition(continent_selection, alt.Color('Continent:N'), alt.value('lightgray'))`.
   - Opacity condition: Use `alt.condition(continent_selection, alt.value(0.9), alt.value(0.15))`.
   - Tooltip: `['Country:N', 'Continent:N', 'Happiness_Score:Q']`.
   - Add parameter: `.add_params(continent_selection)`.
   - Properties: `.properties(width=350, height=350, title='Freedom vs Happiness')`.
3. Build a companion distribution chart (`distribution`):
   - Mark: `mark_bar()`.
   - X: `Happiness_Score:Q` binned (`bin=alt.Bin(maxbins=15)`).
   - Y: `count():Q`.
   - Color condition: `alt.condition(continent_selection, alt.Color('Continent:N'), alt.value('lightgray'))`.
   - Add parameter: `.add_params(continent_selection)`.
   - Properties: `.properties(width=300, height=350, title='Happiness Distribution')`.
4. Combine both charts side-by-side using the horizontal concatenation operator:
   ```python
   dashboard = (scatter | distribution)
   dashboard
   ```
5. Test the interactivity: Click any continent name in the legend and verify that the other continents fade out dynamically in both charts.

---

## Submission Checklist

- [ ] All charts render cleanly using Altair declarative syntax.
- [ ] Explicit data types (`:Q`, `:N`, `:O`) are utilized across all encoding channels.
- [ ] Layering (`+`) and concatenation (`|`) operators are correctly executed.
- [ ] Legend interactivity in Task 4 functions as expected when clicked.
- [ ] Save and submit `assignment_03_altair_<your_name>.ipynb`.
