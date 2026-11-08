# Capstone Project: Predicting Steam Game Popularity

**Section:** 05 - ML - Regression  **Estimated time:** ~6 hours  **Deliverable:** one Jupyter notebook

## Overview

You're a data analyst at an indie game studio. The studio wants to know **what drives a game's popularity on Steam**. Build a regression model that predicts **how many reviews a game receives**, which the industry uses as a rough stand-in for sales.

You'll have to explore, clean, engineer features, build a base model, then improve, validate and evaluate it.

## Dataset

**Steam Games Dataset** by fronkongames (Kaggle): about 126k games with 39 columns
🔗 https://www.kaggle.com/datasets/fronkongames/steam-games-dataset (use only `games.csv`, ~400 MB)

**⚠️ Known issue:** the CSV header merges two column names into `"DiscountDLC count"`, which shifts every column after it by one. Load the file like this:

```python
cols = pd.read_csv(path, nrows=0).columns.tolist()
i = cols.index("DiscountDLC count")
cols[i:i+1] = ["Discount", "DLC count"]
df = pd.read_csv(path, header=None, skiprows=1, names=cols)
```

**Target:** `total_reviews = Positive + Negative`. Model **`log1p(total_reviews)`** (see Part 1).

---

## Part 1: Data Exploration, Visualization & Outliers (~1.5 h)

1. **Load and inspect** the data: `shape`, `info()`, `describe()`, missing values. Confirm that `AppID` is unique, then explain why `Name` has duplicates (demos, playtests, editions).
2. Write a short **data dictionary** for the columns you'll use, such as `Price`, `Discount`, `DLC count`, `Estimated owners`, `Peak CCU`, `Achievements`, `Supported languages`, `Genres`, `Categories`, `Tags`, `Windows/Mac/Linux`, `Average/Median playtime forever` (in minutes) and `Release date`. Note the type of each one (numeric, boolean, text range, list stored as text).
3. **"Zero means missing":** show that many zeros aren't real values:
   - `Estimated owners = "0 - 0"`
   - `Metacritic score = 0`
   - `User score = 0`
   - empty `Movies`, `Reviews` and `Score rank` columns

   Decide which columns to drop.
4. **Define the scope:** keep games that are already released, have owners other than `"0 - 0"`, and have **at least 10 reviews**. Report how many games remain.
5. **Visualize:**
   - `total_reviews` as a raw histogram and as `log1p` (why is the log version better?)
   - the distribution of `Price` and the share of free-to-play games
   - the number of games released per year
   - the top 10 genres (bar chart)
6. **Outliers:**
   - Find **suspicious values** using domain knowledge. Look up and comment on:
     - the $999.98 game
     - games with thousands of achievements or DLCs
     - `Average playtime forever` above 3 million minutes (that's over 6 years)
   - Apply the **IQR method** to at least three numeric columns (try both raw and log scale). Show box plots **before and after**.
   - Explain each removal and how many rows it cost.

> ✅ **Checkpoint:** a clean DataFrame and a short summary of your cleaning decisions.

## Part 2: Feature Engineering & Feature Relevance (~1.5 h)

1. **Create features from the existing columns** (at least 6), for example:

   | New feature | Built from |
   | --- | --- |
   | `release_year`, `game_age_years` | `Release date` (use `pd.to_datetime`) |
   | `owners_mid` | midpoint of the `Estimated owners` range, e.g. `"20000 - 50000"` → 35000 |
   | `original_price` | `Price / (1 - Discount/100)`, because `Price` is the current, possibly discounted, price |
   | `is_free` | `Price == 0` |
   | `n_languages` | `Supported languages`: count the commas (`ast.literal_eval` fails on `"K'iche"`) |
   | `n_tags`, `n_screenshots` | `Tags`, `Screenshots` (comma-separated) |
   | `n_platforms` | `Windows + Mac + Linux` |
   | `has_metacritic` | `Metacritic score > 0` |
   | genre/category flags | `Genres.str.get_dummies(sep=",")`; keep the ~10 most common genres and categories such as `Single-player`, `Multi-player`, `Steam Achievements`, `Full controller support`, `Co-op` |

2. **Log-transform** skewed counts (`DLC count`, `Achievements`, `Peak CCU`, playtime, `owners_mid`) with `np.log1p`. Show one before/after histogram.
3. **Data leakage:** explain why `Positive`, `Negative`, `Recommendations`, `User score`, `Score rank` and anything computed from them (such as the % positive) **must not** be used as features.
4. **Two feature groups.** Split your features into:
   - **Pre-launch:** what the studio knows from its store page (price, genres, languages, achievements, platforms, and so on)
   - **Post-launch:** signals measured after release (`owners_mid`, `Peak CCU`, playtime)
5. **Relevance and complexity:**
   - Draw a correlation heatmap and rank the features by their correlation with `log1p(total_reviews)`.
   - Draw scatter plots of the top 3 features against the target. Are the relationships linear?
   - Find **multicollinearity**, for example average vs. median playtime, `owners_mid` vs. `Peak CCU`, or `is_free` vs. `Price`. Decide what to keep.
   - Compare average review counts across genres (box plot). Which genres attract more reviews?

> ✅ **Checkpoint:** two feature lists (pre-launch and full), plus `X` and `y`, with a written justification.

## Part 3: Base Model (~1 h)

1. Split the data into **train / validation / test** sets (60/20/20, fixed `random_state`). Keep the test set untouched until Part 4.
2. Build **three baselines** on the log target:
   - a "dumb" model that always predicts the mean training value
   - a **simple linear regression** on your single strongest feature
   - a **multiple linear regression** on the **pre-launch** features only (in a `Pipeline` with `StandardScaler`)
3. Report **MAE, RMSE and R²** for train and validation in one table.
4. Print the coefficients of the pre-launch model. Which store-page decisions (languages, achievements, price, genre) are linked to more reviews? Remember that this is correlation, not causation.

> ✅ **Checkpoint:** a baseline to beat. As a rough guide, the pre-launch linear model reaches R² ≈ 0.5.

## Part 4: Final Model, Training & Validation (~2 h)

1. **Improve the model:**
   - Add the **post-launch** features. How much does R² change, and what does that tell the studio?
   - Try `PolynomialFeatures` (degree 2), then **Ridge** and **Lasso** with several `alpha` values. Without the outlier cleaning from Part 1, degree 2 can blow up. Check whether that happens.
2. Use the **validation set** to choose the degree and `alpha`. Plot train vs. validation error against complexity, and mark where the model underfits and where it overfits.
3. Run **k-fold cross-validation** (k = 5) on your chosen model and report the mean ± standard deviation of R² and RMSE.
4. Retrain on train + validation, then evaluate **once** on the **test set**:
   - MSE, RMSE, MAE and **R² (goodness-of-fit)** on the log scale
   - the **median absolute error in number of reviews**, after `np.expm1`. Why is the median more informative than the mean here?
   - a comparison table: final model vs. the Part 3 baselines
5. **Residual analysis** (with plots):
   - residuals vs. predicted values
   - a histogram of residuals and a Q-Q plot
   - predicted vs. actual values, with the y = x line
   - Interpret the plots: Is there a pattern? Does the model over- or under-predict hits and very small games? Name 3 games with the largest errors and suggest why they're hard to predict.
6. If you used Lasso, list the features it dropped.

> 💡 A rough guide: the full model with polynomial features and regularization reaches a test R² of about 0.8–0.85 on the log scale.

---

## Deliverables

- **One notebook** (`Capstone_<YourName>.ipynb`) with the four parts as headings. It must run top to bottom. Don't submit the dataset.
- Markdown explanations throughout. **Explain why you made each choice**, not only what the code does.
- A **conclusion** (5–10 sentences) covering:
  - your final test metrics
  - the most important features
  - one practical recommendation for an indie studio, based on the pre-launch model
  - the model's limitations

## Grading

| Part | Weight |
| --- | --- |
| 1: Exploration, visualization & outliers | 25% |
| 2: Feature engineering & relevance | 25% |
| 3: Base model | 15% |
| 4: Final model, validation & residual analysis | 25% |
| Clarity, explanations & code quality | 10% |

**Bonus (+5%):** train your pre-launch multiple linear regression with your own **mini-batch gradient descent** (notebooks 04–05) and compare the result with scikit-learn's.

## Tips

- Use **09 - NYC Taxi Fare Project** as your reference. The workflow is the same; the decisions are yours.
- The file is large. Load it once and save the cleaned DataFrame with `df.to_pickle(...)` to iterate faster.
- Every visualization needs a title, axis labels and a one-line takeaway.
