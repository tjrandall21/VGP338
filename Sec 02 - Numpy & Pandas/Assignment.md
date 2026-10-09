# Assignment: Numpy & Pandas Essentials

**Estimated Time:** 1 Hour  
**Prerequisites:** Completion of Section 01 (Python Basics) and Section 02 (Numpy & Pandas).

---

## Part 1: Numpy Basics (20 mins)

In this section, you will use Numpy to manipulate arrays. You can do this in a Jupyter Notebook or a Python script.

1.  **Array Creation & Reshaping**
    *   Create a numpy array containing integers from 0 to 49 (inclusive).
    *   Reshape this array into a matrix with 5 rows and 10 columns.
    *   Print the shape of the resulting matrix to verify.

2.  **Indexing & Slicing**
    *   From the matrix you created in Step 1, slice out the sub-matrix consisting of the **last 2 rows** and **columns 3 through 7** (inclusive of 3, exclusive of 8).
    *   Print this sub-matrix.

3.  **Statistics & Broadcasting**
    *   Create a random 5x5 matrix with values drawn from a standard normal distribution (mean 0, std 1). *Hint: Use `np.random.default_rng` or `np.random.randn`*.
    *   Calculate the **mean** of the entire matrix.
    *   Subtract the mean from every element in the matrix (broadcasting).
    *   Verify that the mean of this new centered matrix is distinctively closer to 0.

4.  **Boolean Masking**
    *   Create an array of 20 random integers between 1 and 100.
    *   Find and print all elements that are greater than 50.
    *   Replace all elements less than 30 with the value -1.

---

## Part 2: Pandas Data Analysis (40 mins)

In this section, you will act as a data analyst for a fictional retail company.

### Setup
Copy the following code to your notebook to generate the initial dataset:

```python
import pandas as pd
import numpy as np

data = {
    'Order_ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'Product': ['Laptop', 'Mouse', 'Monitor', 'Mouse', 'Keyboard', 'Laptop', 'Monitor', 'Keyboard', 'Mouse', 'Laptop'],
    'Category': ['Electronics', 'Accessories', 'Electronics', 'Accessories', 'Accessories', 'Electronics', 'Electronics', 'Accessories', 'Accessories', 'Electronics'],
    'Region': ['North', 'South', 'West', 'North', 'West', 'South', 'North', 'West', 'South', 'North'],
    'Date': ['2024-01-15', '2024-01-16', '2024-01-18', '2024-01-20', '2024-01-21', '2024-02-05', '2024-02-10', '2024-02-15', '2024-02-20', '2024-03-01'],
    'Quantity': [1, 5, 2, np.nan, 3, 1, 2, 4, 10, 2],
    'Unit_Price': [1000, 25, 300, 25, 45, 1000, 300, 45, 25, 1200]
}

df = pd.DataFrame(data)
```

### Tasks

1.  **Data Inspection & Types**
    *   Display the first 5 rows of `df`.
    *   Check the data types (`dtypes`) of all columns.
    *   The `Date` column is currently a generic object (string). Convert it to a proper datetime object using `pd.to_datetime`.

2.  **Handling Missing Data**
    *   Note that `Quantity` has a missing value (`NaN`) for Order 104.
    *   Fill this missing value with the **average (mean)** quantity of the entire dataset.

3.  **Feature Engineering**
    *   Create a new column called `Total_Sales`.
    *   Calculate it as `Quantity * Unit_Price`.

4.  **Filtering & Querying**
    *   Create a new DataFrame called `high_value_sales` that contains only the rows where `Total_Sales` is greater than **500**.
    *   How many such orders are there?

5.  **Grouping & Aggregation**
    *   Group the original `df` by `Category`.
    *   Calculate the **total** `Total_Sales` for each category. Which category generated more revenue?
    *   Group the data by `Region` and find the **average** `Unit_Price` sold in each region.

6.  **String Operations (Bonus)**
    *   The `Region` values are currently capitalized (e.g., 'North').
    *   Convert the `Region` column to be all uppercase (e.g., 'NORTH').

### Submission
Save your notebook or script as `Assignment_02_Name.ipynb` or `.py` and submit it.
