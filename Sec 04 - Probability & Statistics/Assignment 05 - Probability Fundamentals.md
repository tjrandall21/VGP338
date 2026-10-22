# Assignment 05: Probability Rules, Simulations & Bayesian Classification

**Course:** AI & Machine Learning Foundations  
**Section:** 04 - Probability & Statistics  
**Reference Notebook:** `05 - Probability Fundamentals.ipynb`  
**Estimated Time to Complete:** 20 – 30 Minutes  
**Deliverables:** Submit a completed Jupyter Notebook (`assignment_05_probability_<your_name>.ipynb`) or Python script (`assignment_05_probability_<your_name>.py`).

---

## Overview & Objectives

Probability provides the mathematical framework for reasoning under uncertainty. In machine learning, probability is central to:
- **Model Training:** Maximum likelihood estimation and cross-entropy loss functions.
- **Model Evaluation:** Interpreting classification thresholds, precision, recall, and false positive rates.
- **System Architecture:** Estimating the uptime and failure probabilities of distributed microservices.

By completing this assignment, you will demonstrate your ability to:
1. Simulate and visualize the **Law of Large Numbers (LLN)** in Python.
2. Apply the **Addition Rule** for both mutually exclusive and non-mutually exclusive events.
3. Use the **Multiplication Rule** to evaluate joint probabilities and cloud pipeline reliability.
4. Calculate **Conditional Probabilities** ($P(B|A) = \frac{P(A \cap B)}{P(A)}$) using frequency analysis and understand the **Base Rate Fallacy** in classification models.

---

## Setup & Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set_theme(style='whitegrid')
np.random.seed(42)
```

---

## Task 1: Simulating the Law of Large Numbers (LLN) (~7 mins)
*Reference: Part 2 in `05 - Probability Fundamentals.ipynb`*

The **Law of Large Numbers** states that as the number of independent trials increases, the empirical relative frequency of an event converges to its theoretical probability.

### Instructions:
1. Consider rolling a fair, 6-sided die. The theoretical probability of rolling a $6$ is:
   $$P(\text{Roll } 6) = \frac{1}{6} \approx 0.1667$$
2. Simulate die rolls across expanding sample sizes:
   ```python
   sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 30000]
   empirical_probs = []

   for n in sample_sizes:
       # Generate n random integers between 1 and 6 (inclusive)
       rolls = np.random.randint(1, 7, size=n)
       # Calculate proportion of rolls that landed on 6
       prop_six = np.mean(rolls == 6)
       empirical_probs.append(prop_six)
   ```
3. Plot the convergence trajectory using `plt.subplots(figsize=(9, 5))`:
   - Plot `sample_sizes` on the x-axis (use `ax.set_xscale('log')`) vs. `empirical_probs` on the y-axis (connected with markers: `marker='o'`, `color='tab:blue'`).
   - Add a horizontal dashed line at $y = \frac{1}{6}$ (`color='crimson'`, `linestyle='--'`, `label='Theoretical P(6) = 1/6'`).
   - Set labels, a title (`'Law of Large Numbers Simulation: Die Rolling'`), and an informative legend.
4. **Observation:** State in 1 sentence how the variability of the empirical probability changes as $n$ scales from 10 to 30,000.

---

## Task 2: The Addition Rule in User Funnel Analytics (~6 mins)
*Reference: Part 3 in `05 - Probability Fundamentals.ipynb`*

Recall the rules of addition:
- **Mutually Exclusive Events:** $P(A \cup B) = P(A) + P(B)$
- **Non-Mutually Exclusive Events (General Rule):**  
  $$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

### Scenario:
An AI SaaS product tracks user engagement events during product onboarding:
- Event $A$: User attends a live onboarding webinar: $P(A) = 0.35$
- Event $B$: User activates an API key: $P(B) = 0.40$
- Event $A \cap B$: User attends the webinar **AND** activates an API key: $P(A \cap B) = 0.15$

### Instructions:
1. Are Events $A$ and $B$ mutually exclusive? Explain why or why not in a brief code comment or markdown cell.
2. Calculate the probability that a randomly chosen new user attends the webinar **OR** activates an API key (or both):
   $$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$
3. Calculate the probability that a user **only** activates an API key without attending the webinar:
   $$P(B \text{ only}) = P(B) - P(A \cap B)$$
4. Print all results clearly with percentage formatting.

---

## Task 3: The Multiplication Rule & AI Pipeline Reliability (~6 mins)
*Reference: Part 4 in `05 - Probability Fundamentals.ipynb`*

For **independent events**:
$$P(A \cap B \cap C) = P(A) \times P(B) \times P(C)$$

### Scenario:
A real-time Generative AI system processes user queries through three sequential, independent microservices:
1. **Safety Filter Service ($S_1$):** Success rate $P(S_1) = 0.995$
2. **LLM Inference Server ($S_2$):** Success rate $P(S_2) = 0.982$
3. **Markdown Formatter & Renderer ($S_3$):** Success rate $P(S_3) = 0.999$

### Instructions:
1. Calculate the probability that a query is successfully processed by **all three stages** without any downtime or failures:
   $$P(\text{Complete Success}) = P(S_1 \cap S_2 \cap S_3) = P(S_1) \cdot P(S_2) \cdot P(S_3)$$
2. Calculate the probability that **at least one service fails** during a request:
   $$P(\text{At least one failure}) = 1 - P(\text{Complete Success})$$
3. Print both probabilities formatted to 4 decimal places and as percentages.

---

## Task 4: Conditional Probability & The Base Rate Fallacy in ML (~9 mins)
*Reference: Part 5 in `05 - Probability Fundamentals.ipynb`*

Conditional probability is defined as:
$$P(B \mid A) = \frac{P(A \cap B)}{P(A)}$$

### Scenario:
You are deploying a deep-learning **Phishing & Spam Filter** to protect enterprise inboxes.
- **Population:** $N = 10,000$ incoming emails per day.
- **Base Rate (Prevalence):** Only **1.5%** of incoming emails are genuine phishing/spam threats ($P(\text{Spam}) = 0.015$).
- **Classifier Sensitivity (True Positive Rate):** When an email is spam, the model flags it **96%** of the time ($P(\text{Flagged} \mid \text{Spam}) = 0.96$).
- **False Positive Rate:** When an email is legitimate, the model mistakenly flags it **2%** of the time ($P(\text{Flagged} \mid \text{Legit}) = 0.02$).

### Instructions:
1. Conduct a **Frequency Analysis** on the $10,000$ emails:
   - Total Spam emails: $N_{\text{spam}} = 10000 \times 0.015$
   - Total Legitimate emails: $N_{\text{legit}} = 10000 - N_{\text{spam}}$
   - **True Positives (TP):** Spam emails correctly flagged $= N_{\text{spam}} \times 0.96$
   - **False Negatives (FN):** Spam emails missed $= N_{\text{spam}} - \text{TP}$
   - **False Positives (FP):** Legitimate emails mistakenly flagged $= N_{\text{legit}} \times 0.02$
   - **True Negatives (TN):** Legitimate emails correctly allowed $= N_{\text{legit}} - \text{FP}$
2. Construct and display a Pandas DataFrame summarizing this $2 \times 2$ Confusion Table:
   ```python
   contingency_df = pd.DataFrame({
       'Actual Spam': [tp, fn, total_spam],
       'Actual Legit': [fp, tn, total_legit],
       'Total': [tp + fp, fn + tn, 10000]
   }, index=['Flagged (Alert)', 'Not Flagged (Allowed)', 'Total'])
   print(contingency_df)
   ```
3. Calculate the conditional probability: **Given that an email is flagged by the filter, what is the probability that it is actually spam?**
   $$P(\text{Spam} \mid \text{Flagged}) = \frac{\text{True Positives}}{\text{Total Flagged (TP + FP)}}$$
4. Print $P(\text{Spam} \mid \text{Flagged})$ as a percentage.
5. **Machine Learning Reflection (The Base Rate Fallacy):**  
   *Even though the classifier is 96% accurate at catching spam, why is $P(\text{Spam} \mid \text{Flagged})$ significantly lower than 96%? Why is understanding this phenomenon critical when deploying automated fraud or defect detection models?*

---

## Submission Checklist

- [ ] Task 1: Simulated LLN with die rolls and plotted the convergence towards $1/6$.
- [ ] Task 2: Applied the general addition rule accounting for overlapping events.
- [ ] Task 3: Applied the multiplication rule to calculate end-to-end pipeline reliability.
- [ ] Task 4: Built a frequency contingency table and calculated $P(\text{Spam} \mid \text{Flagged})$, explaining the base rate fallacy.
- [ ] All code runs cleanly and produces informative outputs.
