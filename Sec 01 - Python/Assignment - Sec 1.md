# Assignment 01: Python Foundations for Artificial Intelligence

**Course:** Introduction to Artificial Intelligence  
**Section:** 01 - Python Essentials, OOP, File Handling & Algorithm Analysis  
**Estimated Time to Complete:** ~2 Hours (120 Minutes)  
**Deliverables:** Submit a Jupyter Notebook (`assignment_01_<your_name>.ipynb`) or Python script (`assignment_01_<your_name>.py`) with all code and written explanations completed.

---

## Overview & Objectives

In modern Artificial Intelligence and Machine Learning workflows, models rarely consume raw, clean data directly. Before training neural networks or ML classifiers, AI engineers must build resilient data pipelines, handle large file formats, design modular object-oriented components, and optimize algorithmic complexity to avoid memory blowups and slow training loops.

This assignment tests your mastery over the fundamental building blocks covered in **Section 01**:
1. **Core Data Structures & Preprocessing** (Lists, Sets, Dictionaries, String parsing, Error handling)
2. **Object-Oriented Programming (OOP) & Generators** (Classes, Inheritance, `yield` vs `return` for memory-efficient batching)
3. **File I/O & Dataset Parsing** (Reading raw text and CSV datasets with standard Python libraries)
4. **Algorithm Analysis & Optimization** (Big-O Time/Space complexity, sorting, search, and algorithmic trade-offs)

---

## Part 1: Text Tokenization & Vocabulary Building (25 mins)
*Topics: Strings, Dictionaries, Sets, Counter, Error Handling (`01 - Introduction.ipynb`)*

In Natural Language Processing (NLP), text must be cleaned, tokenized, and mapped to a numeric vocabulary before being passed to an AI model.

### Task 1.1: Text Preprocessor & Cleaner
Write a function `clean_and_tokenize(text)` that:
1. Validates that the input is a string. If not, raises a `TypeError` with an informative error message.
2. Converts the text to lowercase.
3. Removes punctuation characters (`,`, `.`, `!`, `?`, `;`, `:`, `"`).
4. Splits the text into a list of individual word tokens by whitespace.

### Task 1.2: Stopword Filtering & Vocabulary Counter
In AI, extremely frequent words (e.g., "the", "is", "at") often add noise.
1. Define a `set` named `STOPWORDS` containing: `{"the", "is", "a", "an", "and", "in", "to", "of", "it", "for", "on"}`.
2. Write a function `build_vocabulary(corpus, stopwords, min_freq=2)`:
   - `corpus` is a list of raw text sentences.
   - Use `clean_and_tokenize` on each sentence with proper `try-except` error handling to skip invalid entries (e.g., `None` or non-string values) while logging a warning.
   - Filter out words present in `stopwords`.
   - Count the frequencies of the remaining words using `collections.Counter` or standard dictionaries.
   - Return a `dict` mapping words that appear at least `min_freq` times to a unique integer ID starting from `1` (reserve `0` for `<PAD>`).

### Test Snippet:
```python
sample_corpus = [
    "Machine learning enables computers to learn from experience.",
    "Deep learning is a subset of machine learning.",
    "Artificial intelligence is transforming the modern world.",
    None,  # Should be caught by try-except without crashing
    "Learning algorithms require data and compute power."
]

vocab = build_vocabulary(sample_corpus, STOPWORDS, min_freq=2)
print("Built Vocabulary:", vocab)
# Expected vocabulary should include repeated words like 'learning', 'machine', etc.
```

---

## Part 2: OOP & Memory-Efficient Batch Generators (35 mins)
*Topics: Classes, Inheritance, Encapsulation, `yield` vs `return` (`02 - OOP.ipynb`)*

Deep learning datasets often consist of gigabytes of training samples that cannot all fit into RAM at once. We use OOP to model samples and Python generators (`yield`) to load training batches lazily.

### Task 2.1: The `DataSample` Base Class and `TextSample` Subclass
1. Create a base class `DataSample`:
   - `__init__(self, sample_id, label)`
   - Include a method `summary()` that returns a string formatted as: `f"Sample #{self.sample_id} - Label: {self.label}"`.
2. Create a subclass `TextSample` that inherits from `DataSample`:
   - `__init__(self, sample_id, label, text)`: Call `super().__init__(sample_id, label)` and store `self.text`.
   - Implement a method `word_count()` that returns the number of words in `self.text`.
   - Override `summary()` to return: `f"TextSample #{self.sample_id} ({self.word_count()} words) - Label: {self.label}"`.

### Task 2.2: The `DatasetLoader` Class with Generator Batching
1. Create a class `DatasetLoader`:
   - `__init__(self, samples=None)`: Initialize with a list of `DataSample` objects (default to an empty list).
   - Method `add_sample(self, sample)`: Add a sample to the internal list.
   - Method `__len__(self)`: Return total number of samples.
   - Generator method `batch_generator(self, batch_size)`:
     - Lazily yields batches of size `batch_size` (as a list of samples) using the `yield` keyword.
     - If the total number of samples is not evenly divisible by `batch_size`, the final batch should yield the remaining samples.
2. **Written Reflection (in a comment or markdown cell):**
   - Explain why using `yield` for batching is advantageous over returning a full list of all batches (e.g. `return list_of_all_batches`) when training large AI models.

### Test Snippet:
```python
loader = DatasetLoader()
sample_texts = [
    "AI is fascinating", "Python makes data processing clean", 
    "Neural networks learn patterns", "Generators save memory",
    "Reinforcement learning uses rewards"
]

for idx, text in enumerate(sample_texts, start=1):
    loader.add_sample(TextSample(sample_id=idx, label=(idx % 2), text=text))

print(f"Total samples: {len(loader)}")
for b_idx, batch in enumerate(loader.batch_generator(batch_size=2), start=1):
    print(f"Batch {b_idx}: {[s.summary() for s in batch]}")
```

---

## Part 3: File Handling & Dataset Ingestion (30 mins)
*Topics: `open()`, context managers, `csv` module, statistics (`03 - File Handling.ipynb`)*

You have a dataset located at `data/bank.csv` containing marketing records (customer age, job, balance, loan status, campaign results). In this part, you will write a robust ingestion module using only Python's built-in `csv` module (do not use `pandas`).

### Task 3.1: Parsing `data/bank.csv`
Write a function `load_bank_data(file_path)` that:
1. Opens `data/bank.csv` using a `with open(...)` context manager with `utf-8` encoding.
2. Reads the CSV using `csv.DictReader`.
3. Validates and converts numerical columns:
   - Convert `age` to `int`.
   - Convert `balance` to `int` or `float`.
   - Convert `duration` to `int`.
4. Handles any formatting or missing data errors gracefully using `try-except` blocks.
5. Returns a list of dictionaries containing all successfully parsed rows.

### Task 3.2: Aggregations & Summary Report
Write a function `generate_dataset_summary(records, output_file="summary_report.txt")` that calculates:
1. **Total valid records** loaded.
2. **Average age** across all customers.
3. **Average account balance**.
4. **Subscription breakdown:** Count how many customers subscribed to the term deposit (`y == 'yes'`) vs refused (`y == 'no'`).
5. **Job Distribution:** The top 3 most frequent job titles in the dataset.
6. Writes these statistics into a clean text file `summary_report.txt` and prints the contents.

### Test Snippet:
```python
import os

csv_path = os.path.join("data", "bank.csv")
records = load_bank_data(csv_path)
print(f"Successfully loaded {len(records)} records.")

generate_dataset_summary(records, output_file="summary_report.txt")
```

---

## Part 4: Algorithm Analysis & Optimization (30 mins)
*Topics: Big-O, Time Complexity, Space Complexity, Benchmarking (`04 - Algorithm Analysis.ipynb`)*

In AI, choosing the wrong algorithm or data structure can lead to combinatorial explosions ($O(n^2)$ vs $O(n)$).

### Task 4.1: Pair Similarity / Duplicate Feature Detection
Suppose you have a feature list of $N$ customer IDs or embedding identifiers, and you need to detect whether there are any duplicate entries in your training dataset.

1. Implement **Approach A (Brute Force)**:
   ```python
   def find_duplicates_brute(arr):
       # Nested loop checking all unique pairs (i, j)
       pass
   ```
2. Implement **Approach B (Optimized Hash Set)**:
   ```python
   def find_duplicates_set(arr):
       # Using a set to track seen elements in a single pass
       pass
   ```
3. **Benchmarking**:
   - Generate a synthetic list of 10,000 integers using Python's `random.randint`.
   - Measure the execution time of both approaches using `time.time()`.
   - Print the elapsed time for both approaches.

### Task 4.2: Big-O Theoretical Analysis
Answer the following questions in markdown cells with clear reasoning:
1. What is the **Time Complexity** and **Auxiliary Space Complexity** of `find_duplicates_brute` in Big-O notation?
2. What is the **Time Complexity** and **Auxiliary Space Complexity** of `find_duplicates_set` in Big-O notation?
3. In deep learning feature extraction, suppose an algorithm computes the dot product between every pair of $N$ feature vectors of dimension $D$.
   - What is the total time complexity expressed in terms of $N$ and $D$?
   - If $N$ grows from $1,000$ to $10,000$ (a 10x increase), by what factor does the total computation increase?

---

## Grading Rubric

| Section | Weight | Criteria |
| :--- | :---: | :--- |
| **Part 1: Text Tokenization & Vocabulary** | **25%** | Correct string preprocessing, proper `try-except` error handling, correct vocabulary creation and indexing starting at 1. |
| **Part 2: OOP & Batch Generators** | **25%** | Proper inheritance, method overriding, functioning generator (`yield`), accurate reflection on memory consumption. |
| **Part 3: File Handling & CSV Ingestion** | **25%** | Safe file parsing with `with open()`, correct type conversions, summary calculations, report file generation. |
| **Part 4: Algorithm Analysis & Optimization** | **25%** | Correct implementation of both duplicate checkers, empirical timing comparison, sound Big-O time and space explanations. |
| **Total** | **100%** | |

---

## Submission Checklist
- [ ] Your code executes cleanly from top to bottom without unhandled exceptions.
- [ ] You only used Python standard library modules (`csv`, `collections`, `time`, `random`, `os`, `sys`).
- [ ] All written questions in Part 2 and Part 4 are answered.
- [ ] File named `assignment_01_<your_name>.ipynb` or `assignment_01_<your_name>.py` is saved and ready to submit.
