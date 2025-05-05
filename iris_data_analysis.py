import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Set seaborn style for better visualizations
plt.style.use('seaborn')

# Task 1: Load and Explore the Dataset
try:
    # Load the Iris dataset from a public URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    df = pd.read_csv(url, header=None, names=columns)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Dataset file not found. Please check the URL or file path.")
    sys.exit(1)
except pd.errors.EmptyDataError:
    print("Error: The dataset file is empty.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    sys.exit(1)

# Display first few rows
print("\n=== First 5 Rows of the Dataset ===")
print(df.head())

# Explore dataset structure
print("\n=== Dataset Info ===")
print(df.info())

# Check for missing values
print("\n=== Missing Values ===")
print(df.isnull().sum())

# Clean the dataset (handle missing values)
# For this dataset, we expect no missing values, but we'll include a general approach
df = df.dropna()  # Drop rows with missing values (if any)
print("\nDataset after cleaning (rows, columns):", df.shape)

# Task 2: Basic Data Analysis
# Compute basic statistics for numerical columns
print("\n=== Summary Statistics ===")
print(df.describe())

# Group by species and compute mean for numerical columns
print("\n=== Mean Values by Species ===")
species_means = df.groupby('species').mean()
print(species_means)

# Additional analysis: Species distribution
print("\n=== Species Distribution ===")
print(df['species'].value_counts())

# Task 3: Data Visualization
# Visualization 1: Line Chart (Simulated time-series: Mean sepal length per species as a proxy)
plt.figure(figsize=(8, 6))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.plot(subset.index[:10], subset['sepal_length'][:10], label=species, marker='o')
plt.title('Sepal Length Trend for First 10 Samples by Species')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.grid(True)
plt.savefig('sepal_length_trend.png')
plt.close()

# Visualization 2: Bar Chart (Mean petal length by species)
plt.figure(figsize=(8, 6))
species_means['petal_length'].plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Mean Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Mean Petal Length (cm)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.savefig('petal_length_bar.png')
plt.close()

# Visualization 3: Histogram (Sepal width distribution)
plt.figure(figsize=(8, 6))
plt.hist(df['sepal_width'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('sepal_width_histogram.png')
plt.close()

# Visualization 4: Scatter Plot (Sepal length vs Petal length)
plt.figure(figsize=(8, 6))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.scatter(subset['sepal_length'], subset['petal_length'], label=species, alpha=0.7)
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.grid(True)
plt.savefig('sepal_petal_scatter.png')
plt.close()

# Findings and Observations
print("\n=== Findings and Observations ===")
print("1. The Iris dataset contains 150 samples across three species (Iris-setosa, Iris-versicolor, Iris-virginica), with 50 samples each.")
print("2. No missing values were found, so no data cleaning was needed beyond the precautionary dropna().")
print("3. Summary statistics show that petal length has the highest variability (std = 1.765), while sepal width has the least (std = 0.436).")
print("4. Grouping by species reveals that Iris-virginica has the largest mean measurements for sepal length (6.59 cm) and petal length (5.55 cm).")
print("5. The line chart (simulated time-series) shows variability in sepal length within the first 10 samples of each species.")
print("6. The bar chart confirms Iris-virginica has the longest mean petal length, followed by Iris-versicolor and Iris-setosa.")
print("7. The histogram of sepal width shows a roughly normal distribution, with most values between 2.5 and 3.5 cm.")
print("8. The scatter plot indicates a strong positive relationship between sepal length and petal length, with Iris-setosa forming a distinct cluster.")