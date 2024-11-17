---
title: "DataPreprocessing"
output: rmarkdown::html_vignette
vignette: >
  %\VignetteIndexEntry{DataPreprocessing}
  %\VignetteEngine{knitr::rmarkdown}
  %\VignetteEncoding{UTF-8}
---

```{r, include = FALSE}
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)
```

```{r setup}
library(DataPreprocessing)
```

# Overview

Welcome to the **DataPreprocessing** package! This vignette serves as a quick start guide to help you get started with data preprocessing using our package. 

The DataPreprocessing package simplifies this crucial step -- extracting meaningful insights from raw dat, by offering a wide range of tools and techniques tailored for data cleaning, transformation, and reduction. By automating these tasks, DataPreprocessing allows users to focus on analysis, accelerating the path from raw data to actionable insights.

After acquiring a foundational understanding of data preprocessing techniques, users can employ the **DataPreprocessing** package to tackle a variety of challenges in data preparation. This vignette aims to provide an overview of the package's design and demonstrate its utility in solving common data preprocessing tasks.

The vignette is structured into seven sections:


1. **Data Encoding**
  - **Categorical Encoding:** Employ the `encode_categorical` function to transform categorical variables into numerical representations, facilitating model training and analysis.
  - **Quantitative Encoding:** Use the `encode_quantitative` function to encode quantitative variables using specified methods, enhancing feature representation and model performance.

2. **Data Standardization**
  - **Standardization:** Employ the `standardize` function to standardize data using specified methods, ensuring consistent scale and facilitating model convergence.

3. **Abnormal Data Handling**
  - **Handling Missing Values:** Utilize the `missing_values` function to impute missing values using specified methods, ensuring completeness and accuracy of datasets.
  - **Outlier Handling:** Use the `replace_outliers` function to detect and replace outliers using specified methods, ensuring data integrity and reliability.
  - **Sample Balancing:** Utilize the `balance_sample` function to address class imbalance in datasets, ensuring equal representation for improved model performance.

4. **Feature Engineering & Dimensionality Reduction**
  - **Feature Selection:** Employ the `featureSelectionPearson` function to identify and select relevant features based on Pearson correlation coefficients, enhancing model interpretability and efficiency.
  - **Dimensionality Reduction:** Utilize the `pca` function to perform Principal Component Analysis (PCA) for dimensionality reduction, improving computational efficiency and model interpretability.

5. **Normality Testing**
  - **Normality Testing:** Use the `normal_test` function to assess the normality of data distributions, ensuring the validity of statistical analyses and model assumptions.

6. **Dataset manipulation**

7. **Connection to Course Materials in DataPreprosessing**

By following the examples and demonstrations provided in each section, users can effectively leverage the **DataPreprocessing** package to preprocess their datasets and prepare them for further analysis and modeling tasks.


# Data Encoding

## `encode_categorical()`: Encode categorical variables

Data labeling, also known as encoding, is a fundamental step in data preprocessing where categorical variables are converted into numerical representations. This transformation is necessary because many machine learning algorithms require numerical inputs. Data labeling ensures that categorical variables can be effectively utilized in modeling and analysis.

### Code Explanation

The provided R code implements data labeling for categorical variables using two approaches:

1. **Default Labeling Approach**:
   - If no encoding map (`encoding_map`) is provided, the function uses the original order of categories to encode them.
   - It encodes the categorical variable using the `factor` function and converts the factor levels into integer codes.
   - The encoding map is then updated to associate the encoded values with their respective categories.

2. **Custom Labeling Approach**:
   - If an encoding map is provided, the function uses the specified encoding for each category.
   - It checks each category in the data against the provided encoding map and assigns the corresponding label.
   - Categories not found in the encoding map are encoded as NA, with a warning message generated.

For both approaches:

  - The `new_column` parameter determines whether the encoded values are stored in a new column or overwrite the original column.
  - If `new_column` is set to `TRUE`, a new column with a prefixed name (`new_column_name`) is created to store the encoded values.
  - The `get_picture` parameter determines whether a Nightingale Rose Chart of the encoded data should be displayed. Defaults to TRUE.
  - If `get_picture` is set to `TRUE`, a Nightingale Rose Chart is created to reflect the percentage of data size and the periodicity of the data. This chart, an advanced version of a pie chart, displays data in a circular format where each "slice" or "petal" represents a category. The area or size of each slice is proportional to the frequency of the category it represents, providing a visual comparison across categories.

### Example Usage

```{r}
# Call encode_categorical function with your data, column name, and optionally encoding map and new_column parameter
data <- data.frame(
  category1 = c("A", "B", "C", "A", "B", "C", "A", "B", "C", "A"),
  category2 = c("a", "b", "a", "b", "a", "b", "a", "b", "a", "b")
)

# Example 1: Default labeling approach
encode_categorical(data, "category1")
```

```{r}
# Example 2: Custom labeling approach with a provided encoding map and overwriting the original column
custom_encoding <- list("a" = 2, "b" = 4)
encode_categorical(data, "category2", encoding_map = custom_encoding, new_column = FALSE, get_picture = TRUE)
```

## `encode_quantitative()`: Encode quantitative variables

Data encoding is a process in data preprocessing where quantitative variables are transformed into categorical or binary representations. This transformation is often performed to simplify the data or prepare it for specific modeling techniques that require categorical inputs.

### Code Explanation

The provided R code implements data encoding for quantitative variables using two different methods:

1. **Mean-Based Encoding (`method = "mean"`)**:
   - This method encodes the quantitative variable based on whether each value is greater than or equal to the mean of the variable.
   - Values greater than or equal to the mean are encoded as 1, while values below the mean are encoded as 0.

2. **Quantile-Based Encoding (`method = "quantile"`)**:
   - This method divides the range of the quantitative variable into quantiles.
   - It then encodes each value based on the quantile it belongs to, assigning a unique label to each quantile.

For both methods:

  - The `new_column` parameter determines whether the encoded values are stored in a new column or overwrite the original column.
  - If `new_column` is set to `TRUE`, a new column with a prefixed name (`new_column_name`) is created to store the encoded values.

### Example Usage

```{r}
# Call encode_quantitative function with your data, column name, method, and optionally q and new_column parameters
data <- data.frame(
  value1 = c(1, 2, 5, 7, 6, 10, 8, 9, 4, 3),
  value2 = c(10, 20, 15, 25, 30, 35, 40, 45, 50, 55)
)

# Example 1: Mean-based encoding and overwriting the original column
encode_quantitative(data, "value1", method = "mean", new_column = FALSE)

# Example 2: Quantile-based encoding with custom number of quantiles and storing in a new column
encode_quantitative(data, "value2", method = "quantile", q = 5, new_column = TRUE)
```

# Data Standardization

## `standardize()`: Standardize data

Data standardization, also known as feature scaling, is a process in data preprocessing aimed at transforming numerical features to a common scale. Standardization ensures that all features have the same mean and standard deviation, making them comparable and improving the performance of certain machine learning algorithms.

### Code Explanation

The provided R code implements data standardization using three different methods:

1. **Min-Max Scaling (`method = "min-max"`)**:
   - Min-max scaling transforms the data to a range between 0 and 1 by subtracting the minimum value and dividing by the range (maximum - minimum).
   
$$
X_{\text{new}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}
$$

2. **Z-Score Scaling (`method = "z-score"`)**:
   - Z-score scaling (also known as standardization) scales the data to have a mean of 0 and a standard deviation of 1 by subtracting the mean and dividing by the standard deviation.

$$
     X_{\text{new}} = \frac{X - \mu}{\sigma}
$$     

3. **Decimal Scaling (`method = "decimal"`)**:
   - Decimal scaling scales the data by dividing each value by a power of 10 determined by the maximum absolute value in the column. This method ensures that the absolute values of all features are less than 1.
   
$$
      X_{\text{new}} = \frac{X}{10^d}
$$
Where \(d \) is the smallest number of integers that make the absolute value of all data less than 1.
    
The function allows standardization of a specific column in the dataset. It also provides the option to store the standardized data in new columns or overwrite the original columns.

### Example Usage

```{r}
# Call standardize function with your data, method, and optionally column and new_column parameters
data <- data.frame(
  value1 = c(100, 200, 300, 400, 500, 600, 700, 800),
  value2 = c(11.56, 10.15, 6.02, 11.24, 9.89, 9.69, 7.06, 9.04)
)

# Example 1: Standardize the first column using min-max scaling and store in new columns
standardize(data, method = "min-max", column = 1, new_column = TRUE)

# Example 2: Standardize the second column using z-score scaling and overwrite original columns
standardize(data, method = "z-score", column = "value2", new_column = FALSE)

```

# Abnormal Data Handling

## `missing_values()`: Handle missing values

Missing value handling in data preprocessing aims at dealing with observations that have incomplete or unavailable data. Missing values can arise due to various reasons such as data entry errors, equipment malfunctions, or simply because the information was not collected. Handling missing values is essential to ensure the accuracy and reliability of data analysis and modeling processes.

### Code Explanation

The provided R code implements missing value handling using various methods for both numeric and non-numeric variables:

1. **Numeric Variables**:
   - For numeric variables, missing values can be replaced with the mode, mean, median, previous value, next value, or a fixed value.
   - The `fill_method` parameter specifies the method to use for filling missing values.
   - If `fill_method` is set to "fixed", the `fill_value` parameter determines the fixed value to use for replacement.

2. **Non-Numeric Variables**:
   - For non-numeric variables, missing values can be replaced with the mode, previous value, next value, or a fixed value.
   - The handling of missing values for non-numeric variables is slightly different from numeric variables due to the absence of mean and median calculations.

### Example Usage

```{r}
# Call missing_values function with your data, column name, fill_method, and optionally fill_value parameter
data <- data.frame(
  numeric_column = c(1, 2, NA, 4, 5, 8),
  character_column = c("a", "b", NA, "c", NA, "b")
)

# Example 1: Replace missing values in a numeric column with the mean
data_filled <- missing_values(data, "numeric_column", fill_method = "mean")
print(data_filled)

# Example 2: Replace missing values in a character column with the mode
data_filled <- missing_values(data_filled, "character_column", fill_method = "mode")
print(data_filled)
```

## `replace_outliers()`: Replace outliers in data

Outlier handling in data preprocessing aims at identifying and dealing with observations that significantly deviate from the rest of the dataset. Outliers can occur due to measurement errors, data entry mistakes, or genuine anomalies in the data. Handling outliers is essential to ensure that they do not unduly influence the results of statistical analyses or machine learning models.

### Code Explanation

The provided R code implements outlier handling using three different methods:

1. **3 Sigma Method (`method = "3sigma"`)**:
   - Calculates mean (μ) and standard deviation (σ) of the data.
   - Identifies outliers as values that fall outside of the range defined by mean ± 3 standard deviations.

$$
\text{T}_{\text{min}} = \mu - 3 \sigma
$$

$$
\text{T}_{\text{max}} = \mu + 3 \sigma
$$

$$
\text{outliers} = \text{values} < \text{T}_{\text{min}} \,|\, \text{values} > \text{T}_{\text{max}}
$$

2. **Interquartile Range (IQR) Method (`method = "IQR"`)**:
   - Computes the first quartile (\( Q_1 \)), third quartile (\( Q_3 \)), and interquartile range (\( IQR \)).
   - Identifies outliers as values that fall below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR.

$$
\text{IQR} = Q_3 - Q_1
$$
 
$$
\text{T}_{\text{min}} = Q_1 - 1.5 \times \text{IQR}
$$

$$
\text{T}_{\text{max}} = Q_3 + 1.5 \times \text{IQR}
$$

$$
\text{outliers} = \text{values} < \text{T}_{\text{min}} \,|\, \text{values} > \text{T}_{\text{max}}
$$

3. **Median Absolute Deviation (MAD) Method (`method = "MAD"`)**:
   - Calculates the median (\( M \)) and median absolute deviation (\( MAD \)).
   - Identifies outliers as values that exceed median ± 3 * MAD.
   
$$
\text{MAD} = 1.4826 \times \text{median}(|\text{values} - M|)
$$

$$
\text{T}_{\text{min}} = M - 3 \times \text{MAD}
$$

$$
\text{T}_{\text{max}} = M + 3 \times \text{MAD}
$$

$$
\text{outliers} = \text{values} < \text{T}_{\text{min}} \,|\, \text{values} > \text{T}_{\text{max}}
$$

For each method:

  - Outliers are identified based on predefined threshold values.
  - The `replace_method` parameter determines how outliers are replaced:
    - `median`: Replace outliers with the median of non-outlier values.
    - `mean`: Replace outliers with the mean of non-outlier values.
    - `zero`: Replace outliers with zero.
    - `custom`: Replace outliers with a custom value specified by `replace_value`.
  - The `get_picture` parameter determines whether to generatea violin plot overlaid with jittered points. This visualization compares the original data distribution with the modified data where outliers have been replaced.

### Example Usage

```{r}
# Call replace_outliers function with your data, method, replace_method, and replace_value

# Example 1: Default 3 sigma method and replaced by mean
set.seed(1)
data <- data.frame(
  value1 = c(rnorm(19, 5, 1), 10)
)
replace_outliers(data, "value1")
```
```{r}
# Example 2: IQR method and replaced by a specified value 35
data <- data.frame(
  value2 = c(10, 20, 30, 100, 25, 35, 200, 45, 50, 55)
)
replace_outliers(data, "value2", method = "IQR", replace_method = "custom", replace_value = 35)
```

## `balance_sample()`: Balance the dataset

The process of sample balancing involves adjusting the class distribution in the dataset to ensure that each class is adequately represented. There are several techniques for sample balancing, and one common approach is oversampling, where instances from the minority class (less represented class) are duplicated or synthetically generated to match the number of instances in the majority class.

### Code Explanation

Let's break down the provided R code and its workflow:

1. **Data Preparation and Subset Creation**:
   - Calculate the distribution of class labels in the original dataset.
   - Identify the classes with the minimum and maximum occurrences.
   - Separate the minority and majority class instances from the original dataset.

2. **Balancing the Dataset**:
   - Use the ROSE package for oversampling to generate synthetic samples for the minority class, creating a balanced subset of data.
   - Calculate the class distribution in the original and balanced datasets, and print the original and balanced class distributions for comparison.
   
3. **Data Combination and Visualization**:
   - Combine the balanced subset with the instances of the majority class.
   - Optionally, generate a treemap visualization to compare the class distribution before and after balancing if `get_picture = TRUE`.

### Usage Example


```{r}
# Call balanced_sampling function with your data and class column name
set.seed(1)
data <- data.frame(
  class = c(rep("A", 10), rep("B", 5), rep("C", 3), rep("D", 2)),
  feature1 = rnorm(20),
  feature2 = runif(20)
)
balanced_data <- balanced_sampling(data, 'class')
```

# Feature Engineering & Dimensionality Reduction

## `featureSelectionPearson()`: Feature selection based on Pearson correlation

Feature selection is a crucial step in data preprocessing aimed at identifying and selecting the most relevant features from the dataset for building predictive models. The objective is to reduce the dimensionality of the dataset by eliminating irrelevant or redundant features, thereby improving model performance, reducing overfitting, and speeding up computation.

### Code Explanation

The provided R code implements feature selection based on Pearson correlation coefficient:

1. **`featureSelectionPearson` Function**:
   - This function calculates the Pearson correlation coefficient between each feature and the target variable.
   - It takes parameters such as the dataset (`data`), list of features (`features`), target variable (`target`), correlation threshold (`threshold`), and an optional argument to determine whether to generate a heatmap (`get_picture`).
   - The Pearson correlation coefficient measures the linear relationship between two variables, ranging from -1 to 1. A coefficient closer to 1 indicates a strong positive correlation, while a coefficient closer to -1 indicates a strong negative correlation.
   - Features with absolute correlation coefficients greater than the specified threshold with the target variable are selected.
   - Optionally, it generates a heatmap visualizing the correlation matrix if `get_picture` is set to `TRUE`.
   
2. **Formula of Pearson Correlation Coefficient**:
   Pearson correlation coefficient (\( \rho \)) between two variables \( X \) and \( Y \) is calculated as:
   
$$
\rho = \frac{{\sum (X_i - \bar{X})(Y_i - \bar{Y})}}{{\sqrt{{\sum (X_i - \bar{X})^2}\sum (Y_i - \bar{Y})^2}}}
$$
   Where:
   
   - \( \bar{X} \) and \( \bar{Y} \) are the means of variables \( X \) and \( Y \) respectively.
   
   - \( X_i \) and \( Y_i \) are individual data points of variables \( X \) and \( Y \) respectively.

3. **Usage**:
   - Specify the list of features (`features`), target variable (`target`), and correlation threshold (`threshold`) in the function call.
   - It returns the names of the selected features that meet the correlation threshold.
   - If no features meet the threshold, it provides a message and returns `NULL`.

### Example Usage
    
```{r}
# Call featureSelectionPearson function with your dataset, features, target variable, and threshold
data <- data.frame(
  feature1 = c(1, 3, 3, 4, 6),
  feature2 = c(-2, -1, -4, -5, -4),
  feature3 = c(2, 1, 1, 5, 3),
  target = c(10, 20, 30, 40, 50)
)

selected_features <- featureSelectionPearson(data, c("feature1", "feature2", "feature3"), 'target', threshold = 0.6, get_picture = TRUE)
selected_features
```


## `pca()`: Perform Principal Component Analysis (PCA)

Data dimensionality reduction is a process in data preprocessing aimed at reducing the number of features (dimensions) in a dataset while preserving the most important information. Dimensionality reduction techniques are often employed to address issues such as computational complexity, multicollinearity, and the curse of dimensionality. These techniques help improve the efficiency and effectiveness of machine learning algorithms by simplifying the input space.

### Code Explanation

The provided R code implements data dimensionality reduction using Principal Component Analysis (PCA), a popular technique for reducing the dimensionality of high-dimensional data:

1. **Principal Component Analysis (PCA)**:
   - PCA identifies the orthogonal directions (principal components) that capture the maximum variance in the data.
   - It projects the original data onto a lower-dimensional subspace spanned by the principal components, and allows to choose whether to scale the data before performing PCA by setting the parameter `scale_data`.
   - The number of principal components retained is determined based on the cumulative proportion of variance explained.
   - The code computes PCA using the `prcomp` function from the `stats` package.
   - It calculates the cumulative proportion of variance explained and selects the number of principal components required to achieve a cumulative proportion greater than a specified threshold (defaults to 0.95).
   - The data is then projected onto the selected number of principal components, and a new dataset with reduced dimensions is generated.
   - Optionally, it also prints a summary of PCA results if `verbose` is set to `TRUE`.
   
$$
   \text{PC}_1 = \phi_{11}X_1 + \phi_{21}X_2 + \ldots + \phi_{p1}X_p
$$

$$
   \text{PC}_2 = \phi_{12}X_1 + \phi_{22}X_2 + \ldots + \phi_{p2}X_p
$$
$$   
   \vdots
$$

$$
   \text{PC}_p = \phi_{1p}X_1 + \phi_{2p}X_2 + \ldots + \phi_{pp}X_p
$$
   where:
   
   - \( \text{PC}_i \) is the \( i \)-th principal component.
   
   - \( \phi_{ij} \) is the loading of the \( j \)-th original variable on the \( i \)-th principal component.
   
   The principal components are computed such that they are orthogonal to each other and capture the maximum variance in the data.

2. **Cumulative Proportion of Variance Explained**:

   The cumulative proportion of variance explained by the first \( k \) principal components is calculated as:

$$
\text{Cumulative Proportion}_{k} = \frac{{\sum_{i=1}^{k} \lambda_i}}{{\sum_{i=1}^{p} \lambda_i}}
$$
   where:
   
   - \( \lambda_i \) is the eigenvalue corresponding to the \( i \)-th principal component.

   Typically, a threshold (e.g., 95%) is set for the cumulative proportion of variance explained, and the number of principal components retained is determined accordingly.
   
### Example Usage

```{r}
# Call pca function with your dataset
data <- data.frame(
  Feature1 = c(1.2, 2.3, 3.1, 4.5, 2.8, 3.9, 1.8, 4.0, 2.5, 3.6),
  Feature2 = c(3.4, 4.5, 2.9, 3.2, 3.7, 2.6, 3.0, 3.5, 3.1, 2.8),
  Feature3 = c(2.5, 1.8, 3.0, 2.6, 2.4, 3.2, 2.9, 2.1, 2.8, 2.7),
  Feature4 = c(1.8, 2.1, 1.5, 2.0, 1.9, 2.3, 2.2, 1.7, 1.6, 2.4),
  Feature5 = c(4.3, 3.6, 4.2, 3.9, 4.1, 3.8, 4.0, 3.4, 4.4, 3.5)
)
pca(data, scale_data = TRUE, threshold = 0.95, verbose = TRUE)
```

# Normality Testing

## `normal_test()`: Feature normality testing

Feature normality testing is a process in data preprocessing aimed at assessing whether numerical features in a dataset follow a normal distribution. Normality testing is important because many statistical techniques and machine learning algorithms assume that the data is normally distributed. Assessing the normality of features helps in selecting appropriate statistical methods and ensuring the validity of analysis results.

### Code Explanation

The provided R code implements feature normality testing using the Shapiro-Wilk test, a statistical test used to assess normality based on sample data:

1. **Shapiro-Wilk Test**:
   - The Shapiro-Wilk test is applied to each column of the dataset to assess the normality of the feature's distribution.
   - Optionally, it prints a p-value indicating the likelihood that the data is drawn from a normal distribution .
   - Features with p-values greater than a significance level (0.05 by    default) are considered to be normally distributed.

   The Shapiro-Wilk test statistic (\( W \)) is computed as:
$$
W = \frac{{\left(\sum_{i=1}^{n} a_i x_{(i)}\right)^2}}{{\sum_{i=1}^{n} (x_i - \bar{x})^2}} 
$$
   where:
   
   - \( n \) is the sample size.
   
   - \( x_{(i)} \) are the ordered sample values.
   
   - \( a_i \) are constants computed from the sample size and the covariance matrix of order statistics.

   The test statistic \( W \) is then compared to critical values from the Shapiro-Wilk distribution to determine the significance level (p-value). A higher p-value (> 0.05) suggests that the data is likely drawn from a normal distribution.

2. **Filtering Normal Parameters**:
   - The code filters out the features whose Shapiro-Wilk test p-values are greater than the significance level, indicating that they are likely to follow a normal distribution.

### Example Usage

```{r}
# Call normal_test function with your dataset
data <- data.frame(
  Feature1 = c(1.2, 2.3, 3.1, 4.5, 2.8, 3.9, 1.8, 4.0, 2.5, 3.6),
  Feature2 = c(3.4, 4.5, 2.9, 3.2, 3.7, 2.6, 3.0, 3.5, 3.1, 2.8),
  Feature3 = c(2.5, 1.8, 3.0, 2.6, 2.4, 3.2, 2.9, 2.1, 2.8, 2.7),
  Feature4 = c(1.8, 2.1, 1.5, 2.0, 1.9, 2.3, 2.2, 1.7, 1.6, 2.4),
  Feature5 = c(4.3, 3.6, 4.2, 3.9, 4.1, 3.8, 4.0, 3.4, 4.4, 3.5)
)
normal_test(data, significance_level = 0.05, verbose = TRUE)
```

# Dataset manipulation
In this package, we include a dataset called "transaction_variables". This dataset contains 500000 investments of 27 companies belonging to 5 sectors in the last 10 years with a horizon between 1 day and 2 years. This dataset includes more than 10 variables such as Sharpe ratio and inflation. This dataset is from the website 'https://raw.githubusercontent.com/ImanolR87/AI-Learn-to-invest/main/datasets/final_transactions_dataset.csv'.
```{r}
# Firstly, we need to fill the missing values in the 'amount' column.
transactions_dataset <- missing_values(transactions_dataset, 'amount', fill_method = "mode")
summary(transactions_dataset$amount)

# Secondly, we need to encode the values in the 'ESG_ranking', 'sector', 'company' and 'investment' columns.
transactions_dataset <- encode_quantitative(transactions_dataset, column = "ESG_ranking", method = "quantile", q = 3, new_column = FALSE)
transactions_dataset <- encode_categorical(transactions_dataset, column = 'company', new_column = FALSE)
transactions_dataset <- encode_categorical(transactions_dataset, column = 'sector', new_column = FALSE, encoding_map = c("AUTO"=1, "BANK"=2, "FMCG"=3, "RETAIL"=4, "TECH"=5))
transactions_dataset <- encode_categorical(transactions_dataset, column = 'investment', new_column = FALSE)

# Thirdly, we need to adjust the outliers in the 'expected_return' column.
summary(transactions_dataset$expected_return)
transactions_dataset <- replace_outliers(transactions_dataset, column = "expected_return", method = "IQR", replace_method = "median", new_column = FALSE)
summary(transactions_dataset$expected_return)

# Fourthly, we need to standardize the 'nominal_return' values.
transactions_dataset <- standardize(transactions_dataset, method = "min-max", column = "nominal_return", new_column = FALSE)

# Fifthly, we want to use feature selection method to select the most relevant features to stock price.
features <- c("Sharpe_Ratio", "company", "PE_ratio", "EPS_ratio", "PS_ratio", "PB_ratio", "NetProfitMargin_ratio", "current_ratio" , "roa_ratio", "roe_ratio")
selected_features <- featureSelectionPearson(transactions_dataset, features, 'price_BUY', threshold = 0.3, get_picture = TRUE)

# Sixthly, we use PCA dimentional reduction method to reduce the dimention of the selected features.
PCA_features <- pca(transactions_dataset[, c(selected_features,"price_BUY")])
head(PCA_features)
transactions_dataset <- cbind(transactions_dataset, PCA_features)

# Seventhly, we want to balance the investment classes.
test_data <- transactions_dataset
transactions_dataset <- balanced_sampling(test_data[, c('price_BUY',"PC1","PC2","company")], class = "company")

# Finally, we want to test which variable follows normal distribution.
normal_columns <- normal_test(transactions_dataset[1:5000, c('price_BUY',"PC1","PC2")])
normal_columns
```

# Connection to Course Materials in DataPreprosessing

The construction of this R package draws heavily on the insights gained from the STA3005 course.

## **1**. Vectorized Operations

Utilizing vectorized operations can eliminate the need for redundant for loops and reduce computation time.

Example of implementation in this package:

In 'pca' function, vectorized operation was used to calculate to scale each column of the input data frame data[, -ncol(data)] individually. These operations are performed simultaneously on all elements of each column, making it a vectorized operation.

## **2**. Visualization

Visualization plays a pivotal role in enhancing comprehension, communication, and analysis across diverse domains. In this package, visualization techniques are employed extensively to illustrate data insights effectively.

Example implementations in this package:

1. Utilizing ggplot2 in the 'standardize' function to create boxplots and customize their aesthetics.
2. Generating Nightingale Rose Charts in the encode_categorical function to visualize category distributions.
3. Employing violin plots in the standardize and replace_outliers functions to compare data distributions.
4. Incorporating heatmaps of correlation matrices in the balanced_sampling function to illustrate relationship strengths. We use varied circle sizes in the upper triangle to depict relationships, and numeric values in the lower triangle to show correlation strengths.

## **3**. Control flow

Control flow (such as if, else, etc.) enhances program flexibility and logic by enabling execution of different operations based on conditions.

Example of implementation in this package:

In 'normal_test' function, control flow is managed through conditional statements, including if, to enforce data validation and error handling based on specific conditions. These statements guide the execution path, ensuring the data preprocessing function operates correctly and robustly.

## **4**. Logical Expression

Logical expressions enable the program to determine whether certain commands should be executed or not.

Example of implementation in this package:

In the 'standardize' function, the user needs to provide a TRUE or FALSE value for the built-in parameter to specify how the input data frame should be handled.

## **5**. Apply Family

R provides a range of apply functions, enabling the application of a function across various segments of data. These functions enhance speed, conserve memory, and improve code readability.

Example of implementation in this package:

In 'normal_test' function, sapply is used to iterate over each column of the dataframe data and apply the shapiro.test function to perform the Shapiro-Wilk normality test. In this line of code, sapply(data, ...) applies the shapiro.test function to each column of the dataframe data. Thus, each element in normality_test_results represents the p-value of the Shapiro-Wilk test for the corresponding column.

## **6**. Data Structure

In R, common data structures include:

1. Vectors: Single-dimensional arrays that can contain elements of the same type.
2. Lists: Data structures that can hold elements of different types, similar to dictionaries or JSON objects.
3. Matrices: Two-dimensional arrays containing elements of the same type, with rows and columns.
4. Data Frames: Tabular data structures similar to spreadsheets, capable of holding elements of different types, with each column possibly having different attributes.
5. Factors: Data structures used to represent categorical variables, often used in statistical analysis.
6. Arrays: Data structures that can hold multidimensional data, similar to matrices but with additional dimensions.
and etc...

Example of implementation in this package:

In this project, we use several data structures, such as data frame, and factor.

In the encode_categorical function, several R data structures are utilized:

1. Vectors: Vectors are used to store the values of the column parameter and the encoded values in the encoded vector.
2. Lists: The encoding_map parameter is a list-like structure that maps category names to their encoded values.
3. Factors: Factors are created using the factor() function, which converts categorical variables into factors and ensures proper encoding.
4. Data Frames: The data parameter is assumed to be a data frame, and columns are accessed using data[[column]]. Additionally, the encoded column is added to the data frame using data[[new_column]] <- encoded.

## **7**. Indexing

Indexing in R involves accessing specific elements from data structures like vectors or data frames using their position or label, allowing for the selection of subsets of data based on numerical indices or column/row names.

Example of implementation in this package:

Indexing in this code is demonstrated through the following:

1. data[[column]] is used to index the specific column column in the data frame data, retrieving the numerical data in that column.
2. data[[column]] is also used in encoded <- ifelse(data[[column]] >= mean(data[[column]]), 1, 0) and encoded <- cut(data[[column]], breaks = quantiles, labels = FALSE, include.lowest = TRUE) to index the specific column in data and encode it based on conditions or quantiles.
3. data[[new_column]] <- encoded is utilized to write the encoded results into a new column new_column or overwrite the original column, also involving indexing.
These indexing operations facilitate the selection, processing, and updating of data.

## **8**. Iteration

Iteration is the repetitive execution of a process to approach a solution, enabling gradual refinement and effective handling of complex tasks or large datasets.

Example of implementation in this package:

The pca function utilizes iteration implicitly within the underlying PCA algorithm implemented by R's prcomp function. This iterative process involves computing principal components and associated eigenvalues to derive the final PCA results. While the pca function does not directly contain explicit iteration loops, it relies on iterative methods within prcomp for PCA computation.

## **9**. Pipe

The pipe operator (%>%) in R streams the output of one function as the input of another, promoting concise and readable code by eliminating the need for intermediate variables.

Example of implementation in this package:

In the 'pca' code, the pipe operator (%>%) is utilized to chain together function calls, enhancing code readability and conciseness. It facilitates a clear flow of operations, thereby improving code clarity and comprehensibility.
