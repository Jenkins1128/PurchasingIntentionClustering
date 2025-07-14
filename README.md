# Online Shoppers Purchasing Intention Clustering

Isaiah Jenkins

## Project Overview

This project analyzes the Online Shoppers Purchasing Intention dataset from the UC Irvine Machine Learning Repository to segment users into shoppers and non-shoppers using clustering techniques. The dataset comprises 12,330 sessions, with 84.5% (10,422) non-shoppers (`Revenue=False`) and 15.5% (1,908) shoppers (`Revenue=True`). Using Python libraries (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn), the analysis involves data preprocessing, applying clustering models (KMeans and Agglomerative Clustering with ward, complete, average, and single linkages), and evaluating their ability to distinguish shoppers from non-shoppers despite class imbalance.

## Dataset

The dataset, `online_shoppers_intention.csv`, includes 18 features (10 numerical, 8 categorical):

- **Numerical Features**:
  - `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`: Page visits and time spent in each category.
  - `BounceRates`, `ExitRates`, `PageValues`: Google Analytics metrics for bounce rate, exit rate, and page value.
  - `SpecialDay`: Proximity to special days (e.g., Valentine’s Day).
- **Categorical Features**:
  - `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`: General session and user information.
- **Target Variable**:
  - `Revenue`: Binary label indicating purchase (True) or no purchase (False).
- **Preprocessing**:
  - Dropped less relevant features: `SpecialDay`, `Month`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`.
  - Applied log transformation to handle skewed numerical features.
  - Scaled features using StandardScaler for clustering.

## Analysis

The analysis followed these steps:
1. **Data Exploration**: Examined dataset structure (12,330 rows, 18 columns), confirmed no missing values, and computed descriptive statistics and value counts.
2. **Data Visualization**: Used an elbow plot to assess optimal cluster numbers, selecting 2 clusters to align with the goal of separating shoppers from non-shoppers.
3. **Feature Engineering**: Dropped 7 features, applied log transformations, and scaled numerical features.
4. **Clustering Models**:
   - **KMeans**: Configured with 2 clusters; results indicated 1,440 shoppers and 5,744 non-shoppers.
   - **Agglomerative Clustering**: Tested with ward, complete, average, and single linkages; results ranged from 0 to 1,908 shoppers identified.
5. **Evaluation**: Compared cluster assignments to the `Revenue` label to assess segmentation effectiveness.

## Summary of Models

The analysis evaluated KMeans and Agglomerative Clustering (ward, complete, average, single linkages) for segmenting shoppers from non-shoppers. KMeans identified 1,440 shoppers (75.5% of 1,908 true shoppers) and 5,744 non-shoppers, achieving a balanced cluster ratio (1:4 vs. dataset’s 1:5.5). Agglomerative Clustering with ward and average linkages grouped 1,905 shoppers (99.8%) into one cluster but with high imbalance (94.3% of sessions), while complete linkage identified 1,360 shoppers (71.3%) with better balance but lower accuracy. Single linkage was ineffective, assigning 99.99% of sessions (1,908 shoppers, 10,421 non-shoppers) to one cluster. KMeans outperformed Agglomerative Clustering variations by providing more balanced and effective segmentation, leveraging its simplicity to better handle class imbalance.

## Key Findings

- **Model Performance**: KMeans achieved the best balance, identifying 75.5% of shoppers with a cluster ratio closer to the dataset’s distribution. Agglomerative Clustering, especially with single linkage, struggled with severe imbalance, often failing to separate shoppers effectively.
- **Feature Insights**: Features like `ProductRelated`, `ProductRelated_Duration`, and `PageValues` likely drove shopper identification due to their correlation with purchasing behavior.
- **Challenges**: The dataset’s class imbalance (84.5% non-shoppers) hindered Agglomerative Clustering’s ability to form distinct shopper clusters.
- **Takeaway**: KMeans’ centroid-based approach proved more effective than hierarchical methods, highlighting the value of simplicity in handling imbalanced data.

## Installation

To run this project, install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Download the `online_shoppers_intention.csv` dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) and place it in the `data/` directory.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/online-shoppers-clustering.git
   cd online-shoppers-clustering
   ```

2. Set up the dataset:
   - Place `online_shoppers_intention.csv` in the `data/` directory.

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook PurchasingIntentionClustering-2.ipynb
   ```

4. Follow the notebook to explore data, visualize the elbow plot, apply clustering models, and review results.

## Next Steps

- **Preprocessing Enhancements**: Explore robust scaling or outlier trimming beyond log transformation to address skewed features.
- **Model Exploration**: Test additional clustering algorithms (e.g., DBSCAN, Gaussian Mixture Models) to improve shopper segmentation.
- **Class Imbalance**: Downsample non-shoppers or apply techniques like SMOTE to balance the dataset.
- **Feature Engineering**: Create interaction terms or ratios (e.g., `ProductRelated_Duration` per page) to enhance clustering.
- **Dimensionality Reduction**: Apply PCA or t-SNE to reduce feature space and improve cluster separation.
