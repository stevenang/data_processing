#%% md
# ### SVM Classifier with Feature Comparison
#%%
import chardet
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import scipy
import seaborn as sns
import sys
import warnings

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

#%% md
# #### Load Data
#%%
data = pd.read_csv('processed_data/combined.csv')
data.head()

#%%
# Update adhd_binary column based on diagnosis
data['adhd_binary'] = data['diagnosis'].apply(lambda x: 1 if x == 'ADHD' else 0)

#%%
# Check for NaN values
print("NaN count in each column:")
nan_counts = data.isna().sum()
print(nan_counts[nan_counts > 0])

#%%
# Clean data by removing non-feature columns and columns with NaN values
columns_with_nans = nan_counts[nan_counts > 0].index.tolist()
fields_to_drop = ['subject_id', 'gender', 'age', 'diagnosis'] + columns_with_nans
clean_data = data.drop(columns=fields_to_drop)

# Get all available features (excluding the target variable)
all_features = [col for col in clean_data.columns if col != 'adhd_binary']
print(f"\nTotal available features: {len(all_features)}")

# Prepare target variable
y = clean_data['adhd_binary']

#%%
# Print dataset information
print(f"\nDataset Information:")
print(f"Total number of samples: {len(clean_data)}")
print(f"Number of ADHD samples (class 1): {sum(y == 1)}")
print(f"Number of control samples (class 0): {sum(y == 0)}")

#%% md
# #### Feature Selection and Model Comparison
#%%
def select_top_k_features(X, y, k):
    """Select top k features using ANOVA F-test"""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = np.array(X.columns)[selector.get_support()]
    feature_scores = selector.scores_[selector.get_support()]

    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'score': feature_scores
    }).sort_values('score', ascending=False)

    return X_selected, selected_features, feature_importance

def evaluate_svm(X, y, feature_names, description):
    """Train and evaluate SVM with given features"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {description}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Features: {list(feature_names)[:10]}{'...' if len(feature_names) > 10 else ''}")
    print(f"{'='*60}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train SVM
    svm = SVC(kernel='rbf', probability=True, random_state=42)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv=cv, scoring='accuracy')

    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Train on full training set and make predictions
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    y_pred_prob = svm.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_prob),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

    # Print metrics
    print(f"\nTest Set Performance:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"           Control  ADHD")
    print(f"Actual Control  {cm[0,0]:3d}   {cm[0,1]:3d}")
    print(f"       ADHD     {cm[1,0]:3d}   {cm[1,1]:3d}")

    return metrics, (y_test, y_pred_prob), svm, scaler

#%%
# Prepare data for different feature sets
X_all = clean_data[all_features]

# Dictionary to store results
results = {}
roc_data = {}

#%% md
# #### 1. All Features Analysis
#%%
print("Starting analysis with different feature sets...")

# 1. All features
metrics_all, roc_all, model_all, scaler_all = evaluate_svm(
    X_all, y, all_features, "All Features"
)
results['All Features'] = metrics_all
roc_data['All Features'] = roc_all

#%% md
# #### 2. Top 20 Features Analysis
#%%
# 2. Top 20 features
X_top20, top20_features, importance_top20 = select_top_k_features(X_all, y, 20)
print(f"\nTop 20 Feature Importance Scores:")
print(importance_top20)

metrics_top20, roc_top20, model_top20, scaler_top20 = evaluate_svm(
    pd.DataFrame(X_top20, columns=top20_features), y, top20_features, "Top 20 Features"
)
results['Top 20 Features'] = metrics_top20
roc_data['Top 20 Features'] = roc_top20

#%% md
# #### 3. Top 10 Features Analysis
#%%
# 3. Top 10 features
X_top10, top10_features, importance_top10 = select_top_k_features(X_all, y, 10)
print(f"\nTop 10 Feature Importance Scores:")
print(importance_top10)

metrics_top10, roc_top10, model_top10, scaler_top10 = evaluate_svm(
    pd.DataFrame(X_top10, columns=top10_features), y, top10_features, "Top 10 Features"
)
results['Top 10 Features'] = metrics_top10
roc_data['Top 10 Features'] = roc_top10

#%% md
# #### Results Comparison
#%%
# Create comparison dataframe
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.round(4)

print(f"\n{'='*80}")
print("PERFORMANCE COMPARISON SUMMARY")
print(f"{'='*80}")
print(comparison_df)

#%%
# Find best performing configuration for each metric
print(f"\nBest Performance by Metric:")
print(f"{'='*40}")
for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
    best_config = comparison_df[metric].idxmax()
    best_score = comparison_df.loc[best_config, metric]
    print(f"{metric.capitalize():10}: {best_config:15} ({best_score:.4f})")

#%% md
# #### Visualization
#%%
# Create visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Performance metrics comparison
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
x_pos = np.arange(len(comparison_df.index))
width = 0.15

for i, metric in enumerate(metrics_to_plot):
    ax1.bar(x_pos + i*width, comparison_df[metric], width,
            label=metric.capitalize(), alpha=0.8)

ax1.set_xlabel('Feature Sets')
ax1.set_ylabel('Score')
ax1.set_title('Performance Metrics Comparison')
ax1.set_xticks(x_pos + width*2)
ax1.set_xticklabels(comparison_df.index, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. ROC curves
colors = ['blue', 'red', 'green']
for i, (name, (y_test, y_pred_prob)) in enumerate(roc_data.items()):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score = results[name]['auc']
    ax2.plot(fpr, tpr, color=colors[i], label=f'{name} (AUC = {auc_score:.3f})')

ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curves Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Cross-validation scores
cv_means = [results[name]['cv_mean'] for name in comparison_df.index]
cv_stds = [results[name]['cv_std'] for name in comparison_df.index]

ax3.bar(comparison_df.index, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
ax3.set_ylabel('Cross-Validation Accuracy')
ax3.set_title('Cross-Validation Performance')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3)

# 4. Feature importance for top 10
if len(importance_top10) > 0:
    ax4.barh(range(len(importance_top10)), importance_top10['score'])
    ax4.set_yticks(range(len(importance_top10)))
    ax4.set_yticklabels(importance_top10['feature'], fontsize=8)
    ax4.set_xlabel('ANOVA F-Score')
    ax4.set_title('Top 10 Feature Importance')
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%% md
# #### Recommendations
#%%
print(f"\n{'='*80}")
print("RECOMMENDATIONS")
print(f"{'='*80}")

# Find overall best model
best_f1 = comparison_df['f1'].idxmax()
best_auc = comparison_df['auc'].idxmax()

print(f"1. Best F1 Score: {best_f1} (F1 = {comparison_df.loc[best_f1, 'f1']:.4f})")
print(f"2. Best AUC Score: {best_auc} (AUC = {comparison_df.loc[best_auc, 'auc']:.4f})")

if best_f1 == best_auc:
    print(f"3. Recommended model: {best_f1}")
    print("   This configuration provides the best balance of precision and recall,")
    print("   as well as the best discriminative ability.")
else:
    print(f"3. For balanced performance: {best_f1}")
    print(f"4. For discriminative power: {best_auc}")

# Performance analysis
all_features_f1 = results['All Features']['f1']
top20_f1 = results['Top 20 Features']['f1']
top10_f1 = results['Top 10 Features']['f1']

if top10_f1 > all_features_f1 * 0.95:  # Within 5% of all features
    print(f"\n5. Feature Selection Benefits:")
    print(f"   - Top 10 features achieve {top10_f1/all_features_f1*100:.1f}% of all features performance")
    print(f"   - Significant dimensionality reduction: {len(all_features)} → 10 features")
    print(f"   - Recommended for production: Lower complexity, faster inference")

print(f"\nTop 10 Most Important Features:")
for i, row in importance_top10.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:30} (Score: {row['score']:.2f})")