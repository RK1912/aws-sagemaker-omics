# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
import os

def run_eda(df, target_col='Label', output_dir='plots'):
    
    outdir = os.path.join("results", output_dir)
    os.makedirs(outdir, exist_ok=True)  

    # --- 1. Label Distribution ---
    plt.figure()
    sns.countplot(x=target_col, data=df)
    plt.title("Class Distribution")
    plt.savefig(f"{outdir}/label_distribution.png")
    plt.close()
    print("✅ Label distribution saved.")

    # --- 2. PCA 2D Visualization ---
    numeric_df = df.select_dtypes(include='number')
    features = numeric_df.columns.tolist()

    pca = PCA(n_components=2)
    components = pca.fit_transform(numeric_df)

    pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    pca_df['Label'] = df[target_col].values

    plt.figure()
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Label')
    plt.title("PCA - 2 Component Projection")
    plt.savefig(f"{outdir}/pca_scatter.png")
    plt.close()
    print("✅ PCA scatter plot saved.")

    # --- 3A. Outlier Detection using Isolation Forest ---
    iso = IsolationForest(contamination=0.05, random_state=42)
    yhat = iso.fit_predict(numeric_df)
    df['Outlier_IF'] = yhat

    plt.figure()
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=yhat, palette='coolwarm')
    plt.title("Outliers via Isolation Forest (on PCA)")
    plt.savefig(f"{outdir}/outliers_isolation_forest.png")
    plt.close()
    print("✅ Isolation Forest outlier plot saved.")

    # --- 3B. Z-score outliers ---
    z_scores = zscore(numeric_df)
    z_abs = pd.DataFrame(abs(z_scores), columns=numeric_df.columns)
    outlier_counts = (z_abs > 3).sum(axis=1)
    df['Outlier_zscore'] = outlier_counts > 1

    plt.figure()
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=df['Outlier_zscore'], palette='coolwarm')
    plt.title("Outliers via Z-score (on PCA)")
    plt.savefig(f"{outdir}/outliers_zscore.png")
    plt.close()
    print("✅ Z-score outlier plot saved.")

    # --- Optional: Summary statistics ---
    df.describe().to_csv("results/summary_stats.csv")
    print("Summary statistics saved.")

    return df  # In case caller wants to use this processed df
