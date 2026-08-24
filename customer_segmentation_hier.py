# IMPORTS

import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score
import seaborn as sns

# STEP 1: LOAD DATASET & DATA HYGIENE CHECKS
df=pd.read_csv('Mall_Customers.csv')

df.info()
print(df.shape)
print(df.describe())


print('the number of nulls:',df.isnull().sum())
print('the number of duplicates are:',df.duplicated().sum())

# STEP 2: FEATURE SELECTION
# Dropping CustomerID (arbitrary index) and Gender (categorical string)
features=['Age','Annual Income (k$)','Spending Score (1-100)']
X_main=df[features]

# STEP 3: FEATURE SCALING (StandardScaler -> Mean=0, Std=1)
# Distance math requires all features to sit on an equal scale
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X_main)
print("Scaled Data Shape:", X_scaled.shape)
print("First row of scaled data:", X_scaled[0])

# STEP 4: HIERARCHICAL CLUSTERING & DENDROGRAM (Ward Linkage)
plt.figure(figsize=(12,6))
linkage_matrix=sch.linkage(X_scaled,method="ward")
dendrogram=sch.dendrogram(linkage_matrix)
plt.title("Customer Segmentation Dendrogram (Ward Linkage)", fontsize=14)
plt.xlabel("Customers (Sample Indices)", fontsize=12)
plt.ylabel("Euclidean Distance (Ward Variance)", fontsize=12)
plt.show()

# STEP 5: OPTIMAL K SELECTION (Silhouette Score for K = 2 to 7)
for k in range(2,8):
    labels=fcluster(linkage_matrix,t=k,criterion='maxclust')

    score=silhouette_score(X_scaled,labels)

    print(f"For K = {k} clusters -> Silhouette Score = {score:.4f}")

# STEP 6: CUT TREE AT OPTIMAL K=6 & PROFILE CLUSTERS
df['cluster']=fcluster(linkage_matrix,t=6,criterion='maxclust')    
cluster_profile = df.groupby("cluster")[
    ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
].mean()
cluster_profile["Customer_Count"] = df["cluster"].value_counts()
print("\n--- 6 Customer Cluster Profiles ---")
print(cluster_profile.round(1))

# STEP 7: MAP BUSINESS PERSONAS & EXPORT RESULTS
persona_dict = {
    1: "Frugal High-Earners",
    2: "Budget / Low-Value",
    3: "Mature Middle-Class",
    4: "Best / VIP Customers",
    5: "Risky / Impulse Buyers",
    6: "Young Middle-Class",
}

df["Customer_Segment"] = df["cluster"].map(persona_dict)
df.to_csv("segmented_customers.csv", index=False)

# STEP 8: VISUALIZE FINAL BUSINESS PERSONAS
plt.figure(figsize=(12,7))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Customer_Segment",
    palette="tab10",
    s=120
)
plt.title(
    "Mall Customer Segmentation — Business Personas (K=6)", fontsize=14, pad=15
)
plt.xlabel("Annual Income (k$)", fontsize=12)
plt.ylabel("Spending Score (1-100)", fontsize=12)
plt.legend(
    title="Customer Segments", bbox_to_anchor=(1.02, 1), loc="upper left"
)

plt.tight_layout()
plt.savefig("customer_segmentation_personas.png")
plt.show()