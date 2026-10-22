"""Social Media Post Virality and Engagement Level Prediction.
EDA -> preprocessing -> SVM / Decision Tree / Naive Bayes / Random Forest -> metrics + figures.
Run from the project folder:  python pipeline.py
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.titleweight": "bold"})
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 1. Load
df = pd.read_csv("data/social_media_engagement_dataset.csv")

# 2. Target: quartile bins of Engagement_Rate
labels = ["Low", "Medium", "High", "Viral"]
df["Engagement_Level"] = pd.qcut(df["Engagement_Rate"], q=4, labels=labels)

# 3. Leakage check: Engagement_Rate ~ (Likes+Comments+Shares+Saves)/Follower_Count*100
calc = (df["Likes"] + df["Comments"] + df["Shares"] + df["Saves"]) / df["Follower_Count"] * 100
print("Leakage correlation:", round(calc.corr(df["Engagement_Rate"]), 4))

# 4. Features (pre-publication only)
cat = ["Platform", "Content_Type", "Category", "Day_of_Week", "Sentiment", "Influencer_Tier"]
num = ["Follower_Count", "Hour_of_Day", "Hashtag_Count", "Content_Length"]
boo = ["Has_Media", "Is_Verified"]
X = df[cat + num + boo].copy()
for b in boo:
    X[b] = X[b].astype(int)
le = LabelEncoder()
y = le.fit_transform(df["Engagement_Level"].astype(str))
class_names = le.classes_.tolist()

# 5. EDA figures
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
sns.countplot(x=df["Engagement_Level"].astype(str), order=labels, palette="viridis", ax=ax[0])
ax[0].set_title("Engagement Level Class Distribution"); ax[0].set_xlabel("Engagement Level"); ax[0].set_ylabel("Number of Posts")
sns.countplot(x=df["Platform"], order=df["Platform"].value_counts().index, palette="magma", ax=ax[1])
ax[1].set_title("Posts per Platform"); ax[1].tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig("figures/fig1_eda_distribution.png", bbox_inches="tight"); plt.close()

plt.figure(figsize=(6.5, 5.2))
sns.heatmap(df[num + ["Engagement_Rate"]].corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Heatmap: Structural Features vs. Engagement Rate")
plt.tight_layout(); plt.savefig("figures/fig2_correlation_heatmap.png", bbox_inches="tight"); plt.close()

plt.figure(figsize=(7, 4.2))
sns.boxplot(x="Platform", y="Engagement_Rate", data=df, palette="Set2", showfliers=False)
plt.yscale("log"); plt.title("Engagement Rate Distribution by Platform (log scale)")
plt.tight_layout(); plt.savefig("figures/fig3_engagement_by_platform.png", bbox_inches="tight"); plt.close()

# 6. Split + preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ("num", StandardScaler(), num + boo),
])

# 7. Models
models = {
    "Support Vector Machine": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
}
results, cms = {}, {}
for name, clf in models.items():
    pipe = Pipeline([("prep", prep), ("clf", clf)]).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    auc = roc_auc_score(label_binarize(y_test, classes=range(4)), proba, average="macro", multi_class="ovr")
    cv = cross_val_score(Pipeline([("prep", prep), ("clf", clf)]), X, y, cv=5, scoring="accuracy", n_jobs=-1)
    results[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, pred, average="macro", zero_division=0),
        "F1_Score": f1_score(y_test, pred, average="macro", zero_division=0),
        "ROC_AUC": auc, "CV_Mean_Accuracy": cv.mean(), "CV_Std": cv.std(),
    }
    cms[name] = confusion_matrix(y_test, pred)
    print(f"{name:24s} acc={results[name]['Accuracy']:.3f} f1={results[name]['F1_Score']:.3f} auc={auc:.3f} cv={cv.mean():.3f}")

res = pd.DataFrame(results).T
res.to_csv("results/model_comparison_results.csv")
json.dump(results, open("results/results.json", "w"), indent=2)

# 8. Comparison chart
plot_df = res[["Accuracy", "Precision", "Recall", "F1_Score"]].reset_index().melt(id_vars="index")
plot_df.columns = ["Model", "Metric", "Score"]
plt.figure(figsize=(9, 5))
ax = sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette="tab10")
plt.title("Model Performance Comparison: SVM vs Decision Tree vs Naive Bayes vs Random Forest")
plt.ylim(0, 1); plt.xlabel(""); plt.xticks(rotation=10)
plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
plt.tight_layout(); plt.savefig("figures/fig4_model_comparison.png", bbox_inches="tight"); plt.close()

# 9. Confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5))
for a, (name, cm) in zip(axes.flat, cms.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names, ax=a)
    a.set_title(name); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
plt.suptitle("Confusion Matrices — All 4 Models", fontweight="bold", y=1.01)
plt.tight_layout(); plt.savefig("figures/fig5_confusion_matrices.png", bbox_inches="tight"); plt.close()

# 10. Feature importance + per-class report (Random Forest)
rf = Pipeline([("prep", prep), ("clf", models["Random Forest"])]).fit(X_train, y_train)
names = [f.replace("cat__", "").replace("num__", "").replace("_", " ") for f in rf.named_steps["prep"].get_feature_names_out()]
fi = pd.DataFrame({"feature": names, "importance": rf.named_steps["clf"].feature_importances_}).sort_values("importance", ascending=False).head(12)
plt.figure(figsize=(7.5, 5))
sns.barplot(data=fi, y="feature", x="importance", palette="crest")
plt.title("Top 12 Feature Importances (Random Forest)"); plt.ylabel("")
plt.tight_layout(); plt.savefig("figures/fig6_feature_importance.png", bbox_inches="tight"); plt.close()
fi.to_csv("results/feature_importance.csv", index=False)
rep = classification_report(y_test, rf.predict(X_test), target_names=class_names, output_dict=True, zero_division=0)
pd.DataFrame(rep).T.round(3).to_csv("results/rf_per_class_report.csv")
print("Done. See figures/ and results/")
