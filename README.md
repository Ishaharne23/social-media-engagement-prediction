# Social Media Post Virality and Engagement Level Prediction

ML mini project comparing SVM, Decision Tree, Naive Bayes and Random Forest.

## Run
    pip install -r requirements.txt
    python pipeline.py

## Contents
- data/        dataset (5,000 posts, 20 columns)
- pipeline.py  EDA, preprocessing, 4-model training and evaluation
- figures/     report figures
- results/     metrics CSV/JSON
- ui/          engagement_predictor.html (open in any browser)
- report/      10-page Word report

## Method notes
- Target: Engagement_Level = quartile bins of Engagement_Rate (Low/Medium/High/Viral)
- Leakage control: Likes, Comments, Shares, Saves, Views excluded (Engagement_Rate is derived from them)
- Pipeline: OneHotEncoder (categorical) + StandardScaler (numeric), 80/20 stratified split, 5-fold CV
