# Modèle de scoring pour accorder un credit

**Link Dashboard:** https://dashboard-p7-scoring.herokuapp.com

**Code Dashboard:** P7_01_dashboard.py

# Folders:

**Datasets:** train and test datasets. Saved from notebooks.

**Images:** images used in the dashboard.

**Model:** Final model (LightGBM with custom metric) saved from notebooks.

**Notebooks:** 3 notebooks derived from some on Kaggle. In 'prétraitement' notebook initial datasets are merged, cleaned and some feature engineering variables are added. The output are train and test datasets that will be used in the second notebook 'modèle'. Here the LightGBM model is fitted using different hyperparameters, chosen evaluating ROC AUC, recalls, precisions and customs metrics. In the last notebook 'modèle final' the final model is used to make predictions on the test set and is explained with SHAP Values.

# Others files:

Procfile, requirements and setup files are necessary for the deployment on Heroku.
