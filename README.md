# ML Practicals - how to run

pip install pandas numpy scikit-learn matplotlib seaborn openpyxl statsmodels xgboost requests
python make_data.py        # only if you want to regenerate data/ (already included)
python p1b_lda_iris.py     # run any script from inside this folder

| File | Practical | Data |
|---|---|---|
| p1b_lda_iris.py | 1B LDA on Iris | data/Iris.csv (real) |
| p2a_uber_regression.py | 2A Linear/Ridge/Lasso | data/uber.csv (synthetic) |
| p3a_svm_digits.py | 3A SVM digits | built into scikit-learn |
| p4a_kmeans_iris.py | 4A K-Means + elbow | data/Iris.csv (real) |
| p5b_boosting_iris.py | 5B AdaBoost/GBM/XGBoost + voting | data/Iris.csv (real) |
| p6a_maze_qlearning.py | 6A Maze RL | none |
| p7 ... p13 | 7 File formats, 8 Weather API, 9 Cleaning, 10 Wrangling, 11 Matplotlib, 12 Aggregation, 13 Time series | synthetic CSV/XLSX/JSON in data/ |

Note: Iris, Wine and Digits are real. Uber, Telecom, Real-estate, AQI, Retail, Stock
and Sales files are synthetic stand-ins (same column names as the lab files), because
the Kaggle files need a login. To use the real Kaggle file, just replace the CSV in
data/ with the same filename.
Task 8: paste your OpenWeatherMap key into API_KEY for live data.
