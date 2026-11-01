# ML Practicals — Jupyter Notebook Edition

This folder contains 13 independent `.ipynb` notebooks converted from the supplied Python practicals.

## How to use
1. Extract the ZIP.
2. Open the folder in Jupyter Notebook, JupyterLab, VS Code, or Google Colab.
3. Open any practical notebook and run its cells from top to bottom.

Each notebook creates its required synthetic/sample CSV, Excel, or JSON data in a local `data/` folder before running the practical. Iris is generated from scikit-learn. SVM Digits and Q-Learning do not need external files.

## Requirements
`pandas numpy matplotlib seaborn scikit-learn statsmodels openpyxl requests xgboost`

Install if needed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels openpyxl requests xgboost
```

## Notes
- Weather practical works offline using a generated sample JSON. Add an OpenWeatherMap API key in the `API_KEY` variable to request live forecast data.
- XGBoost is used in Practical 5B and must be installed.
- Synthetic datasets are educational stand-ins, not original Kaggle datasets.
