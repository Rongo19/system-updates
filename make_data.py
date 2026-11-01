# Run once: python make_data.py   -> creates every dataset inside data/
# Iris & Wine come from scikit-learn (real data). The others are realistic
# synthetic stand-ins with the same column names as the Kaggle/lab files.
import json, numpy as np, pandas as pd
from sklearn.datasets import load_iris, load_wine
rng = np.random.default_rng(42)
D = "data/"

# ---- real datasets (Task 1, 4, 5) ----
load_iris(as_frame=True).frame.rename(columns={"target": "species"}).to_csv(D + "Iris.csv", index=False)
load_wine(as_frame=True).frame.to_csv(D + "Wine.csv", index=False)

# ---- Task 2A: Uber fares ----
n = 3000
lat1, lon1 = rng.normal(40.75, .03, n), rng.normal(-73.98, .03, n)
lat2, lon2 = lat1 + rng.normal(0, .04, n), lon1 + rng.normal(0, .04, n)
km = np.sqrt(((lat2 - lat1) * 111) ** 2 + ((lon2 - lon1) * 84) ** 2)
fare = 2.5 + 1.8 * km + rng.normal(0, 1.5, n)
fare[rng.choice(n, 25)] += rng.uniform(40, 120, 25)          # outliers
lat1[rng.choice(n, 10)] = 0                                    # bad coordinates
pd.DataFrame({"pickup_datetime": pd.date_range("2015-01-01", periods=n, freq="3h"),
              "pickup_longitude": lon1, "pickup_latitude": lat1,
              "dropoff_longitude": lon2, "dropoff_latitude": lat2,
              "passenger_count": rng.integers(1, 6, n),
              "fare_amount": fare.round(2)}).to_csv(D + "uber.csv", index=False)

# ---- Task 3B (optional): Social network ads ----
n = 400
age, sal = rng.integers(18, 60, n), rng.integers(15000, 150000, n)
p = 1 / (1 + np.exp(-((age - 38) * .25 + (sal - 75000) / 20000)))
pd.DataFrame({"User ID": np.arange(15600000, 15600000 + n), "Gender": rng.choice(["Male", "Female"], n),
              "Age": age, "EstimatedSalary": sal, "Purchased": (rng.random(n) < p).astype(int)}
             ).to_csv(D + "Social_Network_Ads.csv", index=False)

# ---- Task 7: sales in CSV / Excel / JSON ----
n = 300
s = pd.DataFrame({"OrderID": np.arange(1, n + 1),
                  "Date": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
                  "Category": rng.choice(["Electronics", "Clothing", "Grocery", "Toys"], n),
                  "Quantity": rng.integers(1, 10, n), "UnitPrice": rng.uniform(5, 500, n).round(2)})
s.iloc[:100].to_csv(D + "sales.csv", index=False)
s2 = s.iloc[100:200].copy(); s2.loc[s2.sample(5, random_state=1).index, "UnitPrice"] = np.nan
s2.to_excel(D + "sales.xlsx", index=False)
s.iloc[list(range(200, 300)) + [50, 51]].to_json(D + "sales.json", orient="records")  # 2 duplicates

# ---- Task 8: sample OpenWeatherMap forecast response (offline fallback) ----
t = pd.date_range("2024-10-01", periods=40 * 4, freq="3h")
lst = [{"dt_txt": str(x), "main": {"temp": float(24 + 6 * np.sin(i / 8 * np.pi / 3) + rng.normal(0, 1)),
        "humidity": int(rng.integers(40, 90))}, "wind": {"speed": float(rng.uniform(1, 8))},
        "rain": {"3h": float(rng.choice([0, 0, 0, rng.uniform(.2, 5)]))}} for i, x in enumerate(t)]
json.dump({"city": {"name": "Pune", "coord": {"lat": 18.52, "lon": 73.85}}, "list": lst},
          open(D + "weather_sample.json", "w"))

# ---- Task 9: telecom churn (dirty) ----
n = 1000
c = pd.DataFrame({"customerID": np.arange(n), "gender": rng.choice(["Male", "Female", "male", "FEMALE"], n),
                  "tenure": rng.integers(0, 72, n), "MonthlyCharges": rng.uniform(20, 120, n).round(2),
                  "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .2])})
c["TotalCharges"] = (c.tenure * c.MonthlyCharges).round(2).astype(object)
c["Churn"] = np.where((c.Contract == "Month-to-month") & (rng.random(n) < .5), "Yes", "No")
c.loc[rng.choice(n, 30), "TotalCharges"] = " "; c.loc[rng.choice(n, 20), "MonthlyCharges"] = np.nan
c.loc[rng.choice(n, 8), "MonthlyCharges"] = 999
pd.concat([c, c.iloc[:15]]).to_csv(D + "Telecom_Customer_Churn.csv", index=False)

# ---- Task 10: real estate ----
n = 500
area = rng.integers(500, 3500, n)
r = pd.DataFrame({"Property ID": np.arange(n), "Area (sqft)": area, "Bed rooms": rng.integers(1, 6, n),
                  "Property-Type": rng.choice(["House", "Apartment", "Villa"], n),
                  "Neighborhood": rng.choice(["Kothrud", "Baner", "Hinjewadi", "Wakad"], n),
                  "Sale Date": pd.date_range("2022-01-01", periods=n, freq="2D").strftime("%Y-%m-%d"),
                  "Sale Price $": (area * 120 + rng.normal(0, 30000, n)).round()})
r.loc[rng.choice(n, 25), "Area (sqft)"] = np.nan; r.loc[rng.choice(n, 5), "Sale Price $"] *= 15
r.to_csv(D + "RealEstate_Prices.csv", index=False)

# ---- Task 11: air quality ----
n = 365
d = pd.date_range("2024-01-01", periods=n)
pm25 = 80 + 40 * np.cos((np.arange(n) - 15) / 365 * 2 * np.pi) + rng.normal(0, 12, n)
pm10 = pm25 * 1.6 + rng.normal(0, 15, n)
pd.DataFrame({"Date": d, "PM2.5": pm25.round(1), "PM10": pm10.round(1),
              "CO": rng.uniform(.4, 2.5, n).round(2), "AQI": (pm25 * 1.8 + rng.normal(0, 8, n)).round()}
             ).to_csv(D + "City_Air_Quality.csv", index=False)

# ---- Task 12: retail sales ----
n = 800
pd.DataFrame({"TransactionDate": pd.date_range("2024-01-01", periods=n, freq="6h").strftime("%Y-%m-%d"),
              "Region": rng.choice(["North", "South", "East", "West"], n, p=[.35, .25, .2, .2]),
              "ProductCategory": rng.choice(["Electronics", "Clothing", "Grocery", "Toys"], n),
              "QuantitySold": rng.integers(1, 20, n), "SalesAmount": rng.uniform(50, 2000, n).round(2)}
             ).to_csv(D + "Retail_Sales_Data.csv", index=False)

# ---- Task 13: stock prices ----
n = 750
close = 100 + np.cumsum(rng.normal(.1, 1.5, n))
pd.DataFrame({"Date": pd.bdate_range("2022-01-03", periods=n), "Close": close.round(2),
              "Volume": rng.integers(1_000_000, 5_000_000, n)}).to_csv(D + "Stock_Prices.csv", index=False)
print("datasets created in data/")
