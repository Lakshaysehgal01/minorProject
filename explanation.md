# Smart Energy Microgrid ML Project: Viva Explanation

## 1. Project Overview

This project predicts two hourly energy values for households in a smart microgrid:

1. **Solar generation**: how much electricity the household's solar panels produce.
2. **Energy consumption**: how much electricity the household uses.

These predictions can later help a trading or scheduling system decide which household has surplus energy to sell, which household needs energy, and how much energy must be imported from or exported to the main grid.

The project implements the **forecasting layer**. The actual peer-to-peer trading, pricing, matching, and settlement logic is outside the current scope.

The pipeline is:

```text
Synthetic data generation
        -> Exploratory data analysis
        -> Train multiple forecasting models
        -> Compare models using metrics
        -> Save trained models
        -> Serve predictions through FastAPI
```

## 2. Dataset

The data-generation code is in `ml-service/data/generate_data.py`.

The intended dataset contains:

- 25 households
- 365 days
- 24 hours per day
- 219,000 rows

Calculation:

$$
25 \times 365 \times 24 = 219000
$$

Each row represents one household at one hour.

### Important columns

| Column                        | Meaning                           |
| ----------------------------- | --------------------------------- |
| `timestamp`                   | Date and hour                     |
| `hour`                        | Hour of day, from 0 to 23         |
| `day_of_week`                 | Monday = 0 and Sunday = 6         |
| `day_of_year`                 | Day number from 1 to 365          |
| `month`                       | Month number                      |
| `household_id`                | Household identifier              |
| `consumption_profile`         | Small, medium, or large household |
| `consumption_profile_encoded` | Small = 0, medium = 1, large = 2  |
| `temperature`                 | Temperature in Celsius            |
| `solar_irradiance`            | Available sunlight from 0 to 1    |
| `panel_capacity_kw`           | Solar panel capacity              |
| `solar_generation_kwh`        | Target value 1                    |
| `consumption_kwh`             | Target value 2                    |

### Solar generation

Solar generation is mainly determined by:

- Hour of the day
- Solar irradiance
- Panel capacity
- Season
- Random noise

The basic relationship is:

$$
\text{Solar Generation}
=
\text{Panel Capacity}
\times
\text{Solar Irradiance}
+
\text{Noise}
$$

Solar generation is forced to zero during night hours. It normally increases in the morning, reaches its maximum near 1 PM, and decreases in the evening. Summer generation is higher than winter generation because the daylight period and sunlight intensity are larger.

### Energy consumption

Consumption is generated from:

- Household baseline load
- Morning peak around 8 AM
- Evening peak around 7 PM
- Extra daytime usage on weekends
- Household profile size
- Random noise

Large households consume more than medium households, and medium households consume more than small households.

The data therefore contains daily, weekly, and yearly seasonality, household-specific behavior, and random variation.

## 3. Why Synthetic Data Was Used

Synthetic data was used because real smart-meter data can be difficult to obtain and may have privacy restrictions. Synthetic data provides full control over:

- Number of households
- Time period
- Seasonal behavior
- Household profiles
- Noise level
- Reproducibility

The random seed is fixed at `42`, so the same data can be regenerated.

The limitation is that high accuracy on synthetic data does not guarantee the same accuracy on real-world smart-meter data.

## 4. Exploratory Data Analysis

EDA is implemented in `ml-service/notebooks/eda.py`.

It performs:

1. Missing-value checking.
2. Statistical summaries using `describe()`.
3. Average solar generation and consumption by hour.
4. A correlation heatmap.
5. Consumption boxplots by hour.
6. Monthly seasonal analysis.
7. Consumption comparison between small, medium, and large households.

The purpose is to verify that the generated data behaves realistically before training models.

Expected observations include:

- Solar generation is near zero at night.
- Consumption increases in the morning and evening.
- Large households consume more than small households.
- Solar generation and temperature vary by season.

## 5. Train/Test Splitting

The training code is in `ml-service/train.py`.

Because this is time-series data, a random train/test split is not appropriate. A random split could place future observations in the training data and earlier observations in the test data. This is called **data leakage**.

The baseline models use:

- First 80% of calendar days for training.
- Last 20% of calendar days for testing.

The advanced LSTM and ensemble models use:

- 70% training
- 10% validation
- 20% testing

This simulates real deployment:

```text
Past data       -> train the model
Recent data     -> tune the model
Future data     -> evaluate the model
```

The split is performed by complete calendar days, so rows from the same day are not divided between training and testing.

## 6. Models Used

The project includes:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. XGBoost
5. LightGBM
6. Prophet
7. SARIMAX fallback
8. LSTM
9. Stacking Ensemble using Ridge Regression

The baseline models predict each target independently. One model predicts solar generation and another predicts consumption.

## 7. Linear Regression

Linear Regression assumes that the target can be represented as a weighted sum of the input features:

$$
\hat{y}
=
 b_0+b_1x_1+b_2x_2+\dots+b_nx_n
$$

For example, it tries to learn that higher irradiance and larger panel capacity increase solar generation, while household profile and hour affect consumption.

### Why it was used

It is a simple and fast baseline. It answers the question:

> Can a simple linear relationship solve the problem?

### Limitation

Energy data is not completely linear. Solar generation has a bell-shaped daytime curve, and consumption contains morning and evening peaks. Linear Regression cannot naturally represent all these non-linear interactions, so it is expected to be weaker than tree-based models.

## 8. Decision Tree Regressor

A Decision Tree repeatedly splits the data into smaller groups. For example, it may first check whether the hour is during daylight, then whether irradiance is high, and finally produce a prediction from the relevant leaf.

### Why it was used

Decision Trees can learn:

- Non-linear relationships
- Hour-based thresholds
- Household differences
- Feature interactions

They are also relatively easy to explain.

### Limitation

A single tree can overfit the training data by memorizing very specific patterns.

## 9. Random Forest Regressor

A Random Forest creates many Decision Trees and averages their predictions:

$$
\text{Final Prediction}
=
\frac{1}{N}\sum_{i=1}^{N}\text{Tree Prediction}_i
$$

The project uses approximately 200 trees, a maximum depth of 15, and a minimum leaf size of 25. These parameters control model complexity and reduce overfitting.

### Why it was used

Random Forest:

- Captures non-linear patterns.
- Learns feature interactions.
- Is more stable than a single tree.
- Works well on tabular data.
- Does not require feature scaling.

It performs well because solar generation depends strongly on irradiance and panel capacity, while consumption depends on hour, household, and profile.

The currently stored `best_solar_model.pkl` and `best_consumption_model.pkl` artifacts are both Random Forest models.

## 10. XGBoost

XGBoost is a gradient-boosted decision-tree algorithm. It builds trees sequentially. Each new tree focuses on correcting errors made by the previous trees.

Conceptually:

```text
First prediction -> calculate errors
Second tree      -> correct previous errors
Third tree       -> correct remaining errors
Final prediction -> combined result
```

### Why it was used

XGBoost is effective for structured tabular data because it learns complex non-linear relationships and feature interactions. It is used as:

- A baseline comparison model.
- A dedicated API model.
- A base model in the stacking ensemble.

### Random Forest versus XGBoost

| Random Forest                        | XGBoost                               |
| ------------------------------------ | ------------------------------------- |
| Trees are mostly built independently | Trees are built sequentially          |
| Mainly reduces variance              | Corrects errors step by step          |
| Generally easier to tune             | Often more sensitive to parameters    |
| Strong general-purpose model         | Often highly accurate on tabular data |

## 11. LightGBM

LightGBM is another gradient-boosted tree model. It is similar in purpose to XGBoost but uses a different implementation designed to be fast and memory-efficient.

It is implemented in `ml-service/models/train_lightgbm.py`.

### Why it was used

LightGBM provides a direct comparison with XGBoost using:

- The same features.
- The same time-based split.
- The same number of estimators.
- The same evaluation metrics.

LightGBM commonly grows trees leaf by leaf, while XGBoost commonly grows them level by level. The better model depends on the data and parameter settings.

## 12. Prophet

Prophet is implemented in `ml-service/models/train_prophet.py`.

Prophet is a time-series forecasting model that decomposes a series into components such as:

- Trend
- Daily seasonality
- Weekly seasonality
- Yearly seasonality

It uses:

- `ds` for the timestamp.
- `y` for the target.

### Why Prophet was used

Prophet provides a classical time-series benchmark. It answers:

> How well can a pure time-series model perform without using direct causal features such as solar irradiance or panel capacity?

Prophet is intentionally univariate in this project. It does not receive temperature, irradiance, panel capacity, or household profile features.

### Scope limitation

Prophet is evaluated on only three representative households:

- One small household.
- One medium household.
- One large household.

Therefore, it is a benchmark and is not a general pooled deployment model in this project.

### SARIMAX fallback

If Prophet cannot be installed or its Stan backend cannot run, the code falls back to SARIMAX from `statsmodels`.

SARIMAX uses autoregressive behavior, moving-average behavior, and seasonal behavior. The seasonal period is 24 because the data is hourly and the main fallback seasonality is daily.

## 13. LSTM

The LSTM implementation is in `ml-service/models/train_lstm.py`.

LSTM means **Long Short-Term Memory**. It is a type of Recurrent Neural Network designed for sequential data.

### Why LSTM was used

Energy values depend on previous time steps. The previous 24 hours can help predict the next hour because energy usage has daily patterns and temporal continuity.

The model is pooled, meaning one model is trained using all 25 households instead of training 25 independent LSTM models.

### Sliding-window input

The LSTM uses a 24-hour window:

```text
Hours 1 to 24 -> predict hour 25
Hours 2 to 25 -> predict hour 26
Hours 3 to 26 -> predict hour 27
```

### Cyclical time encoding

Hour, day of week, and day of year are converted to sine and cosine values:

$$
 hour\_sin=\sin\left(\frac{2\pi\times hour}{24}\right)
$$

$$
 hour\_cos=\cos\left(\frac{2\pi\times hour}{24}\right)
$$

This helps the model understand that 23:00 and 00:00 are close together. Ordinary integer encoding would make them appear far apart.

### Scaling

The LSTM uses Min-Max scaling:

$$
 x_{scaled}=\frac{x-x_{min}}{x_{max}-x_{min}}
$$

Scaling keeps features with different numerical ranges stable during neural-network training. The scalers are fitted only on training data to avoid leakage.

### LSTM architecture

```text
LSTM(64, return_sequences=True)
        -> Dropout(0.2)
        -> LSTM(32)
        -> Dropout(0.2)
        -> Dense(16, ReLU)
        -> Dense(1)
```

The model uses:

- Adam optimizer.
- Learning rate `0.001`.
- Mean Squared Error loss.
- Up to 50 epochs.
- Early stopping with patience of 5 validation-loss failures.

Dropout helps reduce overfitting. The final Dense layer outputs one continuous prediction.

## 14. Stacking Ensemble

The ensemble is implemented in `ml-service/models/train_ensemble.py`.

It combines XGBoost and LSTM in both target-specific stacks. LightGBM and CatBoost are added independently when their earlier R2 beats XGBoost.

In the completed run, the actual base-model orders are:

- Solar: `XGBoost -> LightGBM -> LSTM`
- Consumption: `XGBoost -> LightGBM -> CatBoost -> LSTM`

The final combination is performed by a Ridge Regression meta-learner.

### Why use an ensemble?

Different models learn different patterns:

- XGBoost learns tabular feature relationships.
- LightGBM learns another boosted-tree representation.
- LSTM learns sequential relationships.

Combining them can improve robustness.

### Ensemble process

1. Train the base models on the training block.
2. Generate predictions on the validation block.
3. Use those predictions as input features for Ridge Regression.
4. Evaluate the complete ensemble on the untouched test block.

The Ridge model learns weights for the base predictions:

$$
\hat{y}
=
w_1(\text{XGBoost})
+w_2(\text{LightGBM})
+w_3(\text{CatBoost})
+w_4(\text{LSTM})
+b
$$

Ridge regularization keeps these weights stable.

### Why Prophet is not in the ensemble

Prophet was trained only for three households, while the ensemble is designed to represent all 25 households. Including Prophet would make the pooled comparison unfair.

## 15. Evaluation Metrics

The project calculates RMSE, MSE, MAE, and R².

### MSE: Mean Squared Error

$$
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

MSE is the average squared prediction error.

- Lower is better.
- Large errors are punished heavily.
- Its units are squared, so it is less intuitive to interpret directly.

### RMSE: Root Mean Squared Error

$$
RMSE=\sqrt{MSE}
$$

- Lower is better.
- It has the same unit as the target, approximately kWh here.
- It is sensitive to large errors.

If RMSE is `0.16`, the typical prediction error is roughly around `0.16 kWh`, although RMSE is not exactly the average error.

### MAE: Mean Absolute Error

$$
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

- Lower is better.
- It is easy to understand.
- It represents the average absolute prediction error.
- It is less affected by outliers than RMSE.

### R²: Coefficient of Determination

$$
R^2=1-\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}
$$

Interpretation:

- `R² = 1`: perfect prediction.
- `R² = 0`: no better than predicting the average.
- `R² < 0`: worse than predicting the average.

For example, `R² = 0.936` means the model explains approximately 93.6% of the variation in the target on that test set.

## 16. Completed Comparison Results

The complete trained run is recorded in `ml-service/results/solar_model_comparison.csv` and `ml-service/results/consumption_model_comparison.csv`. Values below are RMSE, MAE, and R2; lower RMSE/MAE is better and higher R2 is better.

### Solar generation

| Model                     |   RMSE |    MAE |       R2 |
| ------------------------- | -----: | -----: | -------: |
| Random Forest             | 0.0345 | 0.0164 | 0.999722 |
| LightGBM                  | 0.0360 | 0.0185 | 0.999698 |
| XGBoost                   | 0.0361 | 0.0186 | 0.999697 |
| CatBoost                  | 0.0374 | 0.0199 | 0.999674 |
| Ensemble                  | 0.0379 | 0.0197 | 0.999664 |
| Decision Tree             | 0.0499 | 0.0244 | 0.999420 |
| LSTM                      | 0.2877 | 0.1489 | 0.980654 |
| Linear Regression         | 0.6949 | 0.5690 | 0.887537 |
| Prophet, small household  | 0.3496 | 0.2501 | 0.746522 |
| Prophet, medium household | 1.1685 | 0.8759 | 0.743935 |
| Prophet, large household  | 1.3428 | 1.0470 | 0.597353 |

### Consumption

| Model                     |   RMSE |    MAE |       R2 |
| ------------------------- | -----: | -----: | -------: |
| Ensemble                  | 0.1503 | 0.1200 | 0.959029 |
| LightGBM                  | 0.1502 | 0.1200 | 0.959016 |
| CatBoost                  | 0.1505 | 0.1201 | 0.958893 |
| Random Forest             | 0.1515 | 0.1210 | 0.958316 |
| XGBoost                   | 0.1526 | 0.1218 | 0.957734 |
| LSTM                      | 0.1621 | 0.1292 | 0.952319 |
| Decision Tree             | 0.2154 | 0.1722 | 0.915750 |
| Prophet, large household  | 0.3293 | 0.2656 | 0.868200 |
| Prophet, medium household | 0.3846 | 0.3262 | 0.607153 |
| Prophet, small household  | 0.3145 | 0.2521 | 0.490725 |
| Linear Regression         | 0.5403 | 0.4279 | 0.469901 |

The files are:

- `ml-service/models/solar_generation_kwh_comparison.csv`
- `ml-service/models/consumption_kwh_comparison.csv`

### Why solar accuracy is higher

Solar generation is almost directly determined by solar irradiance and panel capacity, and both are supplied as input features. This makes solar generation relatively easy to predict.

### Why consumption is harder

Consumption contains human behavior, morning and evening habits, household differences, weekend behavior, and random noise. Therefore, it is more difficult to predict.

## 17. Current Trained Repository State

The complete run has now been generated from the current 13-column, 219,000-row dataset. The repository contains:

- Metric JSON files for every active model, both targets, and the ensemble.
- Standalone XGBoost, LightGBM, and CatBoost artifacts.
- Solar and consumption LSTM models plus their target and feature scalers.
- Ensemble XGBoost, LightGBM, and CatBoost artifacts where selected.
- Solar and consumption Ridge meta-learners and `ensemble_*_config.json` files.
- Final comparison CSVs and R2 charts under `ml-service/results/`.

The older `ml-service/models/solar_generation_kwh_comparison.csv` and `consumption_kwh_comparison.csv` files still contain SVR rows from an earlier run. SVR is commented out in the current `train.py`, so it is not part of the newly generated comparison results.

The completed training order was:

```bash
cd ml-service

./venv/bin/python data/generate_data.py
./venv/bin/python notebooks/eda.py
./venv/bin/python train.py
./venv/bin/python models/train_lightgbm.py
./venv/bin/python models/train_prophet.py
./venv/bin/python models/train_lstm.py
./venv/bin/python models/train_ensemble.py
./venv/bin/python results/model_comparison.py
```

The LSTM and ensemble steps require TensorFlow. The final comparison script reads the metric JSON files and does not retrain models.

## 18. FastAPI Prediction Service

The API is implemented in `ml-service/api/main.py`.

Available endpoints:

```text
GET  /health
POST /predict/solar
POST /predict/consumption
```

### XGBoost and LightGBM requests

These models accept one row of input features.

Solar features include:

```text
hour
day_of_week
day_of_year
month
household_id
temperature
solar_irradiance
panel_capacity_kw
```

Consumption features include:

```text
hour
day_of_week
household_id
temperature
consumption_profile_encoded
```

### LSTM and Ensemble requests

These models require exactly 24 historical rows because the LSTM needs a 24-hour sequence. Each row contains the observed target value for that past hour.

### Prophet requests

Prophet is not served through the API because it was trained only for three benchmark households. The API returns an HTTP 400 error when Prophet is requested.

### Validation

Pydantic validates the request values, for example:

- Hour must be between 0 and 23.
- Day of week must be between 0 and 6.
- Solar irradiance must be between 0 and 1.
- Consumption profile encoding must be between 0 and 2.
- LSTM requests must contain exactly 24 rows.

Invalid input returns a validation error.

## 19. Common Viva Questions and Answers

### Why is this a regression problem?

Because the outputs are continuous numerical values measured in kWh. It is not classification because the output is not a category.

### Why not use a random train/test split?

Because future observations could enter the training data, creating data leakage and unrealistically high scores.

### Why do tree models perform well?

The data is tabular and contains non-linear relationships and feature interactions. Tree models can learn those relationships effectively without feature scaling.

### Why is Linear Regression weaker?

It mainly learns additive linear relationships and cannot naturally model complex peaks and interactions.

### Why is LSTM useful?

Energy data is sequential. The previous 24 hours can help predict the next hour.

### Why use cyclical encoding?

Time is circular. 23:00 is close to 00:00, Sunday is close to Monday, and December is close to January. Sine and cosine encoding represents this correctly.

### Why use an ensemble?

Different models learn different patterns. The ensemble combines their predictions and allows Ridge Regression to learn the best combination.

### Why is Prophet not in the ensemble?

Prophet was trained only on three households, while the ensemble is pooled across all 25 households.

### What does a high R² mean?

It means the model explains a large amount of variation in the target compared with simply predicting the average.

### What is the main limitation?

The data is synthetic. Also, temperature and solar irradiance are assumed to be known inputs, while a real system would need weather forecasts for them.

### Is the trading system implemented?

No. The project produces forecasts that a future trading, matching, pricing, or scheduling module could consume.

## 20. Short Viva Answer

> Our project forecasts hourly solar generation and household consumption for a smart microgrid. We generate synthetic data for 25 households with daily, weekly, and yearly patterns. We use a chronological train-test split to avoid future-data leakage. We compare Linear Regression, Decision Tree, Random Forest, XGBoost, LightGBM, Prophet, and LSTM models. Random Forest and boosting models are suitable for tabular features, Prophet provides a classical univariate time-series benchmark, and LSTM learns patterns from the previous 24 hours. Finally, a Ridge stacking ensemble combines predictions from XGBoost, LightGBM, and LSTM. We evaluate the models using RMSE, MSE, MAE, and R². The predictions are exposed through a FastAPI service for future use by a microgrid trading or scheduling system.
