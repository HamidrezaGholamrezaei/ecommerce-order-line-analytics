# Ecommerce Order-Line Analytics

Analytics project built on order-line level e-commerce data. The EDA module covers revenue, returns, product and channel performance, fulfillment timing, and customer segmentation. Later modules add ML for return prediction, revenue modeling, churn, CLV, and forecasting.

---

## Status

Work in progress. EDA module is published. ML modules will be added as they're completed.

---


## Why order-line data

Most e-commerce analyses work at the order level. Order-line data is more granular and more useful: you can see which products within an order were returned, how discounts apply at the line level, and how fulfillment timing varies by product type.

This project uses that detail to answer questions like:
- Which product families drive realized revenue?
- Where do returns reduce business value?
- How do acquisition channels differ in volume, revenue, and return behavior?
- Are there customer groups with high value but high return risk?
- What signals should be carried into downstream prediction models?

---

## Dataset

This project uses a privacy-safe e-commerce order-line dataset built to reflect realistic business patterns such as returns, seasonality, customer behavior, product variety, discounts, fulfillment timing, and acquisition channels:
- ~4.26 million order-line records across one business year
- Order-line structure with product, category, and customer fields
- Gross and realized revenue, before and after returns
- Return flags and timestamps
- Acquisition channel and campaign metadata
- Fulfillment and delivery timing

The dataset is safe for public use and does not contain real customer or company data.

---

## Data access

The repository includes only a sample dataset for quick review: `data/synthetic_orderline_2024_sample.csv.gz`

The EDA notebook outputs were generated on the full dataset, which is not in the repo due to file size, but it can be regenerated locally:

```bash
pip install pyarrow

python scripts/generate_synthetic_orderline_data.py --rows 4262864 --year 2024 --seed 202404 --output-dir data --sample-csv-rows 100000
```
After generating the full dataset, the notebook will automatically prefer the larger Parquet file if it exists.

---

## Modules

### `0_EDA.ipynb`: Exploratory Analysis

- Data validation and quality checks 
- Revenue distribution: gross vs. realized, with log-scale view for skew
- Return behavior by product family, channel, quantity bucket, and price tier
- Monthly revenue trend and seasonality
- Acquisition channel analysis: volume, retained revenue, and customer quality
- Price vs. return rate relationship
- Delivery delay and fulfillment partner performance
- RFM-R customer segmentation (8 segments)
- Business interpretation throughout

---

## RFM-R customer segmentation

Standard RFM plus a return rate dimension. The four inputs:
- **Recency**: days since last purchase
- **Frequency**: number of orders with at least one non-returned line
- **Monetary**: realized revenue after returns
- **Return rate**: share of order lines returned

This helps separate customers who are genuinely valuable from those who may look valuable based on purchase volume but lose value through returns. The scoring uses an adaptive approach so repeated values, such as zero return rate, are handled consistently instead of being split artificially across different score groups.

Eight segments are produced: Champions, Loyal Value Customers, New/Promising, High Return Risk, At Risk, Hibernating/Lost, Regular Customers, and Full Return Customers. 

---

## Selected findings from EDA

A few things that stood out:

- Overall return rate is 11%. Everyday Apparel reaches 20%, well above every other product family.
- Social media and marketplace channels have the highest return rates (13.7% and 12.9%). Direct web has the lowest, and the highest average realized revenue per customer.
- Delivery delays correlate with returns. Orders taking 11-30 days to arrive return at 16.8% vs around 10% for fast delivery.
- Higher-priced items return more: 9.5% return rate in the lowest price bucket, 13.4% in the highest.
- Revenue is right-skewed. Median order-line realized revenue is 24.44, but the 99th percentile is 168. Averages alone mislead here.
- November and December drive the revenue peak, but return rates also rise in those months. Seasonal planning needs to account for both.

---

## Planned modules

The full project roadmap includes:

| Module | Focus |
|---|---|
| `0_EDA.ipynb` | Exploratory analysis |
| `1_Return_Prediction.ipynb` | Predicting order-line return risk |
| `2_Revenue_Prediction.ipynb` | Estimating realized revenue |
| `3_Customer_Churn.ipynb` | Predicting customer inactivity risk |
| `4_CLV_Prediction.ipynb` | Predicting future customer value |
| `5_Recommendation_Engine.ipynb` | Customer-product recommendations |
| `6_Final_Summary.ipynb` | Consolidated findings |
| `7_Revenue_Forecasting.ipynb` | Monthly revenue forecasting |

Order may shift. Each notebook gets published after review.

---

## Stack

Python, pandas, NumPy, matplotlib, seaborn, Jupyter, pyarrow

---

## How to run
```bash
git clone https://github.com/HamidrezaGholamrezaei/ecommerce-order-line-analytics.git
cd ecommerce-order-line-analytics
python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt 
jupyter notebook
```
Open notebooks from the `notebooks/` folder.

