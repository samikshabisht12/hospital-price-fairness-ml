# Hospital Price Fairness & Anomaly Detection System

🔗 **Live App:**  
https://hospital-price-fairness-ml-4svznsijzbgwgsl3hxp5b7.streamlit.app/

## Overview
This project predicts fair hospital service prices using machine learning and flags overpriced or anomalous services.

## Features
- Price prediction using Linear Regression
- Fairness score calculation
- Overpriced vs Fair classification
- Anomaly detection using Isolation Forest
- Interactive Streamlit web application

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit
  
## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app2.py

##Project Structure
hospital-price-fairness-ml/
├── data/
│   └── hospital_prices.csv
├── app2.py            # Streamlit application
├── eda.py             # Data analysis & model training
├── price_model.pkl    # Trained ML model
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation


