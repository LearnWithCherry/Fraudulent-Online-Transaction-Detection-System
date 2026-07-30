# Fraudulent-Online-Transaction-Detection-System
---
This Fraud Detection System uses Random Forest to classify transactions as genuine or fraudulent based on amount, frequency, and location. Built with Python and Flask, it provides a web interface for instant predictions.
# Fraudulent Online Transaction Detection System

A Machine Learning based web application that detects whether an online transaction is **Fraudulent** or **Legitimate** using transaction details such as amount, transaction velocity, and location.
---
## Features

* Real-time fraud prediction
* Random Forest Classifier model
* Interactive dashboard UI
* Risk score and confidence level
* Performance metrics visualization
* Confusion matrix results

##  Machine Learning Model

This project uses **Random Forest Classifier** because it provides:

* High accuracy
* Reduced overfitting
* Strong classification performance
* Reliable results on tabular data

##  Input Features

* Transaction Amount
* Transaction Velocity / Frequency
* Geofence Offset / Distance
* Origin Location

##  Tech Stack

* Python
* Flask
* Scikit-learn
* Pandas
* NumPy
* HTML
* CSS
* JavaScript

##  Project Structure

```text
app.py
model.pkl
templates/
static/
requirements.txt
README.md
```

## How to Run

1. Clone repository

```bash
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run Flask app

```bash
python app.py
```

4. Open browser

```text
http://127.0.0.1:5000/
```

##  Model Metrics

* Accuracy: ~99%
* Precision: High
* Recall: High
* F1 Score: Strong balanced performance

##  Use Cases

* Banking fraud detection
* Credit/debit card monitoring
* E-commerce payment screening
* Real-time suspicious transaction alerts

## Future Improvements

* Live transaction API integration
* Deep learning models
* Real banking dataset
* User authentication
* Cloud deployment

##Authors
Rajat Bhardwaj
