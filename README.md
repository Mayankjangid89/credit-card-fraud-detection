# 🛡️ Credit Card Fraud Detection using Artificial Neural Networks

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Accuracy](https://img.shields.io/badge/Accuracy-99.92%25-brightgreen)

An AI-powered credit card fraud detection system achieving 99.92% accuracy on real-world transaction data.

## 🌟 Features

- ⚡ Real-time fraud detection (<1ms per transaction)
- 🎯 99.92% accuracy with 91.45% precision
- 🎨 Beautiful neural-themed web interface
- 📊 Batch processing for 1000+ transactions
- 🔴 Interactive real-time payment demo
- 🧠 5-layer neural network with 240+ neurons

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or 3.10
- 4GB RAM minimum

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. **Create virtual environment**
```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
- Go to [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Download `creditcard.csv`
- Place in `data/` folder

5. **Train the model**
```bash
python train_model.py
```

6. **Run the application**
```bash
streamlit run app.py
```

## 📊 Dataset

- **Source:** Kaggle Credit Card Fraud Detection Dataset
- **Transactions:** 284,807
- **Features:** 30 (Time, V1-V28 PCA components, Amount)
- **Fraud Rate:** 0.172% (extreme imbalance)

## 🧠 Model Architecture

- **Type:** Sequential Feedforward Neural Network
- **Layers:** 5 dense layers with dropout
- **Neurons:** 128 → 64 → 32 → 16 → 1
- **Parameters:** 14,849 trainable
- **Activation:** ReLU (hidden), Sigmoid (output)

## 📈 Performance

| Metric | Value |
|--------|-------|
| Accuracy | 99.92% |
| Precision | 91.45% |
| Recall | 75.51% |
| F1-Score | 82.5% |

## 💻 Technologies

- Python 3.10
- TensorFlow 2.15
- Keras
- Streamlit 1.28
- Pandas, NumPy, Scikit-learn
- Plotly (visualizations)

## 👥 Team

- Mayank Jangid
- Aayushi soni
- Ishitaba umat

## 📝 License

MIT License

## 🙏 Acknowledgments

- Dataset from Kaggle
- Inspired by real-world fraud detection systems