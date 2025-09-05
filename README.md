# 🩸 Anemia Prediction Web Application

A machine learning-powered web application that predicts anemia in patients based on blood parameters using various classification algorithms.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)



## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Machine Learning Models](#machine-learning-models)
- [Data Visualization](#data-visualization)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This web application uses machine learning algorithms to predict whether a patient has anemia based on key blood parameters. The system provides an intuitive web interface for healthcare professionals and includes comprehensive data visualizations to understand the underlying patterns in the data.

### Key Blood Parameters Used:
- **Gender** (Male/Female)
- **Hemoglobin** levels (g/dL)
- **MCH** (Mean Corpuscular Hemoglobin)
- **MCHC** (Mean Corpuscular Hemoglobin Concentration)
- **MCV** (Mean Corpuscular Volume)

## ✨ Features

- 🎯 **Accurate Predictions**: Multiple ML algorithms with high accuracy
- 🎨 **Modern UI**: Clean, responsive web interface
- 📊 **Data Visualizations**: Comprehensive graphs and charts
- ⚖️ **Balanced Dataset**: Proper data preprocessing and balancing
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🔄 **Real-time Predictions**: Instant results upon form submission

## 🛠 Technologies Used

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-Learn** - Machine learning

### Frontend
- **HTML5** - Markup
- **CSS3** - Styling with gradients and animations
- **JavaScript** - Interactive elements

### Data Visualization
- **Matplotlib** - Static plots
- **Seaborn** - Statistical visualizations

### Machine Learning Models
- Gradient Boosting Classifier (Best performing)
- Random Forest Classifier
- Logistic Regression
- Decision Tree Classifier
- Support Vector Machine
- Gaussian Naive Bayes

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/rvswetha/anemia-prediction-app.git
   cd anemia-prediction-app
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Train the model and generate visualizations**
   ```bash
   python model.py
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Access the application**
   Open your browser and navigate to `http://127.0.0.1:5000`

## 📖 Usage

1. **Home Page**: Enter patient blood parameters in the form
2. **Prediction**: Click "Predict" to get anemia prediction
3. **Results**: View prediction result and comprehensive data visualizations
4. **Analysis**: Explore the generated graphs to understand data patterns

### Sample Input for Testing:
```
Gender: 1 (Female)
Hemoglobin: 8.5
MCH: 22.1
MCHC: 28.9
MCV: 78.2
```

## 🤖 Machine Learning Models

The application compares multiple classification algorithms:

| Model | Accuracy | Use Case |
|-------|----------|----------|
| **Gradient Boosting** | Highest | Final predictions |
| Random Forest | High | Ensemble learning |
| Logistic Regression | Good | Linear relationships |
| SVM | Good | Complex boundaries |
| Decision Tree | Moderate | Interpretability |
| Naive Bayes | Baseline | Probability-based |

### Model Selection Process:
1. **Data Preprocessing**: Handling missing values and data balancing
2. **Feature Selection**: Using relevant blood parameters
3. **Model Training**: Training multiple algorithms
4. **Evaluation**: Comparing accuracy and performance metrics
5. **Selection**: Choosing Gradient Boosting for best results

## 📊 Data Visualization

The application generates seven comprehensive visualizations:

1. **Original Dataset Distribution** - Class imbalance visualization
2. **Balanced Dataset Distribution** - After data preprocessing
3. **Gender Distribution** - Demographics analysis
4. **Hemoglobin Distribution** - Statistical distribution
5. **Hemoglobin by Gender & Status** - Comparative analysis
6. **Feature Relationships** - Pairplot correlation
7. **Correlation Heatmap** - Feature correlation matrix

## 📁 Project Structure

```
anemia-prediction-app/
├── app.py                 # Flask application
├── model.py              # ML model training and visualization
├── model_file.pkl             # Trained model (generated)
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── .gitignore           # Git ignore rules
├── forms/
│   └── anemia.csv       # Dataset
├── templates/
│   ├── index.html       # Home page
│   └── predict.html     # Results page
|   └── error.html     # Error page
└── static/
    ├── Figure_1.png     # Dataset distribution (original)
    ├── Figure_2.png     # Dataset distribution (balanced)
    ├── Figure_3.png     # Gender distribution
    ├── Figure_4.png     # Hemoglobin distribution
    ├── Figure_5.png     # Hemoglobin by gender/status
    ├── Figure_6.png     # Feature relationships
    └── Figure_7.png     # Correlation heatmap
```

## 🎯 Key Learning Outcomes

- **Full-Stack Development**: Frontend and backend integration
- **Machine Learning Pipeline**: Data preprocessing to model deployment
- **Data Visualization**: Creating meaningful insights from data
- **Web Development**: Responsive design and user experience
- **Version Control**: Git workflow and documentation

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@rvswetha](https://github.com/rvswetha)
- LinkedIn: [RV Swetha](https://www.linkedin.com/in/swetha-rv-18992728a/)
- Email: swetha.rv2023@vitstudent.ac.in

## 🙏 Acknowledgments

- Inspiration from healthcare ML applications
- Flask documentation and community

---

⭐ **Star this repository if you found it helpful!**