from flask import Flask, render_template, jsonify, request # type: ignore

import logging

import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score, roc_auc_score # type: ignore
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt # type: ignore

import seaborn as sns # type: ignore
import os

app = Flask(__name__, static_folder='static')


# Configure logging
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'app.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_and_process_data():
    try:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'dataset.csv')
        logger.info(f"Attempting to load dataset from: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"Dataset file not found at: {file_path}")
            raise FileNotFoundError(f"Dataset not found at {file_path}")
        
        logger.info("Loading dataset...")
        # Force reload of dataset on each request
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded dataset with {len(data)} rows (fresh load)")

    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

    # Process the loaded data
    data.fillna(0, inplace=True)

    irrelevant_columns = ['Ad Topic Line', 'City', 'Country', 'Timestamp']
    data.drop(irrelevant_columns, axis=1, inplace=True)
    
    categorical_columns = data.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        data = pd.get_dummies(data, columns=categorical_columns)
    
    X = data.drop('Clicked on Ad', axis=1)
    y = data['Clicked on Ad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Generate visualizations
    try:
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Clicked on Ad', data=data)
        plt.title('Click Distribution')
        logger.info("Successfully created click distribution plot")
    except Exception as e:
        logger.error(f"Failed to create click distribution plot: {str(e)}")
        raise

    try:
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        os.makedirs(static_dir, exist_ok=True)
        image_path = os.path.join(static_dir, 'click_distribution.png')
        plt.savefig(image_path)
        logger.info(f"Successfully saved click distribution plot to {image_path}")
        app.static_folder = static_dir

    except Exception as e:
        logger.error(f"Failed to save click distribution plot: {str(e)}")
    finally:
        plt.close()


    
    try:
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance')
        logger.info("Successfully created feature importance plot")
    except Exception as e:
        logger.error(f"Failed to create feature importance plot: {str(e)}")
        raise

    try:
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        os.makedirs(static_dir, exist_ok=True)
        image_path = os.path.join(static_dir, 'feature_importance.png')
        plt.savefig(image_path)
        logger.info(f"Successfully saved feature importance plot to {image_path}")
        app.static_folder = static_dir

    except Exception as e:
        logger.error(f"Failed to save feature importance plot: {str(e)}")
    finally:
        plt.close()


    
    return {
        'accuracy': round(accuracy, 2),
        'roc_auc': round(roc_auc, 2),
        'data_preview': data.head(20).to_html(classes='table table-striped')

    }

@app.route('/')
def home():
    try:
        data = load_and_process_data()
        return render_template('index.html', 
                            accuracy=data['accuracy'],
                            roc_auc=data['roc_auc'],
                            data_preview=data['data_preview'])
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/api/log', methods=['GET'])
def log_request():
    """Log incoming requests and return a welcome message"""
    logger.info(f"Request received - Method: {request.method}, Path: {request.path}")
    return jsonify({
        'message': 'Welcome to the API!',
        'status': 'success',
        'request': {
            'method': request.method,
            'path': request.path
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
