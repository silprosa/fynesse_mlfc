from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

def train_location_classifier(X_train, y_train, model_type='logistic'):
    """
    Train a classifier to predict location based on geographic features.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        model_type (str): Type of classifier ('logistic' or 'random_forest')
        
    Returns:
        trained model
    """
    if model_type == 'logistic':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42, n_estimators=100)
    else:
        raise ValueError("Model type must be 'logistic' or 'random_forest'")
    
    model.fit(X_train, y_train)
    return model

def evaluate_classifier(model, X_test, y_test):
    """
    Evaluate classifier performance.
    
    Args:
        model: Trained classifier
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels
        
    Returns:
        dict: Evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }

# Example usage function
def create_sample_data():
    """
    Create sample data for demonstration.
    
    Returns:
        tuple: (X, y) feature matrix and labels
    """
    # Sample data for two cities
    X_nyeri = np.array([
        [5, 10, 3, 2, 1, 21],  # Many amenities, few tourist spots
        [4, 8, 2, 1, 0, 15],
        [6, 12, 4, 3, 2, 27]
    ])
    
    X_cambridge = np.array([
        [15, 25, 8, 12, 6, 66],  # Many amenities and tourist spots
        [12, 20, 6, 10, 5, 53],
        [18, 30, 10, 15, 8, 81]
    ])
    
    X = np.vstack([X_nyeri, X_cambridge])
    y = np.array(['Nyeri'] * 3 + ['Cambridge'] * 3)
    
    return X, y
