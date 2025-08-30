
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_synthetic_dataset():
    """Create a synthetic music genre classification dataset"""
    print("Creating synthetic music genre dataset...")
    
    # Define genres
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # Number of samples per genre
    n_samples_per_genre = 100
    
    # Create feature names (simulating MFCC features)
    feature_names = []
    for stat in ['mean', 'std', 'max', 'min']:
        for i in range(13):  # 13 MFCC coefficients
            feature_names.append(f'mfcc_{i}_{stat}')
    
    # Create synthetic data with different distributions for each genre
    data = []
    labels = []
    
    for i, genre in enumerate(genres):
        # Each genre has a different distribution of features
        center = i * 0.5  # Different center for each genre
        scale = 1.0 + i * 0.1  # Different scale for each genre
        
        # Generate features for this genre
        genre_features = np.random.normal(center, scale, (n_samples_per_genre, len(feature_names)))
        
        data.append(genre_features)
        labels.extend([genre] * n_samples_per_genre)
    
    # Combine all data
    X = np.vstack(data)
    y = np.array(labels)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['genre'] = y
    
    # Save to CSV
    df.to_csv('synthetic_music_features.csv', index=False)
    print(f"Synthetic dataset created with {len(df)} samples across {len(genres)} genres")
    print("Saved to 'synthetic_music_features.csv'")
    
    return df

def create_sample_spectrograms():
    """Create sample spectrograms for visualization"""
    print("\nGenerating sample spectrograms...")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Simple sine wave (simulating a pure tone)
    sr = 22050
    t = np.linspace(0, 3, 3*sr)
    signal1 = np.sin(2*np.pi*440*t)  # A4 note
    D1 = librosa.amplitude_to_db(np.abs(librosa.stft(signal1)), ref=np.max)
    librosa.display.specshow(D1, sr=sr, x_axis='time', y_axis='log', ax=axes[0, 0])
    axes[0, 0].set_title('Pure Tone (440 Hz)')
    axes[0, 0].set_xlabel('')
    
    # 2. Chord (multiple frequencies)
    signal2 = (np.sin(2*np.pi*261.63*t) +  # C4
               np.sin(2*np.pi*329.63*t) +  # E4
               np.sin(2*np.pi*392.00*t))   # G4
    D2 = librosa.amplitude_to_db(np.abs(librosa.stft(signal2)), ref=np.max)
    librosa.display.specshow(D2, sr=sr, x_axis='time', y_axis='log', ax=axes[0, 1])
    axes[0, 1].set_title('Chord (C Major)')
    axes[0, 1].set_xlabel('')
    axes[0, 1].set_ylabel('')
    
    # 3. Frequency sweep
    signal3 = np.sin(2*np.pi*100*t*(1 + t/6))
    D3 = librosa.amplitude_to_db(np.abs(librosa.stft(signal3)), ref=np.max)
    librosa.display.specshow(D3, sr=sr, x_axis='time', y_axis='log', ax=axes[1, 0])
    axes[1, 0].set_title('Frequency Sweep')
    
    # 4. Noise with filter (simulating percussion)
    signal4 = np.random.normal(0, 1, len(t)) * np.exp(-t/2)
    D4 = librosa.amplitude_to_db(np.abs(librosa.stft(signal4)), ref=np.max)
    librosa.display.specshow(D4, sr=sr, x_axis='time', y_axis='log', ax=axes[1, 1])
    axes[1, 1].set_title('Filtered Noise (Percussion-like)')
    axes[1, 1].set_ylabel('')
    
    plt.tight_layout()
    plt.suptitle('Sample Spectrograms of Different Audio Types', fontsize=16, y=1.02)
    plt.show()
    
    return fig

def create_mlp_model(input_dim, num_classes):
    """Create MLP model for tabular data"""
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_names, importances, top_n=20):
    """Plot feature importance"""
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title(f"Top {top_n} Most Important Features")
    plt.bar(range(top_n), importances[indices[:top_n]])
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_class_distribution(y, class_names):
    """Plot distribution of classes"""
    unique, counts = np.unique(y, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, counts, color=plt.cm.Set3(np.arange(len(class_names))))
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Genre')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def compare_models(results):
    """Compare performance of different models"""
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    plt.figure(figsize=(10, 6))
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    bars = plt.bar(models, accuracies, color=colors)
    plt.title('Model Performance Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{accuracy:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def main():
    # Create sample spectrograms for visualization
    create_sample_spectrograms()
    
    # Create synthetic dataset
    df = create_synthetic_dataset()
    
    # Find label column
    label_col = 'genre'
    
    # Separate features and labels
    X = df.drop(columns=[label_col]).values
    y = df[label_col]
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    print(f"Encoded {len(class_names)} classes: {list(class_names)}")
    
    # Plot class distribution
    print("\nPlotting class distribution...")
    plot_class_distribution(y_encoded, class_names)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get feature names
    feature_names = df.drop(columns=[label_col]).columns.tolist()
    
    # Dictionary to store results
    results = {}
    
    # 1. Random Forest Classifier
    print("\nTraining Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_accuracy = rf.score(X_test_scaled, y_test)
    results['Random Forest'] = {'model': rf, 'accuracy': rf_accuracy}
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # 2. Support Vector Machine
    print("\nTraining Support Vector Machine...")
    svm = SVC(kernel='rbf', random_state=42, probability=True)
    svm.fit(X_train_scaled, y_train)
    svm_accuracy = svm.score(X_test_scaled, y_test)
    results['SVM'] = {'model': svm, 'accuracy': svm_accuracy}
    print(f"SVM Accuracy: {svm_accuracy:.4f}")
    
    # 3. MLP Classifier (from scikit-learn)
    print("\nTraining MLP Classifier (scikit-learn)...")
    mlp_sklearn = MLPClassifier(hidden_layer_sizes=(128, 64, 32), random_state=42, max_iter=300)
    mlp_sklearn.fit(X_train_scaled, y_train)
    mlp_sklearn_accuracy = mlp_sklearn.score(X_test_scaled, y_test)
    results['MLP (sklearn)'] = {'model': mlp_sklearn, 'accuracy': mlp_sklearn_accuracy}
    print(f"MLP (sklearn) Accuracy: {mlp_sklearn_accuracy:.4f}")
    
    # 4. MLP with Keras
    print("\nTraining MLP with Keras...")
    mlp_keras = create_mlp_model(X_train.shape[1], len(np.unique(y_encoded)))
    
    history = mlp_keras.fit(X_train_scaled, y_train,
                           epochs=50,
                           batch_size=32,
                           validation_split=0.2,
                           verbose=1)
    
    mlp_keras_loss, mlp_keras_accuracy = mlp_keras.evaluate(X_test_scaled, y_test, verbose=0)
    results['MLP (Keras)'] = {'model': mlp_keras, 'accuracy': mlp_keras_accuracy}
    print(f"MLP (Keras) Accuracy: {mlp_keras_accuracy:.4f}")
    
    # Plot training history for Keras model
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Compare all models
    print("\nComparing all models...")
    compare_models(results)
    
    # Get predictions from the best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")
    
    if hasattr(best_model, 'predict'):
        y_pred = best_model.predict(X_test_scaled)
        if hasattr(y_pred, 'shape') and len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = best_model.predict(X_test_scaled)
    
    # Plot confusion matrix
    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, class_names)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = np.mean(np.abs(best_model.coef_), axis=0)
    else:
        rf_for_importance = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_for_importance.fit(X_train_scaled, y_train)
        importances = rf_for_importance.feature_importances_
    
    plot_feature_importance(feature_names, importances)
    
    # Correlation heatmap of features
    print("\nPlotting feature correlation heatmap...")
    plt.figure(figsize=(12, 10))
    corr = df.drop(columns=[label_col]).corr()
    sns.heatmap(corr, cmap='coolwarm', center=0, square=True)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Genres: {', '.join(class_names)}")
    for model_name, result in results.items():
        print(f"{model_name}: {result['accuracy']:.4f} accuracy")
    print("="*50)

if __name__ == "__main__":
    main()