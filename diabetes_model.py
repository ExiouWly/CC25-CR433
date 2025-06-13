# diabetes_model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
    
    def load_and_prepare_data(self, data_path='diabetes.csv'):
        """
        Load dan prepare data Pima Indians Diabetes Dataset dari Kaggle
        Download dari: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
        """
        try:
            # Coba load dataset asli
            data = pd.read_csv(data_path)
            print(f"Dataset berhasil dimuat dari {data_path}")
            print(f"Shape: {data.shape}")
            
            # Cek kolom yang ada di dataset
            expected_columns = self.feature_names + ['Outcome']
            if not all(col in data.columns for col in expected_columns):
                print("Kolom dataset tidak sesuai. Kolom yang tersedia:", data.columns.tolist())
                # Jika nama kolom berbeda, sesuaikan mapping
                if len(data.columns) == 9:
                    # Asumsi urutan kolom standar dataset Pima
                    data.columns = expected_columns
                    print("Kolom telah disesuaikan dengan standar Pima Indians Diabetes")
            
        except FileNotFoundError:
            print(f"File {data_path} tidak ditemukan!")
            print("Silakan download dataset dari:")
            print("https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
            print("Atau letakkan file 'diabetes.csv' di direktori yang sama")
            print("\nMenggunakan synthetic data untuk demo...")
            data = self.create_synthetic_data()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Menggunakan synthetic data untuk demo...")
            data = self.create_synthetic_data()
        
        # Pisahkan features dan target
        X = data[self.feature_names]
        y = data['Outcome']
        
        # Data exploration
        print(f"\nData Info:")
        print(f"Total samples: {len(data)}")
        print(f"Features: {X.shape[1]}")
        print(f"Target distribution:")
        print(y.value_counts())
        print(f"Positive rate: {y.mean():.2%}")
        
        # Handle missing values (0 values yang tidak masuk akal dalam konteks medis)
        # Untuk beberapa fitur, nilai 0 tidak masuk akal secara medis
        cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        print(f"\nMenangani missing values (nilai 0 yang tidak masuk akal):")
        for col in cols_to_fix:
            if col in X.columns:
                zero_count = (X[col] == 0).sum()
                if zero_count > 0:
                    print(f"- {col}: {zero_count} nilai 0 diganti dengan median ({X[col].median():.2f})")
                    X[col] = X[col].replace(0, X[col].median())
        
        # Cek outliers
        print(f"\nStatistik deskriptif:")
        print(X.describe())
        
        return X, y
    
    def create_synthetic_data(self):
        """
        Membuat synthetic data berdasarkan statistik dataset Pima Indians Diabetes asli
        Hanya digunakan jika dataset asli tidak tersedia
        """
        np.random.seed(42)
        n_samples = 768  # Sama dengan dataset asli
        
        # Statistik berdasarkan dataset Pima Indians Diabetes yang asli
        data = {
            'Pregnancies': np.random.poisson(3.8, n_samples),
            'Glucose': np.clip(np.random.normal(120.9, 31.97, n_samples), 0, 200),
            'BloodPressure': np.clip(np.random.normal(69.1, 19.36, n_samples), 0, 122),
            'SkinThickness': np.clip(np.random.normal(20.5, 15.95, n_samples), 0, 99),
            'Insulin': np.clip(np.random.exponential(79.8, n_samples), 0, 846),
            'BMI': np.clip(np.random.normal(31.99, 7.88, n_samples), 0, 67.1),
            'DiabetesPedigreeFunction': np.clip(np.random.exponential(0.47, n_samples), 0.078, 2.42),
            'Age': np.clip(np.random.gamma(2, 15, n_samples), 21, 81)
        }
        
        # Pastikan nilai positif dan dalam range yang masuk akal
        for key in data:
            data[key] = np.abs(data[key])
        
        # Buat target berdasarkan faktor risiko yang realistis
        # Berdasarkan penelitian medis tentang faktor risiko diabetes
        risk_score = (
            (data['Glucose'] > 140) * 0.25 +           # Glukosa tinggi
            (data['BMI'] > 30) * 0.15 +                # Obesitas
            (data['Age'] > 35) * 0.15 +                # Usia
            (data['Pregnancies'] > 4) * 0.1 +          # Riwayat kehamilan
            (data['BloodPressure'] > 80) * 0.1 +       # Hipertensi
            (data['DiabetesPedigreeFunction'] > 0.5) * 0.1 +  # Riwayat keluarga
            (data['SkinThickness'] > 30) * 0.05 +      # Ketebalan kulit
            (data['Insulin'] > 150) * 0.1              # Resistensi insulin
        )
        
        # Tambahkan noise dan threshold untuk outcome
        outcome_prob = risk_score + np.random.normal(0, 0.1, n_samples)
        data['Outcome'] = (outcome_prob > 0.35).astype(int)
        
        # Pastikan distribusi outcome mendekati dataset asli (~34.9% positive)
        positive_rate = data['Outcome'].mean()
        target_rate = 0.349  # Rate dari dataset asli
        
        if abs(positive_rate - target_rate) > 0.05:
            # Adjust jika terlalu jauh dari target
            n_positive_needed = int(n_samples * target_rate)
            sorted_indices = np.argsort(outcome_prob)[::-1]
            data['Outcome'] = np.zeros(n_samples, dtype=int)
            data['Outcome'][sorted_indices[:n_positive_needed]] = 1
        
        df = pd.DataFrame(data)
        
        print("PERINGATAN: Menggunakan synthetic data!")
        print("Untuk hasil yang akurat, gunakan dataset asli dari Kaggle")
        print(f"Synthetic data - Positive rate: {df['Outcome'].mean():.3f}")
        
        return df
    
    def build_model(self, input_dim):
        """
        Membangun model neural network untuk prediksi diabetes
        """
        model = keras.Sequential([
            keras.layers.Dense(16, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X, y, test_size=0.2, epochs=150, batch_size=32, validation_split=0.2):
        """
        Melatih model dengan data yang diberikan
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Training positive rate: {y_train.mean():.3f}")
        print(f"Test positive rate: {y_test.mean():.3f}")
        
        # Normalize data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build model
        self.model = self.build_model(X_train_scaled.shape[1])
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
        
        # Train model
        print(f"\nTraining model...")
        history = self.model.fit(
            X_train_scaled, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate model
        print(f"\nEvaluating model...")
        y_pred_proba = self.model.predict(X_test_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n{'='*50}")
        print(f"MODEL PERFORMANCE")
        print(f"{'='*50}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
        
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
        
        # Feature importance (menggunakan permutation importance)
        try:
            from sklearn.inspection import permutation_importance
            perm_importance = permutation_importance(
                self.model, X_test_scaled, y_test, n_repeats=10, random_state=42
            )
            
            print(f"\nFeature Importance (Permutation):")
            for i, importance in enumerate(perm_importance.importances_mean):
                print(f"{self.feature_names[i]}: {importance:.4f}")
        except:
            print("Feature importance calculation skipped")
        
        return history, accuracy, auc
    
    def predict_risk(self, input_data):
        """
        Memprediksi risiko diabetes berdasarkan input data
        input_data: list atau array dengan 8 fitur
        """
        if self.model is None:
            raise ValueError("Model belum dilatih. Panggil train_model() terlebih dahulu.")
        
        # Convert to numpy array if needed
        if isinstance(input_data, list):
            input_data = np.array(input_data)
        
        # Reshape jika perlu
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        
        # Scale input
        input_scaled = self.scaler.transform(input_data)
        
        # Predict
        risk_probability = self.model.predict(input_scaled)[0][0]
        risk_category = "Tinggi" if risk_probability > 0.5 else "Rendah"
        
        return {
            'risk_probability': float(risk_probability),
            'risk_percentage': float(risk_probability * 100),
            'risk_category': risk_category
        }
    
    def save_model(self, model_path='diabetes_model.h5', scaler_path='scaler.pkl'):
        """
        Menyimpan model dan scaler
        """
        if self.model is not None:
            self.model.save(model_path)
            joblib.dump(self.scaler, scaler_path)
            print(f"Model disimpan ke {model_path}")
            print(f"Scaler disimpan ke {scaler_path}")
        else:
            print("Model belum dilatih!")
    
    def load_model(self, model_path='diabetes_model.h5', scaler_path='scaler.pkl'):
        """
        Memuat model dan scaler yang sudah disimpan
        """
        try:
            self.model = keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            print("Model dan scaler berhasil dimuat!")
        except Exception as e:
            print(f"Error memuat model: {e}")

# Script untuk training dan testing
if __name__ == "__main__":
    # Inisialisasi predictor
    predictor = DiabetesPredictor()
    
    # Load dan prepare data dari Kaggle
    print("="*60)
    print("SISTEM PREDIKSI RISIKO DIABETES")
    print("Dataset: Pima Indians Diabetes Database (Kaggle)")
    print("="*60)
    
    print("\nStep 1: Loading data...")
    print("Pastikan file 'diabetes.csv' ada di direktori ini")
    print("Download dari: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
    
    X, y = predictor.load_and_prepare_data('diabetes.csv')
    
    print(f"\nStep 2: Data overview...")
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target distribution:")
    print(f"  - No Diabetes (0): {(y==0).sum()} samples ({(y==0).mean():.1%})")
    print(f"  - Diabetes (1): {(y==1).sum()} samples ({(y==1).mean():.1%})")
    
    # Train model
    print(f"\nStep 3: Training model...")
    history, accuracy, auc = predictor.train_model(X, y, epochs=150)
    
    # Save model
    print(f"\nStep 4: Saving model...")
    predictor.save_model('diabetes_model.h5', 'scaler.pkl')
    
    # Test prediction dengan beberapa contoh data
    print(f"\nStep 5: Testing predictions...")
    
    print(f"\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("Model siap digunakan untuk prediksi")
    print("Jalankan 'python app.py' untuk memulai web server")
    print("="*60)
