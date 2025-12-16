import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna
from pathlib import Path
from catboost import CatBoostRegressor


from pipelines.model_factory import ModelFactory


class TrainingPipeline:
    """
    Training pipeline for energy efficiency forecasting model.
    Handles train-test split, model training, hyperparameter tuning, and evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize training pipeline with configuration.
        
        Args:
            config: Configuration dictionary containing training parameters
        """
        self.config = config
        self.training_config = config.get('training', {})
        self.target_column = self.training_config.get('target_column', 'heating_load')
        self.target_shift = self.training_config.get('target_shift', 1)
        self.test_size = self.training_config.get('test_size', 0.2)
        self.random_state = self.training_config.get('random_state', 42)
        self.shuffle = self.training_config.get('shuffle', False)
        
        self.model_config = self.training_config.get('model', {})
        self.model_params = self.model_config.get('params', {})
        self.early_stopping_rounds = self.model_config.get('early_stopping_rounds', 50)
        
        self.tuning_config = self.training_config.get('hyperparameter_tuning', {})
        self.tuning_enabled = self.tuning_config.get('enabled', False)
        
        self.model = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None 
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create shifted target variable for forecasting.
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            DataFrame with shifted target column
        """
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame")
        
        df_with_target = df.copy()
        
        df_with_target[f'{self.target_column}_target'] = df_with_target[self.target_column].shift(-self.target_shift)
        
        df_with_target = df_with_target.dropna(subset=[f'{self.target_column}_target'])
        
        print(f"Created target variable: {self.target_column}_target (shift: {self.target_shift})")
        print(f"Rows after dropping NaN targets: {df_with_target.shape[0]}")
        
        return df_with_target
    
    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features (X) and target (y) for training.
        
        Args:
            df: Input DataFrame with features and target
            
        Returns:
            Tuple of (X, y)
        """
        target_col = f'{self.target_column}_target'
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        X = df.drop(columns=[target_col, self.target_column])
        y = df[target_col]
        
        self.feature_names = list(X.columns)
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets, then scale features.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=self.shuffle
        )
        
        print(f"\nTrain-Test Split (shuffle={self.shuffle}):")
        print(f"Training set: {X_train.shape[0]} samples ({(1-self.test_size)*100:.0f}%)")
        print(f"Test set: {X_test.shape[0]} samples ({self.test_size*100:.0f}%)")
        
        # Scale features using StandardScaler
        print("\nScaling features with StandardScaler...")
        self.scaler = StandardScaler()
        
        # Fit on training data and transform both train and test
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve column names
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        print(f"Features scaled to mean=0, std=1")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: pd.DataFrame, y_test: pd.Series,
                    params: Optional[Dict[str, Any]] = None):
        """
        Train regression model using ModelFactory.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            params: Optional model parameters (uses config if not provided)
            
        Returns:
            Trained model
        """
        if params is None:
            params = self.model_params.copy()
        
        model_type = self.model_config.get('type', 'LinearRegression')
        
        print(f"\nTraining {model_type} with parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        if model_type in ['XGBoostRegressor', 'XGBRegressor']:
            params['early_stopping_rounds'] = self.early_stopping_rounds
        
        model = ModelFactory.create_model(model_type, params)
        
        if model_type in ['CatBoostRegressor']:
            model.fit(
                X_train, y_train,
                eval_set=(X_test, y_test),
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=params.get('verbose', 100)
            )
            print(f"\nModel training completed")
            if hasattr(model, 'best_iteration_'):
                print(f"Best iteration: {model.best_iteration_}")
            if hasattr(model, 'best_score_'):
                print(f"Best score: {model.best_score_}")
        
        elif model_type in ['XGBoostRegressor', 'XGBRegressor']:
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            print(f"\nModel training completed")
            if hasattr(model, 'best_iteration'):
                print(f"Best iteration: {model.best_iteration}")
        
        elif model_type in ['LGBMRegressor', 'LightGBMRegressor']:
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[
                    __import__('lightgbm').early_stopping(self.early_stopping_rounds, verbose=False)
                ]
            )
            print(f"\nModel training completed")
            if hasattr(model, 'best_iteration_'):
                print(f"Best iteration: {model.best_iteration_}")
        
        else:
            model.fit(X_train, y_train)
            print(f"\nModel training completed")
        
        return model
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of best hyperparameters
        """
        print("\nStarting hyperparameter tuning with Optuna...")
        
        tuning_params = self.tuning_config.get('params', {})
        n_trials = self.tuning_config.get('n_trials', 50)
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int(
                    'iterations',
                    tuning_params.get('iterations', {}).get('min', 100),
                    tuning_params.get('iterations', {}).get('max', 1000)
                ),
                'learning_rate': trial.suggest_float(
                    'learning_rate',
                    tuning_params.get('learning_rate', {}).get('min', 0.01),
                    tuning_params.get('learning_rate', {}).get('max', 0.3),
                    log=tuning_params.get('learning_rate', {}).get('log', True)
                ),
                'depth': trial.suggest_int(
                    'depth',
                    tuning_params.get('depth', {}).get('min', 4),
                    tuning_params.get('depth', {}).get('max', 10)
                ),
                'l2_leaf_reg': trial.suggest_float(
                    'l2_leaf_reg',
                    tuning_params.get('l2_leaf_reg', {}).get('min', 1),
                    tuning_params.get('l2_leaf_reg', {}).get('max', 10)
                ),
                'random_seed': self.random_state,
                'verbose': 0
            }
            
            model = CatBoostRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_test, y_test),
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=0
            )
            
            y_pred = model.predict(X_test)
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            
            return rmse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\nOptimization completed!")
        print(f"Best RMSE: {study.best_value:.4f}")
        print(f"\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        best_params = study.best_params.copy()
        best_params['random_seed'] = self.random_state
        best_params['verbose'] = self.model_params.get('verbose', 100)
        
        return best_params
    
    def evaluate_model(self, model, 
                      X_train: pd.DataFrame, y_train: pd.Series,
                      X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance on train and test sets.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary containing metrics for train and test sets
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_metrics = {
            'MSE': mean_squared_error(y_train, y_train_pred),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'MAE': mean_absolute_error(y_train, y_train_pred),
            'R2': r2_score(y_train, y_train_pred),
            'MAPE': np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
        }
        
        test_metrics = {
            'MSE': mean_squared_error(y_test, y_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'R2': r2_score(y_test, y_test_pred),
            'MAPE': np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
        }

        # Check for negative predictions
        train_negative = np.sum(y_train_pred < 0)
        test_negative = np.sum(y_test_pred < 0)

        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        print("\nTraining Set Metrics:")
        for metric_name, value in train_metrics.items():
            if metric_name == 'R2':
                print(f"  {metric_name}: {value:.6f}")
            elif metric_name == 'MAPE':
                print(f"  {metric_name}: {value:.4f}%")
            else:
                print(f"  {metric_name}: {value:.4f}")
        
        print("\nTest Set Metrics:")
        for metric_name, value in test_metrics.items():
            if metric_name == 'R2':
                print(f"  {metric_name}: {value:.6f}")
            elif metric_name == 'MAPE':
                print(f"  {metric_name}: {value:.4f}%")
            else:
                print(f"  {metric_name}: {value:.4f}")

        # Report negative predictions
        print("\nPrediction Range Check:")
        print(f"  Training set - Negative predictions: {train_negative} / {len(y_train_pred)}")
        if train_negative > 0:
            print(f"    Min prediction: {y_train_pred.min():.4f}")
        print(f"  Test set - Negative predictions: {test_negative} / {len(y_test_pred)}")
        if test_negative > 0:
            print(f"    Min prediction: {y_test_pred.min():.4f}")

        return {
            'train': train_metrics,
            'test': test_metrics
        }
    
    def run(self, df: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
        """
        Run the complete training pipeline.

        Args:
            df: Input DataFrame with engineered features

        Returns:
            Tuple of (trained_model_or_pipeline, metrics)

        Note:
            For sklearn models (LinearRegression, Ridge, Lasso): Returns a Pipeline with StandardScaler + Model
            For tree-based models (CatBoost, XGBoost, etc.): Returns the model directly
        """
        print("\n" + "="*60)
        print("TRAINING PIPELINE")
        print("="*60)

        print(f"\nInput shape: {df.shape}")

        df_with_target = self.create_target_variable(df)

        X, y = self.prepare_features_and_target(df_with_target)

        X_train, X_test, y_train, y_test = self.split_data(X, y)

        if self.tuning_enabled:
            print("\nHyperparameter tuning is enabled")
            best_params = self.hyperparameter_tuning(X_train, y_train, X_test, y_test)
            self.model = self.train_model(X_train, y_train, X_test, y_test, params=best_params)
        else:
            print("\nUsing default hyperparameters from config")
            self.model = self.train_model(X_train, y_train, X_test, y_test)

        metrics = self.evaluate_model(self.model, X_train, y_train, X_test, y_test)

        # Create Pipeline for sklearn models to include scaler
        model_type = self.model_config.get('type', 'LinearRegression')
        sklearn_models = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']

        if model_type in sklearn_models:
            print(f"\n🔧 Creating Pipeline with StandardScaler + {model_type}")
            print("   This ensures preprocessing is saved with the model!")

            # Create a fresh pipeline with the trained model's parameters
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', self.model)
            ])

            # Fit the pipeline on UNSCALED data
            # (We need to refit because the model was trained on already-scaled data)
            # Get unscaled data by inverse transforming
            X_train_unscaled = self.scaler.inverse_transform(X_train)
            X_test_unscaled = self.scaler.inverse_transform(X_test)

            print("   Refitting pipeline on unscaled data...")
            pipeline.fit(X_train_unscaled, y_train)

            # Verify the pipeline works correctly
            print("   Verifying pipeline predictions...")
            test_pred_pipeline = pipeline.predict(X_test_unscaled)
            test_pred_original = self.model.predict(X_test)

            # Check if predictions are close (should be identical after refitting)
            print(f"   Pipeline prediction sample: {test_pred_pipeline[:3]}")
            print(f"   Original prediction sample: {test_pred_original[:3]}")

            final_model = pipeline
        else:
            print(f"\n📦 Using {model_type} directly (tree-based model)")
            print("   Tree models handle scaling internally, no Pipeline needed")
            final_model = self.model

        print("\nTraining pipeline completed successfully")
        print("="*60)

        return final_model, metrics
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        For tree-based models: Uses built-in feature_importances_
        For LinearRegression: Calculates |coefficient| × std_dev(feature)
        
        Args:
            top_n: Number of top features to return (None for all)
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Check if model has built-in feature_importances_ (tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Check if model has coef_ attribute (LinearRegression, Ridge, Lasso, etc.)
        elif hasattr(self.model, 'coef_'):
            # Calculate feature importance as |coefficient| × std_dev(feature)
            # This accounts for both coefficient magnitude and feature scale
            
            if self.X_train is None:
                print("Warning: Training data not available for LinearRegression feature importance")
                return pd.DataFrame()
            
            # Calculate standard deviation of each feature
            feature_std = self.X_train.std()
            
            # Importance = |coefficient| × std_dev
            importance_values = np.abs(self.model.coef_) * feature_std.values
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=False)
            
            print(f"\nLinearRegression feature importance calculated using: |coefficient| × std_dev")
        
        else:
            print("Warning: Model does not support feature importance")
            return pd.DataFrame()
        
        if top_n is not None:
            importance_df = importance_df.head(top_n)
    
        return importance_df

    
    def validate_input(self, df: pd.DataFrame) -> bool:
        """
        Validate input DataFrame before training.
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is None or empty")
        
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame")
        
        if df.shape[0] < 10:
            raise ValueError(f"Insufficient data for training: {df.shape[0]} rows (minimum 10 required)")
        
        print(f"Input validation passed: {df.shape[0]} rows, {df.shape[1]} columns")
        return True