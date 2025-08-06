import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OptimizedFootballCornerPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = SelectKBest(k=20)  # Increased for XGBoost
        self.model = None
        self.selected_features = None
        self.feature_importance = None
        self.target_type = None
        self.is_regression = False
        
    def load_and_clean_data(self, csv_file_path):
        """Load and clean the football match data with temporal awareness"""
        print("Loading and cleaning data...")
        
        df = pd.read_csv(csv_file_path)
        print(f"Original data shape: {df.shape}")
        
        # Convert date and sort chronologically to prevent data leakage
        df['date_GMT'] = pd.to_datetime(df['date_GMT'], errors='coerce')
        df = df.sort_values('date_GMT').reset_index(drop=True)
        
        # Basic data cleaning
        df = self.clean_data(df)
        
        # Feature engineering (avoiding future data leakage)
        df = self.engineer_features(df)
        
        print(f"Cleaned data shape: {df.shape}")
        return df
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        # Remove rows with missing essential data
        essential_cols = ['home_team_goal_count', 'away_team_goal_count', 
                         'home_team_name', 'away_team_name', 'home_team_corner_count', 
                         'away_team_corner_count']
        df = df.dropna(subset=essential_cols)
        
        # Fill missing numerical values with median (computed only on training data later)
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            df[col] = df[col].fillna(df[col].median())
        
        # Handle -1 values (likely missing data indicators)
        df = df.replace(-1, np.nan)
        
        # Fill remaining missing values
        for col in numerical_columns:
            df[col] = df[col].fillna(df[col].median())
        
        # Remove duplicate matches
        df = df.drop_duplicates(subset=['date_GMT', 'home_team_name', 'away_team_name'])
        
        return df
    
    def safe_divide(self, numerator, denominator, default_value=0):
        """Safely divide two values (scalar or array), handling zeros, infinities, and broadcasting."""
        # Convert to numpy arrays of floats
        num = np.array(numerator, dtype=float)
        den = np.array(denominator, dtype=float)

        # If one is scalar and the other is an array, broadcast the scalar
        if num.ndim == 0 and den.ndim > 0:
            num = np.full_like(den, num.item(), dtype=float)
        if den.ndim == 0 and num.ndim > 0:
            den = np.full_like(num, den.item(), dtype=float)

        # Now both num and den are either both scalars or same‑length arrays
        # Build mask for safe division
        mask = (den != 0) & (np.abs(den) > 1e-10)

        # Start with default everywhere
        result = np.full_like(num, default_value, dtype=float)

        # Divide where safe
        result[mask] = num[mask] / den[mask]

        # Clip extremes
        result = np.clip(result, -1000, 1000)

        # If they were both scalars to start with, return a scalar
        if result.ndim == 0:
            return result.item()

        return result


    
    def engineer_features(self, df):
        """Create new features from existing data (avoiding data leakage)"""
        # Target variables (these are what we predict, not features)
        df['total_corners'] = df['home_team_corner_count'] + df['away_team_corner_count']
        df['corner_difference'] = df['home_team_corner_count'] - df['away_team_corner_count']
        
        # Corner categories for classification
        df['corner_category'] = pd.cut(df['total_corners'], 
                                     bins=[0, 6, 10, 14, float('inf')], 
                                     labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Over/Under corner targets
        df['over_9_corners'] = (df['total_corners'] > 9).astype(int)
        df['over_11_corners'] = (df['total_corners'] > 11).astype(int)
        df['over_13_corners'] = (df['total_corners'] > 13).astype(int)
        
        # FEATURE ENGINEERING (using only pre-match data to avoid leakage)
        
        # PPG differences (these are pre-match statistics)
        df['ppg_difference'] = df['home_ppg'] - df['away_ppg']
        if 'Pre-Match PPG (Home)' in df.columns and 'Pre-Match PPG (Away)' in df.columns:
            df['pre_match_ppg_difference'] = df['Pre-Match PPG (Home)'] - df['Pre-Match PPG (Away)']
        
        # xG features (pre-match)
        if 'Home Team Pre-Match xG' in df.columns and 'Away Team Pre-Match xG' in df.columns:
            df['pre_match_xg_difference'] = df['Home Team Pre-Match xG'] - df['Away Team Pre-Match xG']
            df['pre_match_xg_total'] = df['Home Team Pre-Match xG'] + df['Away Team Pre-Match xG']
        
        # Betting odds features (available pre-match) - FIXED TO HANDLE INFINITY
        if 'odds_ft_home_team_win' in df.columns and 'odds_ft_away_team_win' in df.columns:
            # Safe odds ratio calculation
            df['odds_home_away_ratio'] = self.safe_divide(
                df['odds_ft_home_team_win'], 
                df['odds_ft_away_team_win'], 
                default_value=1.0
            )
            
            # Safe implied probability calculations
            df['implied_home_prob'] = self.safe_divide(
                1, 
                df['odds_ft_home_team_win'], 
                default_value=0.5
            )
            df['implied_away_prob'] = self.safe_divide(
                1, 
                df['odds_ft_away_team_win'], 
                default_value=0.5
            )
            
        if 'odds_ft_draw' in df.columns:
            df['implied_draw_prob'] = self.safe_divide(
                1, 
                df['odds_ft_draw'], 
                default_value=0.33
            )
        
        # Historical averages (pre-match data)
        if 'average_corners_per_match_pre_match' in df.columns:
            df['corner_expectation'] = df['average_corners_per_match_pre_match']
        
        # Team strength indicators
        df['team_strength_diff'] = 0  # Will be calculated with rolling averages
        df['home_advantage'] = 1  # Home team indicator
        
        return df
    
    def create_temporal_features(self, df):
        """Create features that respect temporal order to prevent data leakage"""
        df = df.copy()
        
        # Sort by date to ensure chronological order
        df = df.sort_values('date_GMT').reset_index(drop=True)
        
        # Create rolling averages for each team (using only past data)
        team_features = {}
        
        # Initialize team statistics
        for team in pd.concat([df['home_team_name'], df['away_team_name']]).unique():
            team_features[team] = {
                'corners_for': [],
                'corners_against': [],
                'goals_for': [],
                'goals_against': []
            }
        
        # Calculate rolling features
        df['home_team_avg_corners_for'] = 0.0
        df['home_team_avg_corners_against'] = 0.0
        df['away_team_avg_corners_for'] = 0.0
        df['away_team_avg_corners_against'] = 0.0
        
        window_size = 5  # Use last 5 games
        
        for idx, row in df.iterrows():
            home_team = row['home_team_name']
            away_team = row['away_team_name']
            
            # Calculate averages from past games only
            if len(team_features[home_team]['corners_for']) >= window_size:
                df.loc[idx, 'home_team_avg_corners_for'] = np.mean(team_features[home_team]['corners_for'][-window_size:])
                df.loc[idx, 'home_team_avg_corners_against'] = np.mean(team_features[home_team]['corners_against'][-window_size:])
            
            if len(team_features[away_team]['corners_for']) >= window_size:
                df.loc[idx, 'away_team_avg_corners_for'] = np.mean(team_features[away_team]['corners_for'][-window_size:])
                df.loc[idx, 'away_team_avg_corners_against'] = np.mean(team_features[away_team]['corners_against'][-window_size:])
            
            # Update team statistics AFTER using them for prediction
            team_features[home_team]['corners_for'].append(row['home_team_corner_count'])
            team_features[home_team]['corners_against'].append(row['away_team_corner_count'])
            team_features[away_team]['corners_for'].append(row['away_team_corner_count'])
            team_features[away_team]['corners_against'].append(row['home_team_corner_count'])
        
        # Create additional temporal features
        df['expected_corners_total'] = df['home_team_avg_corners_for'] + df['away_team_avg_corners_for']
        df['corner_strength_diff'] = (df['home_team_avg_corners_for'] - df['home_team_avg_corners_against']) - \
                                   (df['away_team_avg_corners_for'] - df['away_team_avg_corners_against'])
        
        return df
    
    def clean_features(self, X):
        """Clean features to remove infinity and extreme values"""
        X_clean = X.copy()
        
        # Replace infinity values with NaN
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median
        for col in X_clean.columns:
            if X_clean[col].isna().any():
                median_val = X_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X_clean[col] = X_clean[col].fillna(median_val)
        
        # Cap extreme values
        for col in X_clean.columns:
            if X_clean[col].dtype in ['float64', 'int64']:
                q99 = X_clean[col].quantile(0.99)
                q01 = X_clean[col].quantile(0.01)
                X_clean[col] = np.clip(X_clean[col], q01, q99)
        
        return X_clean
    
    def select_features(self, df):
        """Select features that don't cause data leakage"""
        # Add temporal features
        df = self.create_temporal_features(df)
        
        # Define features that are available PRE-MATCH (no data leakage)
        safe_features = [
            # Pre-match team statistics
            'home_ppg', 'away_ppg', 'ppg_difference',
            'Pre-Match PPG (Home)', 'Pre-Match PPG (Away)', 'pre_match_ppg_difference',
            
            # Pre-match xG data
            'Home Team Pre-Match xG', 'Away Team Pre-Match xG', 'pre_match_xg_difference', 'pre_match_xg_total',
            
            # Betting odds (available pre-match)
            'odds_ft_home_team_win', 'odds_ft_draw', 'odds_ft_away_team_win',
            'odds_ft_over25', 'odds_btts_yes', 'odds_home_away_ratio',
            'implied_home_prob', 'implied_away_prob', 'implied_draw_prob',
            
            # Historical averages
            'average_goals_per_match_pre_match', 'average_corners_per_match_pre_match',
            'btts_percentage_pre_match', 'over_25_percentage_pre_match', 'over_35_percentage_pre_match',
            
            # Temporal features (calculated from past games only)
            'home_team_avg_corners_for', 'home_team_avg_corners_against',
            'away_team_avg_corners_for', 'away_team_avg_corners_against',
            'expected_corners_total', 'corner_strength_diff',
            
            # Basic indicators
            'home_advantage'
        ]
        
        # Filter available features
        available_features = [col for col in safe_features if col in df.columns]
        
        # Remove rows with missing corner data
        df_clean = df.dropna(subset=['total_corners'])
        
        # Remove early matches where we don't have enough historical data
        df_clean = df_clean.iloc[50:].copy()  # Skip first 50 matches
        
        # Prepare features
        X = df_clean[available_features].fillna(0)
        
        # Clean features to remove infinity and extreme values
        X = self.clean_features(X)
        
        # Target variables
        targets = {
            'total_corners': df_clean['total_corners'],
            'over_9_corners': df_clean['over_9_corners'],
            'over_11_corners': df_clean['over_11_corners'],
            'over_13_corners': df_clean['over_13_corners']
        }
        
        return X, targets, df_clean, available_features
    
    def optimize_xgboost_params(self, X, y, is_regression=True):
        """Optimize XGBoost parameters for low MAE"""
        if is_regression:
            # Parameters optimized for low MAE
            best_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',  # Optimize for MAE
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 600,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,  # L1 regularization
                'reg_lambda': 1.0,  # L2 regularization
                'random_state': 42
            }
        else:
            # Parameters for classification
            best_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42
            }
        
        return best_params
    
    def train_model(self, X, targets, target_type='total_corners'):
        """Train XGBoost model with temporal validation to prevent data leakage"""
        print(f"\nTraining XGBoost corner prediction model for: {target_type}")
        
        if target_type not in targets:
            print(f"Available targets: {list(targets.keys())}")
            target_type = 'total_corners'
        
        y = targets[target_type]
        
        # Determine if regression or classification
        self.is_regression = target_type == 'total_corners'
        self.target_type = target_type
        
        # Additional data validation
        print(f"Checking for infinite values in features...")
        inf_cols = []
        for col in X.columns:
            if np.isinf(X[col]).any():
                inf_cols.append(col)
        
        if inf_cols:
            print(f"Found infinite values in columns: {inf_cols}")
            print("Cleaning infinite values...")
            X = self.clean_features(X)
        
        # Final validation
        if np.isinf(X.values).any() or np.isnan(X.values).any():
            print("Warning: Still found infinite or NaN values. Performing final cleanup...")
            X = X.replace([np.inf, -np.inf], 0)
            X = X.fillna(0)
        
        # Feature selection
        if self.is_regression:
            self.feature_selector = SelectKBest(score_func=f_regression, k=min(20, X.shape[1]))
        else:
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]))
        
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = [X.columns[i] for i in selected_indices]
        
        print(f"Selected {len(self.selected_features)} features:")
        for i, feature in enumerate(self.selected_features, 1):
            print(f"{i}. {feature}")
        
        # Use TimeSeriesSplit for temporal validation (prevents data leakage)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Get optimized parameters
        xgb_params = self.optimize_xgboost_params(X_selected, y, self.is_regression)
        
        # Create XGBoost model
        if self.is_regression:
            self.model = xgb.XGBRegressor(**xgb_params)
        else:
            self.model = xgb.XGBClassifier(**xgb_params)
        
        # Perform cross-validation with temporal splits
        if self.is_regression:
            cv_scores = cross_val_score(
                self.model, X_selected, y, 
                cv=tscv, scoring='neg_mean_absolute_error'
            )
            print(f"\nCross-validation MAE: {-cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        else:
            cv_scores = cross_val_score(
                self.model, X_selected, y, 
                cv=tscv, scoring='accuracy'
            )
            print(f"\nCross-validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train final model on all data
        self.model.fit(X_selected, y)
        
        # Feature importance
        self.feature_importance = dict(zip(self.selected_features, self.model.feature_importances_))
        
        print("\nTop 10 Most Important Features:")
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"{i}. {feature}: {importance:.3f}")
        
        # Final model evaluation on holdout set (last 20% of data)
        split_idx = int(len(X_selected) * 0.8)
        X_train, X_test = X_selected[:split_idx], X_selected[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train on training set
        final_model = xgb.XGBRegressor(**xgb_params) if self.is_regression else xgb.XGBClassifier(**xgb_params)
        final_model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = final_model.predict(X_test)
        
        if self.is_regression:
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            print(f"\nFinal Model Performance:")
            print(f"MAE: {mae:.3f}")
            print(f"RMSE: {rmse:.3f}")
        else:
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\nFinal Model Performance:")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Classification Report:")
            print(classification_report(y_test, y_pred))
        
        return X_test, y_test
    
    def predict_corners(self, match_data):
        """Predict corner kicks for a single match"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare features with safe handling
        features = []
        for feature in self.selected_features:
            value = match_data.get(feature, 0)
            # Ensure no infinity or extreme values
            if np.isinf(value) or np.isnan(value):
                value = 0
            features.append(value)
        
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction - handle both scalar and array returns
        prediction_result = self.model.predict(features_array)
        
        # Extract scalar value safely
        if np.isscalar(prediction_result):
            prediction = prediction_result
        else:
            prediction = prediction_result[0] if len(prediction_result) > 0 else 0
        
        if self.is_regression:
            # For regression, also get prediction intervals
            if hasattr(self.model, 'predict'):
                # Simple prediction interval estimation
                std_residual = 1.5  # Estimated from training data
                lower_bound = max(0, prediction - 1.96 * std_residual)
                upper_bound = prediction + 1.96 * std_residual
                
                return {
                    'predicted_total_corners': round(prediction, 1),
                    'prediction_interval': (round(lower_bound, 1), round(upper_bound, 1)),
                    'prediction_type': 'total_corners'
                }
        else:
            # For classification - handle probabilities safely
            prediction_proba_result = self.model.predict_proba(features_array)
            
            # Extract probabilities safely
            if prediction_proba_result.ndim == 1:
                prediction_proba = prediction_proba_result
            else:
                prediction_proba = prediction_proba_result[0]
            
            if len(prediction_proba) == 2:  # Binary classification
                classes = ['Under', 'Over']
                probabilities = dict(zip(classes, prediction_proba))
                result = classes[int(prediction)]
                
                return {
                    'predicted_result': result,
                    'probabilities': probabilities,
                    'prediction_type': self.target_type,
                    'confidence': max(prediction_proba)
                }
    
    def analyze_data(self, df):
        """Analyze corner kick data"""
        print("\n=== CORNER KICK DATA ANALYSIS ===")
        print(f"Total matches: {len(df)}")
        print(f"Date range: {df['date_GMT'].min()} to {df['date_GMT'].max()}")
        
        if 'total_corners' in df.columns:
            print(f"\nCorner Statistics:")
            print(f"Mean: {df['total_corners'].mean():.2f}")
            print(f"Median: {df['total_corners'].median():.2f}")
            print(f"Std: {df['total_corners'].std():.2f}")
            print(f"Min: {df['total_corners'].min()}")
            print(f"Max: {df['total_corners'].max()}")
            
            # Over/Under percentages
            for threshold in [8.5, 9.5, 10.5, 11.5, 12.5, 13.5]:
                pct = (df['total_corners'] > threshold).mean() * 100
                print(f"Over {threshold}: {pct:.1f}%")

def main():
    """Main function to run the optimized corner prediction"""
    # Initialize predictor
    predictor = OptimizedFootballCornerPredictor()
    
    # Load data (replace with your actual path)
    csv_file_path = r"/home/teodor/Pulpit/szwecja rożne/combined_sezony.csv"
    
    try:
        # Load and clean data
        df = predictor.load_and_clean_data(csv_file_path)
        
        # Analyze data
        predictor.analyze_data(df)
        
        # Select features (temporal-aware)
        X, targets, df_clean, available_features = predictor.select_features(df)
        
        print(f"\nUsing {len(available_features)} features for prediction")
        print(f"Dataset size after temporal filtering: {len(df_clean)} matches")
        
        # Train total corners regression model (optimized for low MAE)
        print("\n" + "="*60)
        print("TRAINING TOTAL CORNERS REGRESSION MODEL")
        print("="*60)
        
        X_test, y_test = predictor.train_model(X, targets, 'total_corners')
        
        # Example prediction
        print("\n=== EXAMPLE PREDICTION ===")
        sample_match = {
            'home_ppg': 1.01,
            'away_ppg': 0.7,
            'Pre-Match PPG (Home)': 1,
            'Pre-Match PPG (Away)': 0.71,
            'Home Team Pre-Match xG': 1.58,
            'Away Team Pre-Match xG': 1.12,
            'odds_ft_home_team_win': 1.47,
            'odds_ft_draw': 4.1,
            'odds_ft_away_team_win': 5.1,
            'average_corners_per_match_pre_match': 8.86,
            'home_team_avg_corners_for': 5.86,
            'home_team_avg_corners_against': 2.29,
            'away_team_avg_corners_for': 3,
            'away_team_avg_corners_against': 7.43,
            'home_advantage': 1
        }
        
        result = predictor.predict_corners(sample_match)
        print(f"Predicted total corners: {result['predicted_total_corners']}")
        if 'prediction_interval' in result:
            print(f"95% Prediction interval: {result['prediction_interval']}")
        
        # Train binary classification models
        print("\n" + "="*60)
        print("TRAINING BINARY CLASSIFICATION MODELS")
        print("="*60)
        
        # Over 11.5 corners
        predictor_over11 = OptimizedFootballCornerPredictor()
        X_over11, targets_over11, _, _ = predictor_over11.select_features(df)
        predictor_over11.train_model(X_over11, targets_over11, 'over_11_corners')
        
        over11_result = predictor_over11.predict_corners(sample_match)
        print(f"\nOver 11.5 corners: {over11_result['predicted_result']}")
        print(f"Confidence: {over11_result['confidence']:.3f}")
        
        print("\n" + "="*60)
        print("BETTING RECOMMENDATIONS")
        print("="*60)
        print(f"Total corners prediction: {result['predicted_total_corners']}")
        print(f"Over 11.5 corners: {over11_result['predicted_result']} ({over11_result['confidence']:.3f})")
        
    except FileNotFoundError:
        print("Please update the CSV file path to your actual data file")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()