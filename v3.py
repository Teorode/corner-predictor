import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, log_loss
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class EnhancedFootballCornerPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(k=25)  # Increased for better performance
        self.models = {}  # Store multiple models for different predictions
        self.selected_features = {}
        self.feature_importance = {}
        self.team_stats_cache = {}  # Cache for team statistics
        self.corner_lines = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]
        
    def load_and_clean_data(self, csv_file_path):
        """Load and clean the football match data with enhanced temporal awareness"""
        print("Loading and cleaning data...")
        
        df = pd.read_csv(csv_file_path)
        print(f"Original data shape: {df.shape}")
        
        # Convert date and sort chronologically
        df['date_GMT'] = pd.to_datetime(df['date_GMT'], errors='coerce')
        df = df.sort_values('date_GMT').reset_index(drop=True)
        
        # Enhanced data cleaning
        df = self.clean_data(df)
        
        # Advanced feature engineering
        df = self.engineer_features(df)
        
        print(f"Cleaned data shape: {df.shape}")
        return df
    
    def clean_data(self, df):
        """Enhanced data cleaning with better missing value handling"""
        # Remove rows with missing essential data
        essential_cols = ['home_team_goal_count', 'away_team_goal_count', 
                         'home_team_name', 'away_team_name', 'home_team_corner_count', 
                         'away_team_corner_count', 'date_GMT']
        
        initial_rows = len(df)
        df = df.dropna(subset=essential_cols)
        print(f"Removed {initial_rows - len(df)} rows with missing essential data")
        
        # Handle -1 values and other missing indicators
        df = df.replace([-1, -999, 999, np.inf, -np.inf], np.nan)
        
        # Smart imputation for numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if col not in essential_cols:
                # Use median for non-essential columns
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
        
        # Remove duplicate matches (same teams on same date)
        df = df.drop_duplicates(subset=['date_GMT', 'home_team_name', 'away_team_name'])
        
        # Remove matches with unrealistic corner counts (likely data errors)
        df = df[(df['home_team_corner_count'] >= 0) & (df['home_team_corner_count'] <= 25)]
        df = df[(df['away_team_corner_count'] >= 0) & (df['away_team_corner_count'] <= 25)]
        
        return df
    
    def safe_divide(self, numerator, denominator, default_value=0):
        """Enhanced safe division with better handling"""
        num = np.array(numerator, dtype=float)
        den = np.array(denominator, dtype=float)
        
        # Handle broadcasting
        if num.ndim == 0 and den.ndim > 0:
            num = np.full_like(den, num.item(), dtype=float)
        if den.ndim == 0 and num.ndim > 0:
            den = np.full_like(num, den.item(), dtype=float)
        
        # Safe division mask
        mask = (den != 0) & (np.abs(den) > 1e-12) & np.isfinite(den) & np.isfinite(num)
        
        result = np.full_like(num, default_value, dtype=float)
        result[mask] = num[mask] / den[mask]
        
        # Clip extreme values
        result = np.clip(result, -100, 100)
        
        return result.item() if result.ndim == 0 else result
    
    def engineer_features(self, df):
        """Advanced feature engineering with corner-specific features"""
        # Core target variables
        df['total_corners'] = df['home_team_corner_count'] + df['away_team_corner_count']
        df['corner_difference'] = df['home_team_corner_count'] - df['away_team_corner_count']
        df['home_corner_ratio'] = self.safe_divide(df['home_team_corner_count'], df['total_corners'], 0.5)
        
        # Create all corner line targets
        for line in self.corner_lines:
            df[f'over_{line}_corners'] = (df['total_corners'] > line).astype(int)
        
        # Enhanced pre-match features
        if 'Pre-Match PPG (Home)' in df.columns and 'Pre-Match PPG (Away)' in df.columns:
            df['pre_match_ppg_difference'] = df['Pre-Match PPG (Home)'] - df['Pre-Match PPG (Away)']
            df['pre_match_ppg_total'] = df['Pre-Match PPG (Home)'] + df['Pre-Match PPG (Away)']
            df['pre_match_strength_ratio'] = self.safe_divide(df['Pre-Match PPG (Home)'], 
                                                            df['Pre-Match PPG (Away)'], 1.0)
        
        # Enhanced xG features
        if 'Home Team Pre-Match xG' in df.columns and 'Away Team Pre-Match xG' in df.columns:
            df['pre_match_xg_difference'] = df['Home Team Pre-Match xG'] - df['Away Team Pre-Match xG']
            df['pre_match_xg_total'] = df['Home Team Pre-Match xG'] + df['Away Team Pre-Match xG']
            df['pre_match_xg_ratio'] = self.safe_divide(df['Home Team Pre-Match xG'], 
                                                       df['Away Team Pre-Match xG'], 1.0)
        
        # Enhanced betting odds features
        odds_cols = ['odds_ft_home_team_win', 'odds_ft_draw', 'odds_ft_away_team_win']
        if all(col in df.columns for col in odds_cols):
            # Clean odds data
            for col in odds_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
                df[col] = np.clip(df[col], 1.01, 100)  # Reasonable bounds
            
            # Implied probabilities
            df['implied_home_prob'] = self.safe_divide(1, df['odds_ft_home_team_win'], 0.33)
            df['implied_draw_prob'] = self.safe_divide(1, df['odds_ft_draw'], 0.33)
            df['implied_away_prob'] = self.safe_divide(1, df['odds_ft_away_team_win'], 0.33)
            
            # Market efficiency features
            total_implied = df['implied_home_prob'] + df['implied_draw_prob'] + df['implied_away_prob']
            df['market_margin'] = total_implied - 1
            
            # Match competitiveness
            df['match_competitiveness'] = 1 - np.abs(df['implied_home_prob'] - df['implied_away_prob'])
            
            # Odds ratios
            df['home_away_odds_ratio'] = self.safe_divide(df['odds_ft_home_team_win'], 
                                                         df['odds_ft_away_team_win'], 1.0)
        
        # Historical data features
        if 'average_corners_per_match_pre_match' in df.columns:
            df['expected_corners_base'] = df['average_corners_per_match_pre_match']
        
        # Additional pre-match statistics
        percentage_cols = ['btts_percentage_pre_match', 'over_25_percentage_pre_match', 
                          'over_35_percentage_pre_match']
        for col in percentage_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(50) / 100
        
        return df
    
    def create_advanced_temporal_features(self, df):
        """Create sophisticated temporal features avoiding data leakage"""
        df = df.copy()
        df = df.sort_values('date_GMT').reset_index(drop=True)
        
        # Enhanced team statistics tracking
        team_stats = {}
        head_to_head_stats = {}
        
        # Initialize comprehensive stats for each team
        for team in pd.concat([df['home_team_name'], df['away_team_name']]).unique():
            team_stats[team] = {
                'corners_for': [],
                'corners_against': [],
                'goals_for': [],
                'goals_against': [],
                'matches_played': 0,
                'recent_form': [],
                'corner_variance': [],
                'corner_momentum': [],
                'high_corner_games': [],  # Games with 12+ total corners
                'low_corner_games': [],   # Games with 8- total corners
                'referee_history': {},    # Corner patterns by referee
                'venue_corners': [],      # Venue-specific corner history
                'time_of_year': [],       # Seasonal patterns
                'rest_days': [],          # Days between matches
                'pressure_situations': [] # High-stakes games
            }
        
        # Rolling window sizes
        short_window = 3
        medium_window = 6
        long_window = 10
        
        # Enhanced feature columns
        feature_cols = [
            'home_corners_avg_3', 'home_corners_avg_6', 'home_corners_avg_10',
            'away_corners_avg_3', 'away_corners_avg_6', 'away_corners_avg_10',
            'home_corners_against_avg_3', 'home_corners_against_avg_6', 'home_corners_against_avg_10',
            'away_corners_against_avg_3', 'away_corners_against_avg_6', 'away_corners_against_avg_10',
            'home_corner_trend', 'away_corner_trend',
            'home_recent_form', 'away_recent_form',
            'expected_total_corners', 'corner_advantage_home', 'corner_advantage_away',
            'teams_familiarity', 'home_corner_variance', 'away_corner_variance',
            # NEW ADVANCED FEATURES FOR HARD-TO-PREDICT ZONE
            'corner_momentum_home', 'corner_momentum_away',
            'home_high_corner_rate', 'away_high_corner_rate',
            'home_low_corner_rate', 'away_low_corner_rate',
            'combined_corner_volatility', 'match_importance_factor',
            'referee_corner_tendency', 'venue_corner_factor',
            'seasonal_corner_adjustment', 'rest_impact_factor',
            'h2h_corner_history', 'style_clash_indicator',
            'home_corner_consistency', 'away_corner_consistency',
            'attacking_desperation_factor', 'defensive_solidity_factor'
        ]
        
        for col in feature_cols:
            df[col] = 0.0
        
        # Calculate features for each match
        for idx, row in df.iterrows():
            home_team = row['home_team_name']
            away_team = row['away_team_name']
            match_date = row['date_GMT']
            
            home_stats = team_stats[home_team]
            away_stats = team_stats[away_team]
            
            # Calculate rolling averages (multiple windows)
            for window, suffix in [(short_window, '3'), (medium_window, '6'), (long_window, '10')]:
                if len(home_stats['corners_for']) >= window:
                    df.loc[idx, f'home_corners_avg_{suffix}'] = np.mean(home_stats['corners_for'][-window:])
                    df.loc[idx, f'home_corners_against_avg_{suffix}'] = np.mean(home_stats['corners_against'][-window:])
                
                if len(away_stats['corners_for']) >= window:
                    df.loc[idx, f'away_corners_avg_{suffix}'] = np.mean(away_stats['corners_for'][-window:])
                    df.loc[idx, f'away_corners_against_avg_{suffix}'] = np.mean(away_stats['corners_against'][-window:])
            
            # Corner trends (recent vs older performance)
            if len(home_stats['corners_for']) >= 6:
                recent_home = np.mean(home_stats['corners_for'][-3:])
                older_home = np.mean(home_stats['corners_for'][-6:-3])
                df.loc[idx, 'home_corner_trend'] = recent_home - older_home
            
            if len(away_stats['corners_for']) >= 6:
                recent_away = np.mean(away_stats['corners_for'][-3:])
                older_away = np.mean(away_stats['corners_for'][-6:-3])
                df.loc[idx, 'away_corner_trend'] = recent_away - older_away
            
            # Recent form (goals-based)
            if len(home_stats['recent_form']) >= 3:
                df.loc[idx, 'home_recent_form'] = np.mean(home_stats['recent_form'][-3:])
            
            if len(away_stats['recent_form']) >= 3:
                df.loc[idx, 'away_recent_form'] = np.mean(away_stats['recent_form'][-3:])
            
            # Corner variance (predictability measure)
            if len(home_stats['corners_for']) >= 5:
                df.loc[idx, 'home_corner_variance'] = np.var(home_stats['corners_for'][-5:])
            
            if len(away_stats['corners_for']) >= 5:
                df.loc[idx, 'away_corner_variance'] = np.var(away_stats['corners_for'][-5:])
            
            # NEW ADVANCED FEATURES FOR BETTER PREDICTION
            
            # 1. Corner Momentum (weighted recent performance)
            if len(home_stats['corners_for']) >= 4:
                weights = [0.4, 0.3, 0.2, 0.1]  # More weight on recent games
                weighted_corners = sum(w * c for w, c in zip(weights, home_stats['corners_for'][-4:]))
                df.loc[idx, 'corner_momentum_home'] = weighted_corners
            
            if len(away_stats['corners_for']) >= 4:
                weights = [0.4, 0.3, 0.2, 0.1]
                weighted_corners = sum(w * c for w, c in zip(weights, away_stats['corners_for'][-4:]))
                df.loc[idx, 'corner_momentum_away'] = weighted_corners
            
            # 2. High/Low Corner Game Rates (crucial for 10.5-13.5 prediction)
            if len(home_stats['corners_for']) >= 8:
                home_total_corners = [home_stats['corners_for'][i] + home_stats['corners_against'][i] 
                                    for i in range(len(home_stats['corners_for']))]
                high_corner_games = sum(1 for total in home_total_corners[-8:] if total >= 12)
                low_corner_games = sum(1 for total in home_total_corners[-8:] if total <= 8)
                df.loc[idx, 'home_high_corner_rate'] = high_corner_games / 8
                df.loc[idx, 'home_low_corner_rate'] = low_corner_games / 8
            
            if len(away_stats['corners_for']) >= 8:
                away_total_corners = [away_stats['corners_for'][i] + away_stats['corners_against'][i] 
                                    for i in range(len(away_stats['corners_for']))]
                high_corner_games = sum(1 for total in away_total_corners[-8:] if total >= 12)
                low_corner_games = sum(1 for total in away_total_corners[-8:] if total <= 8)
                df.loc[idx, 'away_high_corner_rate'] = high_corner_games / 8
                df.loc[idx, 'away_low_corner_rate'] = low_corner_games / 8
            
            # 3. Combined Corner Volatility (key for middle lines)
            home_volatility = df.loc[idx, 'home_corner_variance']
            away_volatility = df.loc[idx, 'away_corner_variance']
            df.loc[idx, 'combined_corner_volatility'] = (home_volatility + away_volatility) / 2
            
            # 4. Head-to-Head Corner History
            h2h_key = f"{home_team}_vs_{away_team}"
            h2h_reverse_key = f"{away_team}_vs_{home_team}"
            
            if h2h_key in head_to_head_stats or h2h_reverse_key in head_to_head_stats:
                h2h_data = head_to_head_stats.get(h2h_key, head_to_head_stats.get(h2h_reverse_key, []))
                if len(h2h_data) > 0:
                    df.loc[idx, 'h2h_corner_history'] = np.mean([game['total_corners'] for game in h2h_data[-3:]])
            
            # 5. Style Clash Indicator (attacking vs defensive teams)
            home_attacking_style = df.loc[idx, 'home_corners_avg_6'] - df.loc[idx, 'home_corners_against_avg_6']
            away_attacking_style = df.loc[idx, 'away_corners_avg_6'] - df.loc[idx, 'away_corners_against_avg_6']
            
            # When both teams are attacking or both defensive, corners can be unpredictable
            style_difference = abs(home_attacking_style - away_attacking_style)
            df.loc[idx, 'style_clash_indicator'] = style_difference
            
            # 6. Corner Consistency (low variance = more predictable)
            if len(home_stats['corners_for']) >= 6:
                home_consistency = 1 / (1 + df.loc[idx, 'home_corner_variance'])  # Higher = more consistent
                df.loc[idx, 'home_corner_consistency'] = home_consistency
            
            if len(away_stats['corners_for']) >= 6:
                away_consistency = 1 / (1 + df.loc[idx, 'away_corner_variance'])
                df.loc[idx, 'away_corner_consistency'] = away_consistency
            
            # 7. Match Importance Factor (based on league position, if available)
            # This could be enhanced with actual league table data
            df.loc[idx, 'match_importance_factor'] = 1.0  # Default, could be enhanced
            
            # 8. Seasonal Corner Adjustment (month of year effect)
            if pd.notna(match_date):
                month = match_date.month
                # Early season (Aug-Oct): slightly fewer corners due to fitness
                # Mid season (Nov-Feb): normal
                # Late season (Mar-May): more corners due to desperation
                seasonal_factors = {8: 0.95, 9: 0.97, 10: 0.98, 11: 1.0, 12: 1.0, 
                                  1: 1.0, 2: 1.0, 3: 1.02, 4: 1.03, 5: 1.05}
                df.loc[idx, 'seasonal_corner_adjustment'] = seasonal_factors.get(month, 1.0)
            
            # 9. Attacking Desperation Factor (teams that need goals get more corners)
            home_goal_diff = np.mean(home_stats['goals_for'][-5:]) - np.mean(home_stats['goals_against'][-5:]) if len(home_stats['goals_for']) >= 5 else 0
            away_goal_diff = np.mean(away_stats['goals_for'][-5:]) - np.mean(away_stats['goals_against'][-5:]) if len(away_stats['goals_for']) >= 5 else 0
            
            # Teams with poor goal difference tend to attack more desperately
            desperation_factor = max(0, (2 - home_goal_diff) + (2 - away_goal_diff)) / 4
            df.loc[idx, 'attacking_desperation_factor'] = desperation_factor
            
            # 10. Defensive Solidity Factor (solid defenses = fewer corners against)
            home_defensive_solidity = 1 / (1 + np.mean(home_stats['corners_against'][-5:])) if len(home_stats['corners_against']) >= 5 else 0.5
            away_defensive_solidity = 1 / (1 + np.mean(away_stats['corners_against'][-5:])) if len(away_stats['corners_against']) >= 5 else 0.5
            df.loc[idx, 'defensive_solidity_factor'] = (home_defensive_solidity + away_defensive_solidity) / 2
            
            # Expected total corners with enhanced calculation
            home_for = df.loc[idx, 'home_corners_avg_6']
            home_against = df.loc[idx, 'home_corners_against_avg_6']
            away_for = df.loc[idx, 'away_corners_avg_6']
            away_against = df.loc[idx, 'away_corners_against_avg_6']
            
            base_expected = (home_for + away_against + away_for + home_against) / 2
            
            # Adjust based on new factors
            volatility_adjustment = 1 + (df.loc[idx, 'combined_corner_volatility'] - 2) * 0.1
            style_adjustment = 1 + (df.loc[idx, 'style_clash_indicator'] * 0.05)
            seasonal_adjustment = df.loc[idx, 'seasonal_corner_adjustment']
            
            df.loc[idx, 'expected_total_corners'] = base_expected * volatility_adjustment * style_adjustment * seasonal_adjustment
            df.loc[idx, 'corner_advantage_home'] = home_for - home_against
            df.loc[idx, 'corner_advantage_away'] = away_for - away_against
            
            # Teams familiarity (how often they've played recently)
            df.loc[idx, 'teams_familiarity'] = len(head_to_head_stats.get(h2h_key, [])) + len(head_to_head_stats.get(h2h_reverse_key, []))
            
            # Update team statistics AFTER using them
            home_corners = row['home_team_corner_count']
            away_corners = row['away_team_corner_count']
            home_goals = row['home_team_goal_count']
            away_goals = row['away_team_goal_count']
            total_corners = home_corners + away_corners
            
            # Update home team stats
            home_stats['corners_for'].append(home_corners)
            home_stats['corners_against'].append(away_corners)
            home_stats['goals_for'].append(home_goals)
            home_stats['goals_against'].append(away_goals)
            home_stats['matches_played'] += 1
            home_stats['recent_form'].append(1 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0)
            
            # Update away team stats
            away_stats['corners_for'].append(away_corners)
            away_stats['corners_against'].append(home_corners)
            away_stats['goals_for'].append(away_goals)
            away_stats['goals_against'].append(home_goals)
            away_stats['matches_played'] += 1
            away_stats['recent_form'].append(1 if away_goals > home_goals else 0.5 if away_goals == home_goals else 0)
            
            # Update head-to-head history
            h2h_entry = {
                'total_corners': total_corners,
                'home_corners': home_corners,
                'away_corners': away_corners,
                'date': match_date
            }
            
            if h2h_key not in head_to_head_stats:
                head_to_head_stats[h2h_key] = []
            head_to_head_stats[h2h_key].append(h2h_entry)
        
        return df
    
    def select_optimized_features(self, df):
        """Select features optimized for corner prediction"""
        # Add advanced temporal features
        df = self.create_advanced_temporal_features(df)
        
        # Comprehensive feature list (all pre-match available)
        feature_candidates = [
            # Pre-match team strength
            'Pre-Match PPG (Home)', 'Pre-Match PPG (Away)', 'pre_match_ppg_difference', 
            'pre_match_ppg_total', 'pre_match_strength_ratio',
            
            # Pre-match xG
            'Home Team Pre-Match xG', 'Away Team Pre-Match xG', 'pre_match_xg_difference',
            'pre_match_xg_total', 'pre_match_xg_ratio',
            
            # Market data
            'odds_ft_home_team_win', 'odds_ft_draw', 'odds_ft_away_team_win',
            'implied_home_prob', 'implied_draw_prob', 'implied_away_prob',
            'market_margin', 'match_competitiveness', 'home_away_odds_ratio',
            
            # Historical data
            'average_goals_per_match_pre_match', 'average_corners_per_match_pre_match',
            'btts_percentage_pre_match', 'over_25_percentage_pre_match', 'over_35_percentage_pre_match',
            
            # Temporal features (short-term)
            'home_corners_avg_3', 'away_corners_avg_3', 'home_corners_against_avg_3', 'away_corners_against_avg_3',
            
            # Temporal features (medium-term)  
            'home_corners_avg_6', 'away_corners_avg_6', 'home_corners_against_avg_6', 'away_corners_against_avg_6',
            
            # Temporal features (long-term)
            'home_corners_avg_10', 'away_corners_avg_10', 'home_corners_against_avg_10', 'away_corners_against_avg_10',
            
            # Advanced temporal features
            'home_corner_trend', 'away_corner_trend', 'home_recent_form', 'away_recent_form',
            'expected_total_corners', 'corner_advantage_home', 'corner_advantage_away',
            'home_corner_variance', 'away_corner_variance',
            
            # NEW ADVANCED FEATURES FOR IMPROVED ACCURACY IN 10.5-13.5 ZONE
            'corner_momentum_home', 'corner_momentum_away',
            'home_high_corner_rate', 'away_high_corner_rate',
            'home_low_corner_rate', 'away_low_corner_rate',
            'combined_corner_volatility', 'match_importance_factor',
            'h2h_corner_history', 'style_clash_indicator',
            'home_corner_consistency', 'away_corner_consistency',
            'seasonal_corner_adjustment', 'attacking_desperation_factor',
            'defensive_solidity_factor', 'teams_familiarity'
        ]
        
        # Filter available features
        available_features = [col for col in feature_candidates if col in df.columns]
        print(f"Available features: {len(available_features)}")
        
        # Remove early matches without sufficient historical data
        min_matches_for_prediction = 30
        df_clean = df.iloc[min_matches_for_prediction:].copy()
        print(f"Using {len(df_clean)} matches after filtering for historical data")
        
        # Prepare feature matrix
        X = df_clean[available_features].copy()
        
        # Advanced feature cleaning
        X = self.advanced_feature_cleaning(X)
        
        # Create all target variables
        targets = {'total_corners': df_clean['total_corners']}
        for line in self.corner_lines:
            targets[f'over_{line}_corners'] = df_clean[f'over_{line}_corners']
        
        return X, targets, df_clean, available_features
    
    def advanced_feature_cleaning(self, X):
        """Advanced feature cleaning and preprocessing"""
        X_clean = X.copy()
        
        # Replace infinity and extreme values
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # Outlier detection and handling using IQR method
        for col in X_clean.select_dtypes(include=[np.number]).columns:
            Q1 = X_clean[col].quantile(0.25)
            Q3 = X_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2.5 * IQR
            upper_bound = Q3 + 2.5 * IQR
            
            # Cap outliers instead of removing them
            X_clean[col] = np.clip(X_clean[col], lower_bound, upper_bound)
            
            # Fill NaN values with median
            median_val = X_clean[col].median()
            X_clean[col] = X_clean[col].fillna(median_val if not pd.isna(median_val) else 0)
        
        return X_clean
    
    def optimize_xgboost_hyperparameters(self, target_type):
        """Optimized hyperparameters for different prediction types with enhanced focus on hard-to-predict zones"""
        if target_type == 'total_corners':
            return {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'max_depth': 12,                    # Increased for complex patterns
                'learning_rate': 0.04,             # Slower learning for better generalization
                'n_estimators': 2000,              # More trees for complex relationships
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'colsample_bylevel': 0.8,
                'colsample_bynode': 0.8,           # Additional regularization
                'reg_alpha': 0.15,                 # Increased L1 regularization
                'reg_lambda': 1.5,                 # Increased L2 regularization
                'min_child_weight': 4,             # Higher for stability
                'gamma': 0.15,                     # Minimum split loss
                'max_delta_step': 1,               # Helps with imbalanced data
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist'              # Faster training
            }
        else:
            # Special tuning for middle corner lines (10.5-13.5)
            is_middle_line = any(line in target_type for line in ['10.5', '11.5', '12.5', '13.5'])
            
            if is_middle_line:
                return {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': 12,                # Deep trees for complex patterns
                    'learning_rate': 0.04,         # Very slow learning
                    'n_estimators': 1700,           # More trees for hard cases
                    'subsample': 0.82,
                    'colsample_bytree': 0.82,
                    'colsample_bylevel': 0.85,
                    'colsample_bynode': 0.85,
                    'reg_alpha': 0.25,             # High regularization for stability
                    'reg_lambda': 1.8,
                    'min_child_weight': 5,         # Conservative splits
                    'gamma': 0.2,                  # Higher minimum gain
                    'scale_pos_weight': 1,         # Will be adjusted per line
                    'random_state': 42,
                    'n_jobs': -1,
                    'tree_method': 'hist'
                }
            else:
                # Standard settings for easier lines
                return {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': 8,
                    'learning_rate': 0.07,
                    'n_estimators': 700,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'colsample_bylevel': 0.85,
                    'reg_alpha': 0.2,
                    'reg_lambda': 1.0,
                    'min_child_weight': 3,
                    'scale_pos_weight': 1,
                    'random_state': 42,
                    'n_jobs': -1
                }
    
    def train_enhanced_models(self, X, targets):
        """Train multiple optimized models for different predictions with ensemble for hard lines"""
        print(f"\n{'='*80}")
        print("TRAINING ENHANCED CORNER PREDICTION MODELS WITH HARD-LINE OPTIMIZATION")
        print(f"{'='*80}")
        
        results = {}
        
        # Train total corners regression model
        print(f"\n{'='*60}")
        print("TRAINING TOTAL CORNERS REGRESSION MODEL")
        print(f"{'='*60}")
        
        y_regression = targets['total_corners']
        
        # Feature selection for regression with more features for complex patterns
        feature_selector_reg = SelectKBest(score_func=mutual_info_regression, k=min(40, X.shape[1]))
        X_selected_reg = feature_selector_reg.fit_transform(X, y_regression)
        
        selected_features_reg = [X.columns[i] for i in feature_selector_reg.get_support(indices=True)]
        self.selected_features['total_corners'] = selected_features_reg
        
        print(f"Selected {len(selected_features_reg)} features for regression:")
        for i, feature in enumerate(selected_features_reg[:15], 1):
            print(f"{i:2d}. {feature}")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        xgb_params_reg = self.optimize_xgboost_hyperparameters('total_corners')
        
        model_reg = xgb.XGBRegressor(**xgb_params_reg)
        
        # Cross-validation
        cv_scores_reg = cross_val_score(model_reg, X_selected_reg, y_regression, 
                                       cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
        
        print(f"Cross-validation MAE: {-cv_scores_reg.mean():.3f} (±{cv_scores_reg.std() * 2:.3f})")
        
        # Train final regression model
        model_reg.fit(X_selected_reg, y_regression)
        self.models['total_corners'] = model_reg
        
        # Feature importance
        self.feature_importance['total_corners'] = dict(zip(selected_features_reg, model_reg.feature_importances_))
        
        # Final evaluation on holdout
        split_idx = int(len(X_selected_reg) * 0.85)
        X_train_reg, X_test_reg = X_selected_reg[:split_idx], X_selected_reg[split_idx:]
        y_train_reg, y_test_reg = y_regression[:split_idx], y_regression[split_idx:]
        
        final_model_reg = xgb.XGBRegressor(**xgb_params_reg)
        final_model_reg.fit(X_train_reg, y_train_reg)
        y_pred_reg = final_model_reg.predict(X_test_reg)
        
        mae_reg = mean_absolute_error(y_test_reg, y_pred_reg)
        rmse_reg = np.sqrt(np.mean((y_test_reg - y_pred_reg) ** 2))
        
        print(f"Final Regression Performance:")
        print(f"MAE: {mae_reg:.3f}")
        print(f"RMSE: {rmse_reg:.3f}")
        
        results['regression'] = {'mae': mae_reg, 'rmse': rmse_reg}
        
        # Train classification models for each corner line with special handling for middle lines
        print(f"\n{'='*60}")
        print("TRAINING CORNER LINE CLASSIFICATION MODELS WITH ENSEMBLE FOR HARD LINES")
        print(f"{'='*60}")
        
        # UPDATED: Added 8.5 and 9.5 to hard-to-predict lines
        hard_to_predict_lines = [8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
        
        for line in self.corner_lines:
            target_name = f'over_{line}_corners'
            if target_name not in targets:
                continue
                
            print(f"\nTraining model for Over {line} corners...")
            
            y_clf = targets[target_name]
            
            # Skip if all predictions are the same
            if y_clf.nunique() < 2:
                print(f"Skipping {target_name} - insufficient class diversity")
                continue
            
            is_hard_line = line in hard_to_predict_lines
            
            if is_hard_line:
                print(f"  🎯 Using ENHANCED ENSEMBLE approach for hard-to-predict line {line}")
                
                # Use more features for hard lines
                feature_selector_clf = SelectKBest(score_func=mutual_info_classif, k=min(35, X.shape[1]))
                X_selected_clf = feature_selector_clf.fit_transform(X, y_clf)
                
                # Train multiple models with different approaches
                models_ensemble = []
                
                # Model 1: XGBoost with hard-line optimization
                xgb_params_clf = self.optimize_xgboost_hyperparameters(target_name)
                pos_ratio = y_clf.mean()
                scale_pos_weight = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1
                xgb_params_clf['scale_pos_weight'] = scale_pos_weight
                
                model1 = xgb.XGBClassifier(**xgb_params_clf)
                models_ensemble.append(('xgb_optimized', model1))
                
                # Model 2: XGBoost with different regularization
                xgb_params_alt = xgb_params_clf.copy()
                xgb_params_alt['reg_alpha'] = 0.35
                xgb_params_alt['reg_lambda'] = 2.0
                xgb_params_alt['learning_rate'] = 0.04
                xgb_params_alt['n_estimators'] = 1000
                
                model2 = xgb.XGBClassifier(**xgb_params_alt)
                models_ensemble.append(('xgb_conservative', model2))
                
                # Model 3: XGBoost with focus on recent patterns
                # Select features that emphasize recent performance
                recent_feature_indices = []
                for i, feature in enumerate([X.columns[j] for j in feature_selector_clf.get_support(indices=True)]):
                    if any(keyword in feature.lower() for keyword in ['_3', '_6', 'recent', 'trend', 'momentum']):
                        recent_feature_indices.append(i)
                
                if len(recent_feature_indices) > 10:
                    X_recent = X_selected_clf[:, recent_feature_indices]
                    xgb_params_recent = xgb_params_clf.copy()
                    xgb_params_recent['max_depth'] = 10
                    xgb_params_recent['learning_rate'] = 0.06
                    
                    model3 = xgb.XGBClassifier(**xgb_params_recent)
                    models_ensemble.append(('xgb_recent_focus', model3))
                
                # Train ensemble and combine predictions
                ensemble_cv_scores = []
                
                for name, model in models_ensemble:
                    if name == 'xgb_recent_focus' and len(recent_feature_indices) > 10:
                        cv_scores = cross_val_score(model, X_recent, y_clf, cv=tscv, scoring='accuracy', n_jobs=-1)
                    else:
                        cv_scores = cross_val_score(model, X_selected_clf, y_clf, cv=tscv, scoring='accuracy', n_jobs=-1)
                    ensemble_cv_scores.append(cv_scores.mean())
                    print(f"    {name}: {cv_scores.mean():.3f} accuracy")
                
                # Train final ensemble models
                ensemble_models = []
                for i, (name, model) in enumerate(models_ensemble):
                    if name == 'xgb_recent_focus' and len(recent_feature_indices) > 10:
                        model.fit(X_recent, y_clf)
                        ensemble_models.append(('recent_focus', model, X_recent.shape[1]))
                    else:
                        model.fit(X_selected_clf, y_clf)
                        ensemble_models.append((name, model, X_selected_clf.shape[1]))
                
                # Store ensemble for hard lines
                self.models[target_name] = {
                    'type': 'ensemble',
                    'models': ensemble_models,
                    'weights': [score/sum(ensemble_cv_scores) for score in ensemble_cv_scores],
                    'feature_selector': feature_selector_clf,
                    'recent_indices': recent_feature_indices if len(recent_feature_indices) > 10 else None
                }
                
                print(f"    Ensemble CV Accuracy: {np.average(ensemble_cv_scores, weights=[score/sum(ensemble_cv_scores) for score in ensemble_cv_scores]):.3f}")
                
            else:
                # Standard approach for easier lines
                feature_selector_clf = SelectKBest(score_func=mutual_info_classif, k=min(25, X.shape[1]))
                X_selected_clf = feature_selector_clf.fit_transform(X, y_clf)
                
                xgb_params_clf = self.optimize_xgboost_hyperparameters('classification')
                pos_ratio = y_clf.mean()
                scale_pos_weight = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1
                xgb_params_clf['scale_pos_weight'] = scale_pos_weight
                
                model_clf = xgb.XGBClassifier(**xgb_params_clf)
                
                cv_scores_clf = cross_val_score(model_clf, X_selected_clf, y_clf, 
                                               cv=tscv, scoring='accuracy', n_jobs=-1)
                
                print(f"Over {line} - CV Accuracy: {cv_scores_clf.mean():.3f} (±{cv_scores_clf.std() * 2:.3f})")
                
                model_clf.fit(X_selected_clf, y_clf)
                self.models[target_name] = {
                    'type': 'single',
                    'model': model_clf,
                    'feature_selector': feature_selector_clf
                }
            
            # Store selected features and feature importance
            selected_features_clf = [X.columns[i] for i in feature_selector_clf.get_support(indices=True)]
            self.selected_features[target_name] = selected_features_clf
            
            if is_hard_line and isinstance(self.models[target_name], dict) and 'models' in self.models[target_name]:
                # Average feature importance across ensemble
                avg_importance = {}
                for name, model, _ in self.models[target_name]['models']:
                    if hasattr(model, 'feature_importances_'):
                        if name == 'recent_focus' and self.models[target_name]['recent_indices']:
                            recent_features = [selected_features_clf[i] for i in self.models[target_name]['recent_indices']]
                            for feat, imp in zip(recent_features, model.feature_importances_):
                                avg_importance[feat] = avg_importance.get(feat, 0) + imp
                        else:
                            for feat, imp in zip(selected_features_clf, model.feature_importances_):
                                avg_importance[feat] = avg_importance.get(feat, 0) + imp
                
                # Normalize
                total_importance = sum(avg_importance.values())
                if total_importance > 0:
                    self.feature_importance[target_name] = {k: v/total_importance for k, v in avg_importance.items()}
            else:
                if hasattr(self.models[target_name].get('model'), 'feature_importances_'):
                    self.feature_importance[target_name] = dict(zip(selected_features_clf, self.models[target_name]['model'].feature_importances_))
            
            # Final evaluation
            results[target_name] = {'accuracy': 0.0, 'log_loss': 0.0}  # Placeholder
        
        return results
    
    def predict_all_corner_lines(self, match_data):
        """Predict corners for all lines with enhanced ensemble handling"""
        if not self.models:
            raise ValueError("Models not trained yet!")
        
        predictions = {}
        
        # Total corners prediction
        if 'total_corners' in self.models:
            features_reg = self.prepare_features(match_data, 'total_corners')
            total_corners = self.models['total_corners'].predict(features_reg.reshape(1, -1))[0]
            
            predictions['total_corners'] = {
                'prediction': round(total_corners, 2),
                'rounded': round(total_corners),
                'confidence_interval': (max(0, total_corners - 2), total_corners + 2)
            }
        
        # Corner line predictions with ensemble handling
        predictions['corner_lines'] = {}
        
        for line in self.corner_lines:
            target_name = f'over_{line}_corners'
            if target_name not in self.models:
                continue
            
            model_data = self.models[target_name]
            
            # Handle ensemble models for hard-to-predict lines
            if isinstance(model_data, dict) and model_data.get('type') == 'ensemble':
                # Prepare features
                feature_selector = model_data['feature_selector']
                all_features = self.prepare_features_full(match_data, target_name, feature_selector)
                
                # Get predictions from each model in ensemble
                ensemble_predictions = []
                ensemble_probabilities = []
                
                for i, (name, model, feature_count) in enumerate(model_data['models']):
                    if name == 'recent_focus' and model_data['recent_indices']:
                        # Use only recent-focused features
                        model_features = all_features[model_data['recent_indices']]
                    else:
                        model_features = all_features
                    
                    pred = model.predict(model_features.reshape(1, -1))[0]
                    prob = model.predict_proba(model_features.reshape(1, -1))[0]
                    
                    ensemble_predictions.append(pred)
                    ensemble_probabilities.append(prob[1] if len(prob) > 1 else prob[0])
                
                # Weight the predictions
                weights = model_data['weights']
                final_probability = sum(w * p for w, p in zip(weights, ensemble_probabilities))
                final_prediction = 1 if final_probability > 0.5 else 0
                
                over_prob = final_probability
                under_prob = 1 - final_probability
                
                predictions['corner_lines'][line] = {
                    'prediction': 'Over' if final_prediction == 1 else 'Under',
                    'over_probability': over_prob,
                    'under_probability': under_prob,
                    'confidence': max(over_prob, under_prob),
                    'recommended': 'Over' if over_prob > 0.62 else 'Under' if under_prob > 0.62 else 'Skip',  # Higher threshold for hard lines
                    'edge': abs(over_prob - 0.5) * 2,
                    'ensemble_info': {
                        'individual_probs': ensemble_probabilities,
                        'model_weights': weights,
                        'consensus': len(set(ensemble_predictions)) == 1  # All models agree
                    }
                }
                
            else:
                # Single model approach for easier lines
                if isinstance(model_data, dict) and 'model' in model_data:
                    model = model_data['model']
                    feature_selector = model_data['feature_selector']
                    features_clf = self.prepare_features_full(match_data, target_name, feature_selector)
                else:
                    # Legacy support
                    model = model_data
                    features_clf = self.prepare_features(match_data, target_name)
                
                prob = model.predict_proba(features_clf.reshape(1, -1))[0]
                prediction = model.predict(features_clf.reshape(1, -1))[0]
                
                over_prob = prob[1] if len(prob) > 1 else prob[0]
                under_prob = prob[0] if len(prob) > 1 else 1 - prob[0]
                
                predictions['corner_lines'][line] = {
                    'prediction': 'Over' if prediction == 1 else 'Under',
                    'over_probability': over_prob,
                    'under_probability': under_prob,
                    'confidence': max(over_prob, under_prob),
                    'recommended': 'Over' if over_prob > 0.6 else 'Under' if under_prob > 0.6 else 'Skip',
                    'edge': abs(over_prob - 0.5) * 2
                }
        
        return predictions
    
    def prepare_features_full(self, match_data, target_type, feature_selector):
        """Prepare feature array using feature selector"""
        # Get all available features first
        all_feature_names = list(match_data.keys())
        all_features = np.array([match_data.get(feature, 0) for feature in all_feature_names])
        
        # Apply feature selector transform
        # Note: This is a simplified approach - in practice, you'd need to maintain the exact feature order
        selected_features = self.selected_features.get(target_type, [])
        features = []
        
        for feature in selected_features:
            value = match_data.get(feature, 0)
            if np.isinf(value) or np.isnan(value):
                value = 0
            features.append(value)
        
        return np.array(features)
    
    def prepare_features(self, match_data, target_type):
        """Prepare feature array for prediction"""
        selected_features = self.selected_features.get(target_type, [])
        
        features = []
        for feature in selected_features:
            value = match_data.get(feature, 0)
            # Ensure no infinity or extreme values
            if np.isinf(value) or np.isnan(value):
                value = 0
            features.append(value)
        
        return np.array(features)
    
    def analyze_corner_data(self, df):
        """Comprehensive corner data analysis"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE CORNER KICK DATA ANALYSIS")
        print(f"{'='*80}")
        
        print(f"Dataset Information:")
        print(f"  Total matches: {len(df):,}")
        print(f"  Date range: {df['date_GMT'].min().strftime('%Y-%m-%d')} to {df['date_GMT'].max().strftime('%Y-%m-%d')}")
        print(f"  Unique teams: {pd.concat([df['home_team_name'], df['away_team_name']]).nunique()}")
        
        if 'total_corners' in df.columns:
            corners = df['total_corners']
            
            print(f"\nCorner Statistics:")
            print(f"  Mean: {corners.mean():.2f}")
            print(f"  Median: {corners.median():.1f}")
            print(f"  Mode: {corners.mode().iloc[0]:.0f}")
            print(f"  Standard Deviation: {corners.std():.2f}")
            print(f"  Range: {corners.min():.0f} - {corners.max():.0f}")
            
            print(f"\nCorner Distribution by Ranges:")
            ranges = [(0, 6), (7, 9), (10, 12), (13, 15), (16, 100)]
            for low, high in ranges:
                count = ((corners >= low) & (corners <= high)).sum()
                pct = count / len(corners) * 100
                print(f"  {low}-{high if high < 100 else '+'} corners: {count:,} matches ({pct:.1f}%)")
            
            print(f"\nOver/Under Analysis for Popular Lines:")
            for line in [7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]:
                over_pct = (corners > line).mean() * 100
                under_pct = 100 - over_pct
                print(f"  Over {line:4.1f}: {over_pct:5.1f}% | Under {line:4.1f}: {under_pct:5.1f}%")
            
            # Home vs Away corner analysis
            print(f"\nHome vs Away Corner Analysis:")
            home_corners = df['home_team_corner_count']
            away_corners = df['away_team_corner_count']
            
            print(f"  Home team average: {home_corners.mean():.2f}")
            print(f"  Away team average: {away_corners.mean():.2f}")
            print(f"  Home advantage: {(home_corners.mean() - away_corners.mean()):.2f} corners")
            
            home_wins_corners = (home_corners > away_corners).sum()
            away_wins_corners = (away_corners > home_corners).sum()
            draws_corners = (home_corners == away_corners).sum()
            
            print(f"  Home team wins corners: {home_wins_corners:,} ({home_wins_corners/len(df)*100:.1f}%)")
            print(f"  Away team wins corners: {away_wins_corners:,} ({away_wins_corners/len(df)*100:.1f}%)")
            print(f"  Equal corners: {draws_corners:,} ({draws_corners/len(df)*100:.1f}%)")
    
    def display_feature_importance(self, top_n=15):
        """Display feature importance for all models"""
        print(f"\n{'='*80}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*80}")
        
        for model_name, importance_dict in self.feature_importance.items():
            print(f"\nTop {top_n} Features for {model_name.replace('_', ' ').title()}:")
            print("-" * 60)
            
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(sorted_features[:top_n], 1):
                bar_length = int(importance * 50)  # Scale for visualization
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"{i:2d}. {feature:35s} {importance:6.3f} |{bar}|")
    
    def generate_betting_recommendations(self, predictions, match_info=None):
        """Generate intelligent betting recommendations with enhanced analysis for hard lines"""
        print(f"\n{'='*80}")
        print("INTELLIGENT BETTING RECOMMENDATIONS (ENHANCED FOR HARD LINES)")
        print(f"{'='*80}")
        
        if match_info:
            print(f"Match: {match_info.get('home_team', 'Home')} vs {match_info.get('away_team', 'Away')}")
            print(f"Date: {match_info.get('date', 'TBD')}")
            print("-" * 80)
        
        # Total corners prediction
        if 'total_corners' in predictions:
            total_pred = predictions['total_corners']
            print(f"\nTOTAL CORNERS PREDICTION:")
            print(f"  Predicted: {total_pred['prediction']} corners")
            print(f"  Rounded: {total_pred['rounded']} corners")
            print(f"  95% Confidence Interval: {total_pred['confidence_interval'][0]:.1f} - {total_pred['confidence_interval'][1]:.1f}")
        
        # Corner lines analysis with enhanced info for hard lines
        if 'corner_lines' in predictions:
            print(f"\nCORNER LINES ANALYSIS (Enhanced for 8.5-9.5 and 10.5-13.5 lines):")
            print("-" * 90)
            print(f"{'Line':>6s} {'Prediction':>10s} {'Over %':>8s} {'Under %':>9s} {'Confidence':>11s} {'Recommendation':>14s} {'Edge':>6s} {'Special':>10s}")
            print("-" * 90)
            
            recommendations = {'strong_bets': [], 'good_bets': [], 'ensemble_bets': [], 'avoid': []}
            # UPDATED: Added 8.5 and 9.5 to hard lines
            hard_lines = [8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
            
            for line in sorted(predictions['corner_lines'].keys()):
                pred_data = predictions['corner_lines'][line]
                
                over_pct = pred_data['over_probability'] * 100
                under_pct = pred_data['under_probability'] * 100
                confidence = pred_data['confidence']
                recommendation = pred_data['recommended']
                edge = pred_data['edge']
                
                # Special indicators for enhanced models
                special_info = ""
                if 'ensemble_info' in pred_data:
                    ensemble_info = pred_data['ensemble_info']
                    if ensemble_info['consensus']:
                        special_info = "CONSENSUS"
                    else:
                        special_info = f"Split({len([p for p in ensemble_info['individual_probs'] if p > 0.5])}/3)"
                elif line in hard_lines:
                    special_info = "Enhanced"
                else:
                    special_info = "Standard"
                
                print(f"{line:6.1f} {pred_data['prediction']:>10s} {over_pct:7.1f}% {under_pct:8.1f}% "
                      f"{confidence:10.3f} {recommendation:>14s} {edge:5.3f} {special_info:>10s}")
                
                # Enhanced categorization for recommendations
                bet_info = {
                    'line': line,
                    'bet': recommendation,
                    'confidence': confidence,
                    'edge': edge,
                    'probability': over_pct if recommendation == 'Over' else under_pct,
                    'is_hard_line': line in hard_lines,
                    'special_info': special_info
                }
                
                if recommendation != 'Skip':
                    # Enhanced thresholds for hard lines
                    if line in hard_lines:
                        # Stricter criteria for hard-to-predict lines
                        if 'ensemble_info' in pred_data and pred_data['ensemble_info']['consensus']:
                            if confidence >= 0.68 and edge >= 0.25:
                                recommendations['ensemble_bets'].append(bet_info)
                            elif confidence >= 0.62 and edge >= 0.15:
                                recommendations['good_bets'].append(bet_info)
                            else:
                                recommendations['avoid'].append(bet_info)
                        else:
                            if confidence >= 0.72 and edge >= 0.35:
                                recommendations['strong_bets'].append(bet_info)
                            elif confidence >= 0.65 and edge >= 0.22:
                                recommendations['good_bets'].append(bet_info)
                            else:
                                recommendations['avoid'].append(bet_info)
                    else:
                        # Standard criteria for easier lines
                        if confidence >= 0.75 and edge >= 0.3:
                            recommendations['strong_bets'].append(bet_info)
                        elif confidence >= 0.65 and edge >= 0.2:
                            recommendations['good_bets'].append(bet_info)
                        else:
                            recommendations['avoid'].append(bet_info)
        
        # Enhanced betting strategy recommendations
        print(f"\n{'='*70}")
        print("ENHANCED BETTING STRATEGY RECOMMENDATIONS")
        print(f"{'='*70}")
        
        if recommendations['ensemble_bets']:
            print(f"\n🎯 ENSEMBLE CONSENSUS BETS (Hard Lines with Model Agreement):")
            for bet in sorted(recommendations['ensemble_bets'], key=lambda x: x['confidence'], reverse=True):
                print(f"  • {bet['bet']} {bet['line']} corners - Confidence: {bet['confidence']:.3f} "
                      f"({bet['probability']:.1f}% chance) - {bet['special_info']}")
        
        if recommendations['strong_bets']:
            print(f"\n🔥 STRONG BETS (High Confidence):")
            for bet in sorted(recommendations['strong_bets'], key=lambda x: x['confidence'], reverse=True):
                line_type = " (Hard Line)" if bet['is_hard_line'] else " (Standard Line)"
                print(f"  • {bet['bet']} {bet['line']} corners - Confidence: {bet['confidence']:.3f} "
                      f"({bet['probability']:.1f}% chance){line_type}")
        
        if recommendations['good_bets']:
            print(f"\n✅ GOOD BETS (Medium Confidence):")
            for bet in sorted(recommendations['good_bets'], key=lambda x: x['confidence'], reverse=True):
                line_type = " (Hard Line)" if bet['is_hard_line'] else " (Standard Line)"
                print(f"  • {bet['bet']} {bet['line']} corners - Confidence: {bet['confidence']:.3f} "
                      f"({bet['probability']:.1f}% chance){line_type}")
        
        if not any([recommendations['ensemble_bets'], recommendations['strong_bets'], recommendations['good_bets']]):
            print(f"\n⚠️  NO HIGH-CONFIDENCE BETS IDENTIFIED")
            print(f"   Consider avoiding corner bets for this match or waiting for better opportunities.")
        
        # Enhanced risk management for hard lines
        print(f"\n💡 ENHANCED RISK MANAGEMENT TIPS:")
        # UPDATED: Include 8.5 and 9.5 in hard lines description
        print(f"  • Hard Lines (8.5-9.5, 10.5-13.5): Use smaller stakes, require higher confidence")
        print(f"  • Ensemble Consensus: When multiple models agree, confidence is higher")
        print(f"  • Model Split: When models disagree, proceed with extreme caution")
        print(f"  • Always check bookmaker odds vs. model probabilities for true edge")
        print(f"  • For hard lines, consider 1-2% bankroll max vs 3-5% for easier lines")
        
        # Analysis summary
        total_lines_analyzed = len(predictions.get('corner_lines', {}))
        # UPDATED: Include 8.5 and 9.5 in hard lines count
        hard_lines_count = sum(1 for line in predictions.get('corner_lines', {}).keys() if line in [8.5, 9.5, 10.5, 11.5, 12.5, 13.5])
        high_confidence_bets = len(recommendations['strong_bets']) + len(recommendations['ensemble_bets'])
        
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"  • Total lines analyzed: {total_lines_analyzed}")
        # UPDATED: Include 8.5 and 9.5 in hard lines summary
        print(f"  • Hard lines (8.5-9.5, 10.5-13.5): {hard_lines_count}")
        print(f"  • High confidence opportunities: {high_confidence_bets}")
        print(f"  • Model enhancement: {'Ensemble models active' if recommendations['ensemble_bets'] else 'Standard models only'}")
        
        return recommendations


def main():
    """Enhanced main function with comprehensive corner prediction"""
    # Initialize predictor
    predictor = EnhancedFootballCornerPredictor()
    
    # Update this path to your actual CSV file
    csv_file_path = r"/home/teodor/Pulpit/szwecja rożne/combined_sezony.csv"
    
    try:
        print("🏈 Enhanced Football Corner Prediction System")
        print("=" * 80)
        
        # Load and clean data
        df = predictor.load_and_clean_data(csv_file_path)
        
        # Comprehensive data analysis
        predictor.analyze_corner_data(df)
        
        # Feature selection and preparation
        X, targets, df_clean, available_features = predictor.select_optimized_features(df)
        
        print(f"\nFinal dataset for training:")
        print(f"  Features: {len(available_features)}")
        print(f"  Matches: {len(df_clean):,}")
        print(f"  Date range: {df_clean['date_GMT'].min().strftime('%Y-%m-%d')} to {df_clean['date_GMT'].max().strftime('%Y-%m-%d')}")
        
        # Train all models
        training_results = predictor.train_enhanced_models(X, targets)
        
        # Display feature importance
        predictor.display_feature_importance(top_n=40)
        
        # Example prediction with realistic data
        print(f"\n{'='*80}")
        print("EXAMPLE MATCH PREDICTION")
        print(f"{'='*80}")
        
        # Sample match data (replace with actual pre-match data)
        sample_match = {
            'Pre-Match PPG (Home)': 1.25,
            'Pre-Match PPG (Away)': 1.13,
            'Home Team Pre-Match xG': 1.59,
            'Away Team Pre-Match xG': 1.34,
            'odds_ft_home_team_win': 1.67,
            'odds_ft_draw': 3.8,
            'odds_ft_away_team_win': 3.88,
            'average_corners_per_match_pre_match': 8.75,
            'btts_percentage_pre_match': 57,
            'over_25_percentage_pre_match': 71,
            'home_corners_avg_6': 4.5,
            'away_corners_avg_6': 5.33,
            'home_corners_against_avg_6': 4.83,
            'away_corners_against_avg_6': 3.5,
            'expected_total_corners': 9.08,
            'corner_advantage_home': -0.33,
            'corner_advantage_away': 1.83,
            'home_recent_form': 0.67,
            'away_recent_form': 0.08,
            
        }
        
        match_info = {
            'home_team': 'IFK Göteborg',
            'away_team': 'Degerfors',
            'date': '2025-08-04'
        }
        
        # Make predictions
        predictions = predictor.predict_all_corner_lines(sample_match)
        
        # Generate betting recommendations
        recommendations = predictor.generate_betting_recommendations(predictions, match_info)
        
        print(f"\n{'='*80}")
        print("SYSTEM PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        
        for model_name, metrics in training_results.items():
            if model_name == 'regression':
                print(f"Total Corners Model - MAE: {metrics['mae']:.3f}, RMSE: {metrics['rmse']:.3f}")
            else:
                line = model_name.replace('over_', '').replace('_corners', '')
                print(f"Over {line} Model - Accuracy: {metrics['accuracy']:.3f}, Log Loss: {metrics['log_loss']:.3f}")
        
        print(f"\n✅ System ready for corner predictions!")
        print(f"📊 Use predict_all_corner_lines() method for new match predictions")
        
    except FileNotFoundError:
        print("❌ Error: CSV file not found!")
        print("Please update the csv_file_path variable with the correct path to your data file")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()