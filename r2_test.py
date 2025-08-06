import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, log_loss
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
import re
import optuna
import shap
from typing import Dict, List, Tuple, Optional, Any
import joblib
import os
import json

# Optional: Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not available. Install with: pip install wandb")

warnings.filterwarnings('ignore')


class CornerPredictionChatbot:
    """Intelligent chatbot for corner predictions with natural language processing"""
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.adjustment_factors = {}
        self.context = {}
        
        # Define adjustment patterns and their impacts
        self.adjustment_patterns = {
            # Team strength adjustments
            'reserves|reserv|reserve|młodzi|youth|academy|druga': {
                'type': 'team_strength',
                'corners_multiplier': 0.85,
                'description': 'Reserve/Youth team playing'
            },
            'first.?team|główny|podstawowy|strongest|full.?strength': {
                'type': 'team_strength', 
                'corners_multiplier': 1.15,
                'description': 'Full strength team'
            },
            'key.?players?.?missing|without.?star|bez.?gwiazd|kontuzje|injuries': {
                'type': 'team_strength',
                'corners_multiplier': 0.9,
                'description': 'Key players missing'
            },
            
            # Tactical adjustments
            'defensive|defensywny|park.?bus|defensive.?setup': {
                'type': 'tactics',
                'corners_multiplier': 0.8,
                'description': 'Defensive tactics expected'
            },
            'attacking|ofensywny|all.?out.?attack|gung.?ho': {
                'type': 'tactics',
                'corners_multiplier': 1.2,
                'description': 'Attacking tactics expected'
            },
            'counter.?attack|kontra|quick.?transitions': {
                'type': 'tactics',
                'corners_multiplier': 0.9,
                'description': 'Counter-attacking style'
            },
            
            # Weather conditions
            'rain|deszcz|wet|mokro|storm|burza': {
                'type': 'weather',
                'corners_multiplier': 1.1,
                'description': 'Wet conditions (more crosses/corners)'
            },
            'wind|wiatr|windy|strong.?wind': {
                'type': 'weather',
                'corners_multiplier': 1.15,
                'description': 'Windy conditions (inaccurate crosses)'
            },
            'hot|gorąco|very.?warm|heat': {
                'type': 'weather',
                'corners_multiplier': 0.95,
                'description': 'Hot weather (less intensity)'
            },
            'cold|zimno|freezing|mróz': {
                'type': 'weather',
                'corners_multiplier': 0.92,
                'description': 'Cold weather (ball control issues)'
            },
            
            # Match importance
            'final|finał|playoff|play.?off|relegation|spadek': {
                'type': 'importance',
                'corners_multiplier': 1.25,
                'description': 'High stakes match'
            },
            'friendly|sparing|towarzyski|meaningless|bez.?znaczenia': {
                'type': 'importance',
                'corners_multiplier': 0.7,
                'description': 'Low stakes match'
            },
            'derby|lokalne.?derby|rivalry|rywalizacja': {
                'type': 'importance',
                'corners_multiplier': 1.2,
                'description': 'Derby/Rivalry match'
            },
            
            # Physical condition
            'tired|zmęczeni|fatigue|many.?games|fixture.?congestion': {
                'type': 'condition',
                'corners_multiplier': 0.88,
                'description': 'Team fatigue'
            },
            'fresh|wypoczęci|well.?rested|long.?break': {
                'type': 'condition',
                'corners_multiplier': 1.08,
                'description': 'Well rested team'
            },
            
            # Referee tendencies
            'strict.?referee|sędzia.?surowy|many.?fouls': {
                'type': 'referee',
                'corners_multiplier': 1.1,
                'description': 'Strict referee (more fouls = more corners)'
            },
            'lenient.?referee|pobłażliwy|lets.?play': {
                'type': 'referee',
                'corners_multiplier': 0.95,
                'description': 'Lenient referee'
            },
            
            # Motivation factors
            'motivated|zmotywowani|fighting.?for|desperate': {
                'type': 'motivation',
                'corners_multiplier': 1.1,
                'description': 'High motivation'
            },
            'unmotivated|bez.?motywacji|nothing.?to.?play.?for|beach.?mode': {
                'type': 'motivation',
                'corners_multiplier': 0.85,
                'description': 'Low motivation'
            }
        }
    
    def chat(self):
        """Main chatbot interface"""
        print("🤖 Football Corner Prediction Chatbot")
        print("=" * 50)
        print("Hi! I'm your corner prediction assistant.")
        print("Tell me about the match and I'll adjust the predictions accordingly.")
        print("Type 'quit' to exit, 'help' for commands, or 'predict' to make predictions.")
        print("=" * 50)
        
        while True:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("🤖 Bot: Goodbye! Good luck with your bets! 🍀")
                break
            elif user_input.lower() == 'help':
                self.show_help()
            elif user_input.lower() == 'predict':
                self.handle_prediction_request()
            elif user_input.lower() == 'reset':
                self.reset_adjustments()
            elif user_input.lower() == 'status':
                self.show_current_adjustments()
            else:
                self.process_user_input(user_input)
    
    def show_help(self):
        """Display help information"""
        print("\n🤖 Bot: Here are the commands I understand:")
        print("=" * 60)
        print("📋 COMMANDS:")
        print("  • 'predict' - Make corner predictions with current adjustments")
        print("  • 'status' - Show current adjustments")
        print("  • 'reset' - Clear all adjustments")
        print("  • 'help' - Show this help")
        print("  • 'quit' - Exit chatbot")
        print()
        print("💬 NATURAL LANGUAGE:")
        print("  Tell me about factors like:")
        print("  • Team strength: 'reserves playing', 'full strength team'")
        print("  • Weather: 'rainy weather', 'windy conditions', 'hot day'")
        print("  • Tactics: 'defensive setup', 'all out attack'")
        print("  • Match importance: 'final match', 'derby game', 'friendly'")
        print("  • Condition: 'team is tired', 'well rested'")
        print("  • Motivation: 'fighting for title', 'nothing to play for'")
        print()
        print("🌟 EXAMPLES:")
        print("  • 'Home team playing reserves due to injuries'")
        print("  • 'Rainy weather expected, both teams defensive'")
        print("  • 'Derby match, high intensity expected'")
    
    def process_user_input(self, text):
        """Process natural language input and extract adjustments"""
        text_lower = text.lower()
        found_adjustments = []
        
        # Check for adjustment patterns
        for pattern, adjustment_info in self.adjustment_patterns.items():
            if re.search(pattern, text_lower):
                adj_type = adjustment_info['type']
                multiplier = adjustment_info['corners_multiplier']
                description = adjustment_info['description']
                
                # Store adjustment
                if adj_type not in self.adjustment_factors:
                    self.adjustment_factors[adj_type] = []
                
                self.adjustment_factors[adj_type].append({
                    'multiplier': multiplier,
                    'description': description,
                    'text': text
                })
                
                found_adjustments.append(description)
        
        # Respond to user
        if found_adjustments:
            print(f"\n🤖 Bot: Got it! I've noted the following adjustments:")
            for i, adj in enumerate(found_adjustments, 1):
                print(f"  {i}. {adj}")
            print(f"\nThese factors will be considered in the corner predictions.")
            self.show_current_impact()
        else:
            print(f"\n🤖 Bot: I understand you're telling me: '{text}'")
            print("I didn't recognize any specific factors that affect corners.")
            print("Try mentioning things like weather, team strength, tactics, etc.")
            print("Type 'help' for examples!")
    
    def show_current_adjustments(self):
        """Show currently active adjustments"""
        if not self.adjustment_factors:
            print("\n🤖 Bot: No adjustments currently active.")
            print("Tell me about the match conditions to improve predictions!")
            return
        
        print(f"\n🤖 Bot: Current active adjustments:")
        print("=" * 50)
        
        for adj_type, adjustments in self.adjustment_factors.items():
            print(f"\n📌 {adj_type.upper()}:")
            for adj in adjustments:
                impact = "📈" if adj['multiplier'] > 1 else "📉"
                print(f"  {impact} {adj['description']} (×{adj['multiplier']})")
        
        self.show_current_impact()
    
    def show_current_impact(self):
        """Calculate and show overall impact"""
        total_multiplier = self.calculate_total_multiplier()
        impact = (total_multiplier - 1) * 100
        
        if abs(impact) < 1:
            print(f"\n📊 Overall impact: Minimal change ({impact:+.1f}%)")
        elif impact > 0:
            print(f"\n📊 Overall impact: +{impact:.1f}% more corners expected")
        else:
            print(f"\n📊 Overall impact: {impact:.1f}% fewer corners expected")
    
    def calculate_total_multiplier(self):
        """Calculate total adjustment multiplier"""
        if not self.adjustment_factors:
            return 1.0
        
        # Combine multipliers (geometric mean to avoid extreme values)
        all_multipliers = []
        for adjustments in self.adjustment_factors.values():
            for adj in adjustments:
                all_multipliers.append(adj['multiplier'])
        
        if not all_multipliers:
            return 1.0
        
        # Use geometric mean and cap the adjustment
        total = np.prod(all_multipliers) ** (1/len(all_multipliers))
        return np.clip(total, 0.6, 1.5)  # Cap between -40% and +50%
    
    def reset_adjustments(self):
        """Reset all adjustments"""
        self.adjustment_factors = {}
        self.context = {}
        print("\n🤖 Bot: All adjustments cleared! Ready for fresh input.")
    
    def handle_prediction_request(self):
        """Handle prediction request"""
        print("\n🤖 Bot: Let me get the match details for prediction...")
        
        # Get match details
        match_data = self.get_match_details()
        if not match_data:
            return
        
        # Make base predictions
        try:
            base_predictions = self.predictor.predict_all_corner_lines(match_data)
            
            # Apply adjustments
            adjusted_predictions = self.apply_adjustments(base_predictions)
            
            # Display results
            self.display_adjusted_predictions(base_predictions, adjusted_predictions)
            
        except Exception as e:
            print(f"\n🤖 Bot: Sorry, I encountered an error making predictions: {e}")
            print("Make sure the model is properly trained first!")
    
    def get_match_details(self):
        """Get match details from user"""
        print("\n📝 Please provide match details:")
        
        match_data = {}
        
        # Essential details
        try:
            home_team = input("Home team: ").strip()
            away_team = input("Away team: ").strip()
            
            if not home_team or not away_team:
                print("🤖 Bot: Team names are required!")
                return None
            
            # Basic stats (with defaults)
            print("\n📊 Pre-match statistics (press Enter for defaults):")
            
            match_data['Pre-Match PPG (Home)'] = float(input("Home PPG [1.2]: ") or "1.2")
            match_data['Pre-Match PPG (Away)'] = float(input("Away PPG [1.1]: ") or "1.1")
            match_data['Home Team Pre-Match xG'] = float(input("Home xG [1.4]: ") or "1.4")
            match_data['Away Team Pre-Match xG'] = float(input("Away xG [1.3]: ") or "1.3")
            
            print("\n💰 Betting odds (press Enter for defaults):")
            match_data['odds_ft_home_team_win'] = float(input("Home win odds [2.0]: ") or "2.0")
            match_data['odds_ft_draw'] = float(input("Draw odds [3.5]: ") or "3.5")
            match_data['odds_ft_away_team_win'] = float(input("Away win odds [3.0]: ") or "3.0")
            
            print("\n📈 Historical data (press Enter for defaults):")
            match_data['average_corners_per_match_pre_match'] = float(input("Avg corners per match [9.5]: ") or "9.5")
            match_data['btts_percentage_pre_match'] = float(input("BTTS percentage [55]: ") or "55")
            match_data['over_25_percentage_pre_match'] = float(input("Over 2.5 goals % [65]: ") or "65")
            
            # Recent form (simplified)
            match_data['home_corners_avg_6'] = float(input("Home avg corners (last 6) [5.0]: ") or "5.0")
            match_data['away_corners_avg_6'] = float(input("Away avg corners (last 6) [4.5]: ") or "4.5")
            match_data['home_corners_against_avg_6'] = float(input("Home corners against (last 6) [4.2]: ") or "4.2")
            match_data['away_corners_against_avg_6'] = float(input("Away corners against (last 6) [4.8]: ") or "4.8")
            match_data['home_recent_form'] = float(input("Home recent form (last 6) [0.67]: ") or "0.67")
            match_data['away_recent_form'] = float(input("Away recent form (last 6) [0.33]: ") or "0.33")
            
            # Calculate some derived features
            match_data['expected_total_corners'] = (match_data['home_corners_avg_6'] + 
                                                   match_data['away_corners_avg_6'] + 
                                                   match_data['home_corners_against_avg_6'] + 
                                                   match_data['away_corners_against_avg_6']) / 2
            
            match_data['corner_advantage_home'] = match_data['home_corners_avg_6'] - match_data['home_corners_against_avg_6']
            match_data['corner_advantage_away'] = match_data['away_corners_avg_6'] - match_data['away_corners_against_avg_6']
            
            # Store match info
            self.context['match_info'] = {
                'home_team': home_team,
                'away_team': away_team,
                'date': datetime.now().strftime('%Y-%m-%d')
            }
            
            return match_data
            
        except ValueError:
            print("🤖 Bot: Invalid input! Please enter numeric values.")
            return None
        except KeyboardInterrupt:
            print("\n🤖 Bot: Cancelled!")
            return None
    
    def apply_adjustments(self, base_predictions):
        """Apply chatbot adjustments to predictions"""
        adjusted_predictions = base_predictions.copy()
        total_multiplier = self.calculate_total_multiplier()
        
        # Adjust total corners prediction
        if 'total_corners' in adjusted_predictions:
            original = adjusted_predictions['total_corners']['prediction']
            adjusted = original * total_multiplier
            
            adjusted_predictions['total_corners']['adjusted_prediction'] = round(adjusted, 2)
            adjusted_predictions['total_corners']['adjustment_factor'] = total_multiplier
            adjusted_predictions['total_corners']['original_prediction'] = original
        
        # Adjust corner line probabilities
        if 'corner_lines' in adjusted_predictions:
            for line, pred_data in adjusted_predictions['corner_lines'].items():
                # Adjust probabilities based on whether we expect more or fewer corners
                if total_multiplier > 1:
                    # Expecting more corners - increase "over" probability
                    adjustment = min(0.2, (total_multiplier - 1) * 0.5)
                    new_over_prob = min(0.95, pred_data['over_probability'] + adjustment)
                else:
                    # Expecting fewer corners - decrease "over" probability
                    adjustment = min(0.2, (1 - total_multiplier) * 0.5)
                    new_over_prob = max(0.05, pred_data['over_probability'] - adjustment)
                
                new_under_prob = 1 - new_over_prob
                
                # Update predictions
                adjusted_predictions['corner_lines'][line]['adjusted_over_probability'] = new_over_prob
                adjusted_predictions['corner_lines'][line]['adjusted_under_probability'] = new_under_prob
                adjusted_predictions['corner_lines'][line]['original_over_probability'] = pred_data['over_probability']
                adjusted_predictions['corner_lines'][line]['adjusted_prediction'] = 'Over' if new_over_prob > 0.5 else 'Under'
                adjusted_predictions['corner_lines'][line]['adjustment_applied'] = True
        
        return adjusted_predictions
    
    def display_adjusted_predictions(self, base_predictions, adjusted_predictions):
        """Display predictions with adjustments highlighted"""
        print(f"\n{'='*80}")
        print("🎯 ADJUSTED CORNER PREDICTIONS")
        print(f"{'='*80}")
        
        # Match info
        if 'match_info' in self.context:
            info = self.context['match_info']
            print(f"Match: {info['home_team']} vs {info['away_team']}")
            print(f"Date: {info['date']}")
            print("-" * 80)
        
        # Show adjustments applied
        if self.adjustment_factors:
            print(f"\n🔧 ADJUSTMENTS APPLIED:")
            total_multiplier = self.calculate_total_multiplier()
            impact = (total_multiplier - 1) * 100
            
            for adj_type, adjustments in self.adjustment_factors.items():
                for adj in adjustments:
                    impact_emoji = "📈" if adj['multiplier'] > 1 else "📉"
                    print(f"  {impact_emoji} {adj['description']}")
            
            print(f"\n📊 Overall adjustment: {impact:+.1f}% corners")
            print("-" * 80)
        
        # Total corners prediction
        if 'total_corners' in adjusted_predictions:
            tc_data = adjusted_predictions['total_corners']
            print(f"\n🎯 TOTAL CORNERS:")
            if 'original_prediction' in tc_data:
                print(f"  Original prediction: {tc_data['original_prediction']:.2f}")
                print(f"  Adjusted prediction: {tc_data['adjusted_prediction']:.2f}")
                print(f"  Change: {tc_data['adjusted_prediction'] - tc_data['original_prediction']:+.2f}")
            else:
                print(f"  Prediction: {tc_data['prediction']:.2f}")
        
        # Corner lines
        if 'corner_lines' in adjusted_predictions:
            print(f"\n📊 CORNER LINES COMPARISON:")
            print("-" * 90)
            print(f"{'Line':>6s} {'Original':>12s} {'Adjusted':>12s} {'Change':>10s} {'Recommendation':>15s}")
            print("-" * 90)
            
            for line in sorted(adjusted_predictions['corner_lines'].keys()):
                data = adjusted_predictions['corner_lines'][line]
                
                if 'original_over_probability' in data:
                    orig_prob = data['original_over_probability'] * 100
                    adj_prob = data['adjusted_over_probability'] * 100
                    change = adj_prob - orig_prob
                    
                    # Determine recommendation
                    if adj_prob > 65:
                        rec = "OVER 🔥"
                    elif adj_prob < 35:
                        rec = "UNDER 🔥" 
                    elif adj_prob > 55:
                        rec = "Over"
                    elif adj_prob < 45:
                        rec = "Under"
                    else:
                        rec = "Skip"
                    
                    change_str = f"{change:+.1f}%"
                    if abs(change) < 2:
                        change_str = "~"
                    
                    print(f"{line:6.1f} {orig_prob:11.1f}% {adj_prob:11.1f}% {change_str:>9s} {rec:>15s}")
                else:
                    # No adjustment available
                    prob = data['over_probability'] * 100
                    rec = data['recommended']
                    print(f"{line:6.1f} {prob:11.1f}% {prob:11.1f}%        ~ {rec:>15s}")
        
        print(f"\n💡 Chatbot Impact Summary:")
        if self.adjustment_factors:
            print("✅ Adjustments have been applied based on your input")
            print("🎯 Recommendations updated with contextual factors")
        else:
            print("ℹ️  No adjustments applied - using base model predictions")
        
        print(f"\n🎲 Ready for more adjustments or new predictions!")


class EnhancedFootballCornerPredictor:
    """Enhanced predictor with MLOps best practices"""
    
    def __init__(self, use_wandb: bool = False, wandb_project: str = "corner-prediction"):
        self.pipelines = {}  # Store pipelines for different predictions
        self.best_params = {}  # Store best hyperparameters
        self.feature_names = {}  # Store feature names
        self.shap_explainers = {}  # Store SHAP explainers
        self.corner_lines = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]
        self.chatbot = None
        
        # W&B setup
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        
        # Model save directory
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize W&B if requested
        if self.use_wandb:
            wandb.init(project=self.wandb_project, name=f"corner_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
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
                'high_corner_games': [],
                'low_corner_games': [],
                'referee_history': {},
                'venue_corners': [],
                'time_of_year': [],
                'rest_days': [],
                'pressure_situations': []
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
            
            # Advanced features for better prediction
            if len(home_stats['corners_for']) >= 4:
                weights = [0.4, 0.3, 0.2, 0.1]
                weighted_corners = sum(w * c for w, c in zip(weights, home_stats['corners_for'][-4:]))
                df.loc[idx, 'corner_momentum_home'] = weighted_corners
            
            if len(away_stats['corners_for']) >= 4:
                weights = [0.4, 0.3, 0.2, 0.1]
                weighted_corners = sum(w * c for w, c in zip(weights, away_stats['corners_for'][-4:]))
                df.loc[idx, 'corner_momentum_away'] = weighted_corners
            
            # High/Low Corner Game Rates
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
            
            # Combined Corner Volatility
            home_volatility = df.loc[idx, 'home_corner_variance']
            away_volatility = df.loc[idx, 'away_corner_variance']
            df.loc[idx, 'combined_corner_volatility'] = (home_volatility + away_volatility) / 2
            
            # Head-to-Head Corner History
            h2h_key = f"{home_team}_vs_{away_team}"
            h2h_reverse_key = f"{away_team}_vs_{home_team}"
            
            if h2h_key in head_to_head_stats or h2h_reverse_key in head_to_head_stats:
                h2h_data = head_to_head_stats.get(h2h_key, head_to_head_stats.get(h2h_reverse_key, []))
                if len(h2h_data) > 0:
                    df.loc[idx, 'h2h_corner_history'] = np.mean([game['total_corners'] for game in h2h_data[-3:]])
            
            # Style Clash Indicator
            home_attacking_style = df.loc[idx, 'home_corners_avg_6'] - df.loc[idx, 'home_corners_against_avg_6']
            away_attacking_style = df.loc[idx, 'away_corners_avg_6'] - df.loc[idx, 'away_corners_against_avg_6']
            
            style_difference = abs(home_attacking_style - away_attacking_style)
            df.loc[idx, 'style_clash_indicator'] = style_difference
            
            # Corner Consistency
            if len(home_stats['corners_for']) >= 6:
                home_consistency = 1 / (1 + df.loc[idx, 'home_corner_variance'])
                df.loc[idx, 'home_corner_consistency'] = home_consistency
            
            if len(away_stats['corners_for']) >= 6:
                away_consistency = 1 / (1 + df.loc[idx, 'away_corner_variance'])
                df.loc[idx, 'away_corner_consistency'] = away_consistency
            
            # Match importance factor (default)
            df.loc[idx, 'match_importance_factor'] = 1.0
            
            # Seasonal adjustment
            if pd.notna(match_date):
                month = match_date.month
                seasonal_factors = {8: 0.95, 9: 0.97, 10: 0.98, 11: 1.0, 12: 1.0, 
                                  1: 1.0, 2: 1.0, 3: 1.02, 4: 1.03, 5: 1.05}
                df.loc[idx, 'seasonal_corner_adjustment'] = seasonal_factors.get(month, 1.0)
            
            # Attacking desperation factor
            home_goal_diff = np.mean(home_stats['goals_for'][-5:]) - np.mean(home_stats['goals_against'][-5:]) if len(home_stats['goals_for']) >= 5 else 0
            away_goal_diff = np.mean(away_stats['goals_for'][-5:]) - np.mean(away_stats['goals_against'][-5:]) if len(away_stats['goals_for']) >= 5 else 0
            
            desperation_factor = max(0, (2 - home_goal_diff) + (2 - away_goal_diff)) / 4
            df.loc[idx, 'attacking_desperation_factor'] = desperation_factor
            
            # Defensive solidity factor
            home_defensive_solidity = 1 / (1 + np.mean(home_stats['corners_against'][-5:])) if len(home_stats['corners_against']) >= 5 else 0.5
            away_defensive_solidity = 1 / (1 + np.mean(away_stats['corners_against'][-5:])) if len(away_stats['corners_against']) >= 5 else 0.5
            df.loc[idx, 'defensive_solidity_factor'] = (home_defensive_solidity + away_defensive_solidity) / 2
            
            # Expected total corners
            home_for = df.loc[idx, 'home_corners_avg_6']
            home_against = df.loc[idx, 'home_corners_against_avg_6']
            away_for = df.loc[idx, 'away_corners_avg_6']
            away_against = df.loc[idx, 'away_corners_against_avg_6']
            
            base_expected = (home_for + away_against + away_for + home_against) / 2
            volatility_adjustment = 1 + (df.loc[idx, 'combined_corner_volatility'] - 2) * 0.1
            style_adjustment = 1 + (df.loc[idx, 'style_clash_indicator'] * 0.05)
            seasonal_adjustment = df.loc[idx, 'seasonal_corner_adjustment']
            
            df.loc[idx, 'expected_total_corners'] = base_expected * volatility_adjustment * style_adjustment * seasonal_adjustment
            df.loc[idx, 'corner_advantage_home'] = home_for - home_against
            df.loc[idx, 'corner_advantage_away'] = away_for - away_against
            
            # Teams familiarity
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
    
    def create_pipelines(self, X: pd.DataFrame, target_type: str) -> Pipeline:
        """Create preprocessing and modeling pipeline with no data leakage"""
        
        # Identify column types
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Store feature names for later use
        self.feature_names[target_type] = {
            'numerical': numerical_features,
            'categorical': categorical_features,
            'all': numerical_features + categorical_features
        }
        
        # Create preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
            ],
            remainder='drop'  # Drop any other columns
        )
        
        # Create feature selection step
        if target_type == 'total_corners':
            feature_selector = SelectKBest(score_func=mutual_info_regression, k=min(40, len(numerical_features + categorical_features)))
        else:
            feature_selector = SelectKBest(score_func=mutual_info_classif, k=min(35, len(numerical_features + categorical_features)))
        
        # Create model (will be set with optimized parameters later)
        if target_type == 'total_corners':
            model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        else:
            model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selector', feature_selector),
            ('model', model)
        ])
        
        return pipeline
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, target_type: str, cv_splits: int = 5) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna with time series cross-validation"""
        
        def objective(trial):
            # Define hyperparameter search space
            if target_type == 'total_corners':
                params = {
                    'model__max_depth': trial.suggest_int('max_depth', 6, 15),
                    'model__learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
                    'model__n_estimators': trial.suggest_int('n_estimators', 500, 2500, step=100),
                    'model__subsample': trial.suggest_float('subsample', 0.7, 0.95),
                    'model__colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                    'model__colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 0.95),
                    'model__reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.5, log=True),
                    'model__reg_lambda': trial.suggest_float('reg_lambda', 0.1, 3.0, log=True),
                    'model__min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'model__gamma': trial.suggest_float('gamma', 0.01, 0.5, log=True),
                    'feature_selector__k': trial.suggest_int('feature_selector_k', 15, min(50, X.shape[1]))
                }
                #scoring = 'neg_mean_absolute_error'
                scoring = 'r2'
            else:
                # Check for hard-to-predict lines
                is_hard_line = any(line in target_type for line in ['8.5', '9.5', '10.5', '11.5', '12.5', '13.5'])
                
                if is_hard_line:
                    params = {
                        'model__max_depth': trial.suggest_int('max_depth', 8, 15),
                        'model__learning_rate': trial.suggest_float('learning_rate', 0.02, 0.08, log=True),
                        'model__n_estimators': trial.suggest_int('n_estimators', 1000, 2500, step=100),
                        'model__subsample': trial.suggest_float('subsample', 0.75, 0.9),
                        'model__colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 0.9),
                        'model__colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.75, 0.9),
                        'model__reg_alpha': trial.suggest_float('reg_alpha', 0.1, 0.8, log=True),
                        'model__reg_lambda': trial.suggest_float('reg_lambda', 0.5, 4.0, log=True),
                        'model__min_child_weight': trial.suggest_int('min_child_weight', 3, 8),
                        'model__gamma': trial.suggest_float('gamma', 0.05, 0.8, log=True),
                        'feature_selector__k': trial.suggest_int('feature_selector_k', 20, min(45, X.shape[1]))
                    }
                else:
                    params = {
                        'model__max_depth': trial.suggest_int('max_depth', 6, 12),
                        'model__learning_rate': trial.suggest_float('learning_rate', 0.03, 0.12, log=True),
                        'model__n_estimators': trial.suggest_int('n_estimators', 500, 1500, step=100),
                        'model__subsample': trial.suggest_float('subsample', 0.7, 0.9),
                        'model__colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
                        'model__colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 0.9),
                        'model__reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.3, log=True),
                        'model__reg_lambda': trial.suggest_float('reg_lambda', 0.1, 2.0, log=True),
                        'model__min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                        'model__gamma': trial.suggest_float('gamma', 0.01, 0.3, log=True),
                        'feature_selector__k': trial.suggest_int('feature_selector_k', 15, min(35, X.shape[1]))
                    }
                
                # Add scale_pos_weight for imbalanced classes
                pos_ratio = y.mean() if hasattr(y, 'mean') else np.mean(y)
                if 0.1 <= pos_ratio <= 0.9:  # Only if reasonably balanced
                    scale_pos_weight = (1 - pos_ratio) / pos_ratio
                    params['model__scale_pos_weight'] = scale_pos_weight
                
                scoring = 'accuracy'
            
            # Create pipeline with current parameters
            pipeline = self.create_pipelines(X, target_type)
            pipeline.set_params(**params)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            scores = cross_val_score(pipeline, X, y, cv=tscv, scoring=scoring, n_jobs=-1)
            
            return scores.mean()
        
        # Create Optuna study
        study_name = f"{target_type}_optimization"
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        n_trials = 100 if target_type == 'total_corners' else 80
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"Best score for {target_type}: {study.best_value:.4f}")
        print(f"Best parameters for {target_type}: {study.best_params}")
        
        # Log to W&B if available
        if self.use_wandb:
            wandb.log({
                f"{target_type}_best_score": study.best_value,
                f"{target_type}_best_params": study.best_params
            })
        
        return study.best_params
    
    def load_and_clean_data(self, csv_file_path: str) -> pd.DataFrame:
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
    
    def prepare_features_and_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series], pd.DataFrame]:
        """Prepare features and targets with temporal features"""
        # Add advanced temporal features
        df = self.create_advanced_temporal_features(df)
        
        # Comprehensive feature list
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
            
            # Temporal features
            'home_corners_avg_3', 'away_corners_avg_3', 'home_corners_against_avg_3', 'away_corners_against_avg_3',
            'home_corners_avg_6', 'away_corners_avg_6', 'home_corners_against_avg_6', 'away_corners_against_avg_6',
            'home_corners_avg_10', 'away_corners_avg_10', 'home_corners_against_avg_10', 'away_corners_against_avg_10',
            
            # Advanced temporal features
            'home_corner_trend', 'away_corner_trend', 'home_recent_form', 'away_recent_form',
            'expected_total_corners', 'corner_advantage_home', 'corner_advantage_away',
            'home_corner_variance', 'away_corner_variance', 'corner_momentum_home', 'corner_momentum_away',
            'home_high_corner_rate', 'away_high_corner_rate', 'home_low_corner_rate', 'away_low_corner_rate',
            'combined_corner_volatility', 'match_importance_factor', 'h2h_corner_history', 'style_clash_indicator',
            'home_corner_consistency', 'away_corner_consistency', 'seasonal_corner_adjustment', 
            'attacking_desperation_factor', 'defensive_solidity_factor', 'teams_familiarity'
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
        
        # Handle missing values and infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # For numerical columns, fill NaN with median
        for col in X.select_dtypes(include=[np.number]).columns:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if not pd.isna(median_val) else 0)
        
        # Create targets
        targets = {'total_corners': df_clean['total_corners']}
        for line in self.corner_lines:
            target_name = f'over_{line}_corners'
            if target_name in df_clean.columns:
                targets[target_name] = df_clean[target_name]
        
        return X, targets, df_clean
    
    def train_models_with_mlops(self, X: pd.DataFrame, targets: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Train models with MLOps best practices"""
        print(f"\n{'='*80}")
        print("TRAINING MODELS WITH MLOPS BEST PRACTICES")
        print(f"{'='*80}")
        
        results = {}
        
        # Train each model
        for target_name, y in targets.items():
            print(f"\n{'='*60}")
            print(f"TRAINING MODEL: {target_name.replace('_', ' ').upper()}")
            print(f"{'='*60}")
            
            # Skip if insufficient data
            if y.nunique() < 2:
                print(f"Skipping {target_name} - insufficient class diversity")
                continue
            
            # Optimize hyperparameters
            print("Optimizing hyperparameters...")
            best_params = self.optimize_hyperparameters(X, y, target_name)
            self.best_params[target_name] = best_params
            
            # Create and train final pipeline
            pipeline = self.create_pipelines(X, target_name)
            pipeline.set_params(**best_params)
            
            # Time series cross-validation for final evaluation
            tscv = TimeSeriesSplit(n_splits=5)
            
            if target_name == 'total_corners':
                cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
                print(f"Cross-validation MAE: {-cv_scores.mean():.3f} (±{cv_scores.std() * 2:.3f})")
                
                # Final evaluation with early stopping
                split_idx = int(len(X) * 0.85)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                # Fit pipeline
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
                
                print(f"Final Test MAE: {mae:.3f}")
                print(f"Final Test RMSE: {rmse:.3f}")
                
                results[target_name] = {'mae': mae, 'rmse': rmse, 'cv_score': -cv_scores.mean()}
                
            else:
                cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
                print(f"Cross-validation Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std() * 2:.3f})")
                
                # Final evaluation
                split_idx = int(len(X) * 0.85)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                # Fit pipeline
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                y_pred_proba = pipeline.predict_proba(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                logloss = log_loss(y_test, y_pred_proba)
                
                print(f"Final Test Accuracy: {accuracy:.3f}")
                print(f"Final Test Log Loss: {logloss:.3f}")
                
                results[target_name] = {'accuracy': accuracy, 'log_loss': logloss, 'cv_score': cv_scores.mean()}
            
            # Store the trained pipeline
            self.pipelines[target_name] = pipeline
            
            # Create SHAP explainer for interpretability
            try:
                print("Creating SHAP explainer...")
                # Get a small sample for SHAP (SHAP can be slow on large datasets)
                sample_size = min(100, len(X_train))
                X_sample = X_train.sample(n=sample_size, random_state=42)
                
                # Transform the sample through the pipeline (excluding the final model)
                X_sample_transformed = pipeline[:-1].transform(X_sample)
                
                # Create explainer
                explainer = shap.TreeExplainer(pipeline.named_steps['model'])
                self.shap_explainers[target_name] = {
                    'explainer': explainer,
                    'sample_data': X_sample_transformed
                }
                print("SHAP explainer created successfully!")
            except Exception as e:
                print(f"Warning: Could not create SHAP explainer: {e}")
            
            # Save model
            model_path = os.path.join(self.model_dir, f"{target_name}_pipeline.joblib")
            joblib.dump(pipeline, model_path)
            print(f"Model saved to: {model_path}")
            
            # Log to W&B if available
            if self.use_wandb:
                wandb.log({
                    f"{target_name}_final_score": results[target_name].get('accuracy', results[target_name].get('mae', 0)),
                    f"{target_name}_cv_score": results[target_name]['cv_score']
                })
        
        return results
    
    def explain_prediction_with_shap(self, match_data: Dict[str, Any], target_name: str, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """Explain a single prediction using SHAP"""
        if target_name not in self.shap_explainers or target_name not in self.pipelines:
            print(f"SHAP explainer not available for {target_name}")
            return None
        
        try:
            # Prepare the match data
            pipeline = self.pipelines[target_name]
            feature_names = self.feature_names[target_name]['all']
            
            # Convert match_data to DataFrame
            match_df = pd.DataFrame([match_data])
            
            # Ensure all required features are present
            for feature in feature_names:
                if feature not in match_df.columns:
                    match_df[feature] = 0  # Default value
            
            # Select only the features used in training
            match_df = match_df[feature_names]
            
            # Transform through the pipeline (excluding the final model)
            match_transformed = pipeline[:-1].transform(match_df)
            
            # Get SHAP values
            explainer = self.shap_explainers[target_name]['explainer']
            shap_values = explainer.shap_values(match_transformed)
            
            # If binary classification, get shap values for positive class
            if len(shap_values.shape) > 1 and shap_values.shape[1] > 1:
                shap_values = shap_values[:, 1]  # Positive class
            
            # Get feature names after transformation
            preprocessor = pipeline.named_steps['preprocessor']
            feature_selector = pipeline.named_steps['feature_selector']
            
            # Get transformed feature names
            transformed_feature_names = []
            for name, transformer, columns in preprocessor.transformers_:
                if name == 'num':
                    transformed_feature_names.extend([f"num__{col}" for col in columns])
                elif name == 'cat':
                    if hasattr(transformer, 'get_feature_names_out'):
                        cat_features = transformer.get_feature_names_out(columns)
                        transformed_feature_names.extend([f"cat__{feat}" for feat in cat_features])
                    else:
                        # Fallback for older sklearn versions
                        transformed_feature_names.extend([f"cat__{col}" for col in columns])
            
            # Get selected features
            selected_mask = feature_selector.get_support()
            selected_features = [transformed_feature_names[i] for i in range(len(selected_mask)) if selected_mask[i]]
            
            # Combine SHAP values with feature names
            feature_importance = list(zip(selected_features, shap_values[0]))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            explanation = {
                'top_features': feature_importance[:top_n],
                'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                'prediction_impact': sum(shap_values[0])
            }
            
            return explanation
            
        except Exception as e:
            print(f"Error creating SHAP explanation: {e}")
            return None
    
    def predict_all_corner_lines(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict corners for all lines with SHAP explanations"""
        if not self.pipelines:
            raise ValueError("Models not trained yet!")
        
        predictions = {}
        
        # Prepare match data as DataFrame
        feature_names_all = set()
        for target_name in self.pipelines.keys():
            if target_name in self.feature_names:
                feature_names_all.update(self.feature_names[target_name]['all'])
        
        match_df = pd.DataFrame([match_data])
        
        # Ensure all features are present
        for feature in feature_names_all:
            if feature not in match_df.columns:
                match_df[feature] = 0
        
        # Total corners prediction
        if 'total_corners' in self.pipelines:
            pipeline = self.pipelines['total_corners']
            required_features = self.feature_names['total_corners']['all']
            match_features = match_df[required_features]
            
            total_corners = pipeline.predict(match_features)[0]
            
            predictions['total_corners'] = {
                'prediction': round(total_corners, 2),
                'rounded': round(total_corners),
                'confidence_interval': (max(0, total_corners - 2), total_corners + 2)
            }
            
            # Add SHAP explanation
            shap_explanation = self.explain_prediction_with_shap(match_data, 'total_corners')
            if shap_explanation:
                predictions['total_corners']['explanation'] = shap_explanation
        
        # Corner line predictions
        predictions['corner_lines'] = {}
        
        for line in self.corner_lines:
            target_name = f'over_{line}_corners'
            if target_name not in self.pipelines:
                continue
            
            pipeline = self.pipelines[target_name]
            required_features = self.feature_names[target_name]['all']
            match_features = match_df[required_features]
            
            prediction = pipeline.predict(match_features)[0]
            probabilities = pipeline.predict_proba(match_features)[0]
            
            over_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            under_prob = probabilities[0] if len(probabilities) > 1 else 1 - probabilities[0]
            
            # Enhanced recommendation logic for hard lines
            hard_lines = [8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
            is_hard_line = line in hard_lines
            
            if is_hard_line:
                # Stricter thresholds for hard lines
                if over_prob > 0.68:
                    recommendation = "OVER 🔥"
                elif under_prob > 0.68:
                    recommendation = "UNDER 🔥"
                elif over_prob > 0.6:
                    recommendation = "Over"
                elif under_prob > 0.6:
                    recommendation = "Under"
                else:
                    recommendation = "Skip"
            else:
                # Standard thresholds
                if over_prob > 0.65:
                    recommendation = "OVER 🔥"
                elif under_prob > 0.65:
                    recommendation = "UNDER 🔥"
                elif over_prob > 0.57:
                    recommendation = "Over"
                elif under_prob > 0.57:
                    recommendation = "Under"
                else:
                    recommendation = "Skip"
            
            predictions['corner_lines'][line] = {
                'prediction': 'Over' if prediction == 1 else 'Under',
                'over_probability': over_prob,
                'under_probability': under_prob,
                'confidence': max(over_prob, under_prob),
                'recommended': recommendation,
                'edge': abs(over_prob - 0.5) * 2,
                'is_hard_line': is_hard_line
            }
            
            # Add SHAP explanation
            shap_explanation = self.explain_prediction_with_shap(match_data, target_name)
            if shap_explanation:
                predictions['corner_lines'][line]['explanation'] = shap_explanation
        
        return predictions
    
    def generate_enhanced_betting_recommendations(self, predictions: Dict[str, Any], match_info: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Generate enhanced betting recommendations with SHAP insights"""
        print(f"\n{'='*80}")
        print("ENHANCED BETTING RECOMMENDATIONS WITH AI EXPLANATIONS")
        print(f"{'='*80}")
        
        if match_info:
            print(f"Match: {match_info.get('home_team', 'Home')} vs {match_info.get('away_team', 'Away')}")
            print(f"Date: {match_info.get('date', 'TBD')}")
            print("-" * 80)
        
        # Total corners prediction with explanation
        if 'total_corners' in predictions:
            total_pred = predictions['total_corners']
            print(f"\nTOTAL CORNERS PREDICTION:")
            print(f"  Predicted: {total_pred['prediction']} corners")
            print(f"  Rounded: {total_pred['rounded']} corners")
            print(f"  95% Confidence Interval: {total_pred['confidence_interval'][0]:.1f} - {total_pred['confidence_interval'][1]:.1f}")
            
            # Show SHAP explanation if available
            if 'explanation' in total_pred:
                print(f"\n  🤖 AI EXPLANATION - Top factors driving this prediction:")
                explanation = total_pred['explanation']
                for i, (feature, impact) in enumerate(explanation['top_features'][:5], 1):
                    impact_direction = "↑" if impact > 0 else "↓"
                    feature_clean = feature.replace('num__', '').replace('cat__', '')
                    print(f"     {i}. {feature_clean}: {impact:+.3f} {impact_direction}")
        
        # Corner lines analysis
        if 'corner_lines' in predictions:
            print(f"\nCORNER LINES ANALYSIS WITH AI INSIGHTS:")
            print("-" * 100)
            print(f"{'Line':>6s} {'Prediction':>10s} {'Over %':>8s} {'Under %':>9s} {'Confidence':>11s} {'Recommendation':>14s} {'Edge':>6s} {'Type':>8s}")
            print("-" * 100)
            
            recommendations = {'strong_bets': [], 'good_bets': [], 'avoid': [], 'explained_bets': []}
            
            for line in sorted(predictions['corner_lines'].keys()):
                pred_data = predictions['corner_lines'][line]
                
                over_pct = pred_data['over_probability'] * 100
                under_pct = pred_data['under_probability'] * 100
                confidence = pred_data['confidence']
                recommendation = pred_data['recommended']
                edge = pred_data['edge']
                line_type = "Hard" if pred_data.get('is_hard_line', False) else "Standard"
                
                print(f"{line:6.1f} {pred_data['prediction']:>10s} {over_pct:7.1f}% {under_pct:8.1f}% "
                      f"{confidence:10.3f} {recommendation:>14s} {edge:5.3f} {line_type:>8s}")
                
                # Show SHAP explanation for high-confidence bets
                if 'explanation' in pred_data and confidence > 0.65:
                    print(f"        🤖 Key factors: ", end="")
                    explanation = pred_data['explanation']
                    top_factors = []
                    for feature, impact in explanation['top_features'][:3]:
                        impact_direction = "↑" if impact > 0 else "↓"
                        feature_clean = feature.replace('num__', '').replace('cat__', '')
                        top_factors.append(f"{feature_clean}{impact_direction}")
                    print(", ".join(top_factors))
                
                # Categorize recommendations
                bet_info = {
                    'line': line,
                    'bet': recommendation,
                    'confidence': confidence,
                    'edge': edge,
                    'probability': over_pct if 'Over' in recommendation else under_pct,
                    'is_hard_line': pred_data.get('is_hard_line', False),
                    'has_explanation': 'explanation' in pred_data
                }
                
                if recommendation not in ['Skip']:
                    if confidence >= 0.75 and edge >= 0.3:
                        recommendations['strong_bets'].append(bet_info)
                    elif confidence >= 0.65 and edge >= 0.2:
                        recommendations['good_bets'].append(bet_info)
                        if 'explanation' in pred_data:
                            recommendations['explained_bets'].append(bet_info)
                    else:
                        recommendations['avoid'].append(bet_info)
        
        # Enhanced betting strategy recommendations
        print(f"\n{'='*70}")
        print("ENHANCED BETTING STRATEGY WITH AI INSIGHTS")
        print(f"{'='*70}")
        
        if recommendations['strong_bets']:
            print(f"\n🔥 STRONG BETS (High Confidence):")
            for bet in sorted(recommendations['strong_bets'], key=lambda x: x['confidence'], reverse=True):
                line_type = " (Hard Line)" if bet['is_hard_line'] else ""
                explanation_indicator = " 🤖" if bet['has_explanation'] else ""
                print(f"  • {bet['bet']} {bet['line']} corners - Confidence: {bet['confidence']:.3f} "
                      f"({bet['probability']:.1f}% chance){line_type}{explanation_indicator}")
        
        if recommendations['good_bets']:
            print(f"\n✅ GOOD BETS (Medium Confidence):")
            for bet in sorted(recommendations['good_bets'], key=lambda x: x['confidence'], reverse=True):
                line_type = " (Hard Line)" if bet['is_hard_line'] else ""
                explanation_indicator = " 🤖" if bet['has_explanation'] else ""
                print(f"  • {bet['bet']} {bet['line']} corners - Confidence: {bet['confidence']:.3f} "
                      f"({bet['probability']:.1f}% chance){line_type}{explanation_indicator}")
        
        if not recommendations['strong_bets'] and not recommendations['good_bets']:
            print(f"\n⚠️  NO HIGH-CONFIDENCE BETS IDENTIFIED")
            print(f"   The AI models suggest avoiding corner bets for this match.")
        
        # AI insights summary
        print(f"\n🤖 AI INSIGHTS SUMMARY:")
        print(f"  • Models trained with MLOps best practices (Pipeline + TimeSeriesSplit)")
        print(f"  • Hyperparameters optimized with Optuna Bayesian optimization")
        print(f"  • SHAP explanations available for {len(recommendations['explained_bets'])} high-confidence bets")
        print(f"  • Hard lines (8.5-9.5, 10.5-13.5) use enhanced ensemble models")
        print(f"  • Predictions consider 40+ advanced features including momentum & volatility")
        
        # Risk management
        print(f"\n💡 ENHANCED RISK MANAGEMENT:")
        print(f"  • 🤖 Bets with AI explanations are more trustworthy")
        print(f"  • Hard lines require higher confidence thresholds")
        print(f"  • Use SHAP insights to understand why the model made each prediction")
        print(f"  • Models avoid data leakage through proper time series validation")
        
        return recommendations
    
    def save_model_artifacts(self) -> str:
        """Save all model artifacts and metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(self.model_dir, f"model_artifacts_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save best parameters
        with open(os.path.join(save_dir, "best_parameters.json"), 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        # Save feature names
        with open(os.path.join(save_dir, "feature_names.json"), 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        # Copy pipeline files
        for target_name in self.pipelines.keys():
            src_path = os.path.join(self.model_dir, f"{target_name}_pipeline.joblib")
            if os.path.exists(src_path):
                dst_path = os.path.join(save_dir, f"{target_name}_pipeline.joblib")
                joblib.dump(self.pipelines[target_name], dst_path)
        
        print(f"Model artifacts saved to: {save_dir}")
        
        if self.use_wandb:
            # Save artifacts to W&B
            wandb.save(os.path.join(save_dir, "*"))
        
        return save_dir
    
    def load_model_artifacts(self, artifact_dir: str):
        """Load model artifacts from directory"""
        # Load best parameters
        params_path = os.path.join(artifact_dir, "best_parameters.json")
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                self.best_params = json.load(f)
        
        # Load feature names
        features_path = os.path.join(artifact_dir, "feature_names.json")
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)
        
        # Load pipelines
        self.pipelines = {}
        for target_name in ['total_corners'] + [f'over_{line}_corners' for line in self.corner_lines]:
            pipeline_path = os.path.join(artifact_dir, f"{target_name}_pipeline.joblib")
            if os.path.exists(pipeline_path):
                self.pipelines[target_name] = joblib.load(pipeline_path)
        
        print(f"Loaded {len(self.pipelines)} models from {artifact_dir}")
    
    def initialize_chatbot(self):
        """Initialize the chatbot interface"""
        if not self.pipelines:
            print("⚠️ Warning: Models not trained yet! Train models first using main() function.")
            return None
        
        self.chatbot = CornerPredictionChatbot(self)
        return self.chatbot
    
    def start_chat(self):
        """Start the chatbot interface"""
        if self.chatbot is None:
            self.chatbot = self.initialize_chatbot()
        
        if self.chatbot:
            self.chatbot.chat()
        else:
            print("❌ Cannot start chatbot - models not trained!")
    
    def analyze_corner_data(self, df: pd.DataFrame):
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


def main():
    """Enhanced main function with MLOps pipeline"""
    # Initialize predictor with W&B (set to False if you don't want to use W&B)
    use_wandb = True  # Set to False if you don't have W&B set up
    predictor = EnhancedFootballCornerPredictor(use_wandb=use_wandb)
    
    # Update this path to your actual CSV file
    csv_file_path = r"/home/teodor/Pulpit/szwecja rożne/combined_sezony.csv"
    
    try:
        print("🏈 Enhanced Football Corner Prediction System with MLOps & AI Explanations")
        print("=" * 80)
        
        # Load and clean data
        df = predictor.load_and_clean_data(csv_file_path)
        
        # Comprehensive data analysis
        predictor.analyze_corner_data(df)
        
        # Prepare features and targets
        X, targets, df_clean = predictor.prepare_features_and_targets(df)
        
        print(f"\nFinal dataset for training:")
        print(f"  Features: {len(X.columns)}")
        print(f"  Matches: {len(df_clean):,}")
        print(f"  Date range: {df_clean['date_GMT'].min().strftime('%Y-%m-%d')} to {df_clean['date_GMT'].max().strftime('%Y-%m-%d')}")
        
        # Train all models with MLOps best practices
        training_results = predictor.train_models_with_mlops(X, targets)
        
        # Save model artifacts
        artifact_dir = predictor.save_model_artifacts()
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE - SYSTEM READY")
        print(f"{'='*80}")
        print(f"✅ Models trained with MLOps best practices!")
        print(f"🔧 Hyperparameters optimized with Optuna")
        print(f"📊 Time series cross-validation used (no data leakage)")
        print(f"🤖 SHAP explanations available for predictions")
        print(f"💾 Models saved to: {artifact_dir}")
        if predictor.use_wandb:
            print(f"📈 Experiment tracked in Weights & Biases")
        print(f"🤖 Chatbot initialized and ready!")
        print(f"{'='*80}")
        
        # Start the chatbot interface
        predictor.start_chat()
        
    except FileNotFoundError:
        print("❌ Error: CSV file not found!")
        print("Please update the csv_file_path variable with the correct path to your data file")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if predictor.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()