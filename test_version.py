import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, log_loss, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
import re
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import optuna
from scipy import stats
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
            
            match_data['home_ppg'] = float(input("Home PPG [1.2]: ") or "1.2")
            match_data['away_ppg'] = float(input("Away PPG [1.1]: ") or "1.1")
            match_data['home_xg'] = float(input("Home xG [1.4]: ") or "1.4")
            match_data['away_xg'] = float(input("Away xG [1.3]: ") or "1.3")
            
            print("\n💰 Betting odds (press Enter for defaults):")
            match_data['home_odds'] = float(input("Home win odds [2.0]: ") or "2.0")
            match_data['draw_odds'] = float(input("Draw odds [3.5]: ") or "3.5")
            match_data['away_odds'] = float(input("Away win odds [3.0]: ") or "3.0")
            
            print("\n📈 Historical data (press Enter for defaults):")
            match_data['avg_corners_per_match'] = float(input("Avg corners per match [9.5]: ") or "9.5")
            match_data['btts_pct'] = float(input("BTTS percentage [55]: ") or "55")
            match_data['over25_pct'] = float(input("Over 2.5 goals % [65]: ") or "65")
            
            # Recent form (simplified)
            match_data['home_corners_avg_recent'] = float(input("Home avg corners (recent) [5.0]: ") or "5.0")
            match_data['away_corners_avg_recent'] = float(input("Away avg corners (recent) [4.5]: ") or "4.5")
            match_data['home_corners_against_recent'] = float(input("Home corners against (recent) [4.2]: ") or "4.2")
            match_data['away_corners_against_recent'] = float(input("Away corners against (recent) [4.8]: ") or "4.8")
            match_data['home_form'] = float(input("Home recent form [0.67]: ") or "0.67")
            match_data['away_form'] = float(input("Away recent form [0.33]: ") or "0.33")
            
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
                    rec = data.get('recommended', 'N/A')
                    print(f"{line:6.1f} {prob:11.1f}% {prob:11.1f}%        ~ {rec:>15s}")
        
        print(f"\n💡 Chatbot Impact Summary:")
        if self.adjustment_factors:
            print("✅ Adjustments have been applied based on your input")
            print("🎯 Recommendations updated with contextual factors")
        else:
            print("ℹ️  No adjustments applied - using base model predictions")
        
        print(f"\n🎲 Ready for more adjustments or new predictions!")


class EnhancedFootballCornerPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(k=30)
        self.models = {}
        self.selected_features = {}
        self.feature_importance = {}
        self.team_stats_cache = {}
        self.corner_lines = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]
        self.chatbot = None
        self.preprocessor = None
        self.categorical_features = []
        self.numerical_features = []
        
        # Study storage for hyperparameter optimization
        self.study_storage = {}
        
    def initialize_chatbot(self):
        """Initialize the chatbot interface"""
        if not self.models:
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
    
    def safe_divide(self, numerator, denominator, default_value=0):
        """Safe division with proper handling"""
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
        result = np.clip(result, -10, 10)  # More reasonable bounds
        
        return result.item() if result.ndim == 0 else result
    
    def clean_data(self, df):
        """Enhanced data cleaning with red card and walkover removal"""
        print("Cleaning data...")
        
        # Store original count
        initial_rows = len(df)
        print(f"Original dataset: {initial_rows:,} matches")
        
        # Remove rows with missing essential data
        essential_cols = ['home_team_goal_count', 'away_team_goal_count', 
                        'home_team_name', 'away_team_name', 'home_team_corner_count', 
                        'away_team_corner_count', 'date_GMT']
        
        df = df.dropna(subset=essential_cols)
        print(f"After removing missing essential data: {len(df):,} matches (-{initial_rows - len(df):,})")
        
        # Remove matches with red cards (to reduce noise)
        red_card_cols = ['home_team_red_cards', 'away_team_red_cards']
        matches_before_red_cards = len(df)
        
        for col in red_card_cols:
            if col in df.columns:
                # Convert to numeric, treating any non-numeric as 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        if 'home_team_red_cards' in df.columns and 'away_team_red_cards' in df.columns:
            # Remove matches where any team had red cards
            df = df[(df['home_team_red_cards'] == 0) & (df['away_team_red_cards'] == 0)]
            red_cards_removed = matches_before_red_cards - len(df)
            print(f"After removing matches with red cards: {len(df):,} matches (-{red_cards_removed:,})")
        

        
        # Remove matches with extremely unusual scorelines (likely data errors or walkovers)
        matches_before_score_filter = len(df)
        
        # Remove matches with total goals > 10 (likely data errors)
        df = df[df['home_team_goal_count'] + df['away_team_goal_count'] <= 15]
        
        # Remove matches with individual team scoring > 8 goals (very rare, likely errors)
        df = df[(df['home_team_goal_count'] <= 10) & (df['away_team_goal_count'] <= 10)]
        
        unusual_scores_removed = matches_before_score_filter - len(df)
        if unusual_scores_removed > 0:
            print(f"After removing unusual scorelines: {len(df):,} matches (-{unusual_scores_removed:,})")
        
        # Handle -1 values and other missing indicators
        df = df.replace([-1, -999, 999, np.inf, -np.inf], np.nan)
        
        # Remove duplicate matches (same teams on same date)
        matches_before_duplicates = len(df)
        df = df.drop_duplicates(subset=['date_GMT', 'home_team_name', 'away_team_name'])
        duplicates_removed = matches_before_duplicates - len(df)
        if duplicates_removed > 0:
            print(f"After removing duplicates: {len(df):,} matches (-{duplicates_removed:,})")
        
        # Remove matches with unrealistic corner counts (likely data errors)
        matches_before_corners = len(df)
        df = df[(df['home_team_corner_count'] >= 0) & (df['home_team_corner_count'] <= 36)]
        df = df[(df['away_team_corner_count'] >= 0) & (df['away_team_corner_count'] <= 36)]
        corner_outliers_removed = matches_before_corners - len(df)
        if corner_outliers_removed > 0:
            print(f"After removing corner outliers: {len(df):,} matches (-{corner_outliers_removed:,})")
        
        # Additional quality filters
        matches_before_quality = len(df)
        
        # Remove matches with missing possession data if it's all zeros (likely incomplete data)
        if 'home_team_possession' in df.columns and 'away_team_possession' in df.columns:
            # Remove matches where both possessions are 0 or sum doesn't make sense
            df = df[~((df['home_team_possession'] == 0) & (df['away_team_possession'] == 0))]
            
            # Remove matches where possession sum is way off from 100% (data quality issue)
            possession_sum = df['home_team_possession'] + df['away_team_possession']
            df = df[(possession_sum >= 85) & (possession_sum <= 120)]  # Allow some tolerance
        
        # Remove matches with no shots data (likely incomplete)
        if 'home_team_shots' in df.columns and 'away_team_shots' in df.columns:
            df = df[~((df['home_team_shots'] == 0) & (df['away_team_shots'] == 0))]
        
        quality_removed = matches_before_quality - len(df)
        if quality_removed > 0:
            print(f"After additional quality filters: {len(df):,} matches (-{quality_removed:,})")
        
        # Final summary
        total_removed = initial_rows - len(df)
        removal_pct = (total_removed / initial_rows) * 100
        
        print(f"\n📊 Data Cleaning Summary:")
        print(f"  Original matches: {initial_rows:,}")
        print(f"  Final clean matches: {len(df):,}")
        print(f"  Total removed: {total_removed:,} ({removal_pct:.1f}%)")
        
        # Show breakdown of what was removed
        print(f"\n🔍 Removal Breakdown:")
        if red_cards_removed > 0:
            print(f"  🟥 Red card matches: {red_cards_removed:,}")
        if unusual_scores_removed > 0:
            print(f"  📈 Unusual scores: {unusual_scores_removed:,}")
        if duplicates_removed > 0:
            print(f"  🔄 Duplicates: {duplicates_removed:,}")
        if corner_outliers_removed > 0:
            print(f"  ⚽ Corner outliers: {corner_outliers_removed:,}")
        if quality_removed > 0:
            print(f"  🎯 Quality issues: {quality_removed:,}")
        
        return df
    
    def engineer_enhanced_features(self, df):
        """Create enhanced features with better predictive power"""
        print("Engineering enhanced features...")
        
        # Core target variables
        df['total_corners'] = df['home_team_corner_count'] + df['away_team_corner_count']
        df['corner_difference'] = df['home_team_corner_count'] - df['away_team_corner_count']
        df['home_corner_ratio'] = self.safe_divide(df['home_team_corner_count'], df['total_corners'], 0.5)
        
        # Create all corner line targets
        for line in self.corner_lines:
            df[f'over_{line}_corners'] = (df['total_corners'] > line).astype(int)
        
        # Enhanced pre-match features with proper column handling
        pre_match_cols = {
            'Pre-Match PPG (Home)': 'home_ppg',
            'Pre-Match PPG (Away)': 'away_ppg',
            'Home Team Pre-Match xG': 'home_xg', 
            'Away Team Pre-Match xG': 'away_xg'
        }
        
        for old_col, new_col in pre_match_cols.items():
            if old_col in df.columns:
                df[new_col] = pd.to_numeric(df[old_col], errors='coerce')
            else:
                # Create default values if column doesn't exist
                df[new_col] = 1.0
        
        # Team strength features
        df['ppg_difference'] = df['home_ppg'] - df['away_ppg']
        df['ppg_total'] = df['home_ppg'] + df['away_ppg']
        df['xg_difference'] = df['home_xg'] - df['away_xg']
        df['xg_total'] = df['home_xg'] + df['away_xg']
        df['strength_ratio'] = self.safe_divide(df['home_ppg'], df['away_ppg'], 1.0)
        df['attacking_ratio'] = self.safe_divide(df['home_xg'], df['away_xg'], 1.0)
        
        # Enhanced betting odds features
        odds_cols = {
            'odds_ft_home_team_win': 'home_odds',
            'odds_ft_draw': 'draw_odds', 
            'odds_ft_away_team_win': 'away_odds'
        }
        
        for old_col, new_col in odds_cols.items():
            if old_col in df.columns:
                df[new_col] = pd.to_numeric(df[old_col], errors='coerce')
                df[new_col] = df[new_col].fillna(df[new_col].median() if df[new_col].notna().any() else 2.5)
                df[new_col] = np.clip(df[new_col], 1.01, 50)  # Reasonable bounds
            else:
                # Create default odds
                if new_col == 'home_odds':
                    df[new_col] = 2.0
                elif new_col == 'draw_odds':
                    df[new_col] = 3.5
                else:  # away_odds
                    df[new_col] = 3.0
        
        # Market analysis features
        df['implied_home_prob'] = self.safe_divide(1, df['home_odds'], 0.4)
        df['implied_draw_prob'] = self.safe_divide(1, df['draw_odds'], 0.3)
        df['implied_away_prob'] = self.safe_divide(1, df['away_odds'], 0.3)
        
        # Market efficiency
        total_implied = df['implied_home_prob'] + df['implied_draw_prob'] + df['implied_away_prob']
        df['market_margin'] = total_implied - 1
        df['match_competitiveness'] = 1 - np.abs(df['implied_home_prob'] - df['implied_away_prob'])
        df['odds_variance'] = np.var([df['home_odds'], df['draw_odds'], df['away_odds']], axis=0)
        
        # Historical features
        hist_cols = {
            'average_corners_per_match_pre_match': 'avg_corners_historical',
            'btts_percentage_pre_match': 'btts_pct',
            'over_25_percentage_pre_match': 'over25_pct'
        }
        
        for old_col, new_col in hist_cols.items():
            if old_col in df.columns:
                df[new_col] = pd.to_numeric(df[old_col], errors='coerce')
                if new_col in ['btts_pct', 'over25_pct']:
                    df[new_col] = df[new_col].fillna(50) / 100  # Convert to 0-1
            else:
                if new_col == 'avg_corners_historical':
                    df[new_col] = 10.0  # Default average
                elif new_col == 'btts_pct':
                    df[new_col] = 0.55
                else:  # over25_pct
                    df[new_col] = 0.60
        
        # Goal-related features
        df['total_goals'] = df['home_team_goal_count'] + df['away_team_goal_count']
        df['goal_difference'] = df['home_team_goal_count'] - df['away_team_goal_count']
        df['high_scoring'] = (df['total_goals'] > 2.5).astype(int)
        df['both_teams_scored'] = ((df['home_team_goal_count'] > 0) & (df['away_team_goal_count'] > 0)).astype(int)
        
        # Match outcome features
        df['home_win'] = (df['home_team_goal_count'] > df['away_team_goal_count']).astype(int)
        df['draw'] = (df['home_team_goal_count'] == df['away_team_goal_count']).astype(int)
        df['away_win'] = (df['home_team_goal_count'] < df['away_team_goal_count']).astype(int)
        
        # Temporal features
        df['date_GMT'] = pd.to_datetime(df['date_GMT'])
        df['month'] = df['date_GMT'].dt.month
        df['day_of_week'] = df['date_GMT'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
        
        # Season period (assuming European football calendar)
        df['season_period'] = 'mid_season'  # Default
        df.loc[df['month'].isin([8, 9]), 'season_period'] = 'early_season'
        df.loc[df['month'].isin([5, 6, 7]), 'season_period'] = 'late_season'
        df.loc[df['month'].isin([12, 1]), 'season_period'] = 'winter'
        
        return df
    
    def create_advanced_temporal_features_no_leakage(self, df):
        """Create temporal features without data leakage using proper time series approach"""
        print("Creating temporal features without data leakage...")
        
        df = df.copy()
        df = df.sort_values('date_GMT').reset_index(drop=True)
        
        # Initialize feature columns
        temporal_features = [
            'home_corners_avg_3', 'home_corners_avg_6', 'home_corners_avg_10',
            'away_corners_avg_3', 'away_corners_avg_6', 'away_corners_avg_10',
            'home_corners_against_avg_3', 'home_corners_against_avg_6', 'home_corners_against_avg_10',
            'away_corners_against_avg_3', 'away_corners_against_avg_6', 'away_corners_against_avg_10',
            'home_goals_avg_3', 'home_goals_avg_6', 'home_goals_avg_10',
            'away_goals_avg_3', 'away_goals_avg_6', 'away_goals_avg_10',
            'home_goals_against_avg_3', 'home_goals_against_avg_6', 'home_goals_against_avg_10',
            'away_goals_against_avg_3', 'away_goals_against_avg_6', 'away_goals_against_avg_10',
            'home_corner_trend', 'away_corner_trend',
            'home_form_3', 'away_form_3', 'home_form_6', 'away_form_6',
            'home_corner_variance_6', 'away_corner_variance_6',
            'home_high_corner_rate_8', 'away_high_corner_rate_8',
            'home_low_corner_rate_8', 'away_low_corner_rate_8',
            'home_attacking_tendency', 'away_attacking_tendency',
            'home_defensive_solidity', 'away_defensive_solidity',
            'h2h_avg_corners', 'h2h_matches_count',
            'home_rest_days', 'away_rest_days',
            'expected_corners_base'
        ]
        
        for col in temporal_features:
            df[col] = 0.0
        
        # Team statistics tracking
        team_stats = {}
        h2h_stats = {}
        
        # Initialize team stats
        for team in pd.concat([df['home_team_name'], df['away_team_name']]).unique():
            team_stats[team] = {
                'corners_for': [],
                'corners_against': [],
                'goals_for': [],
                'goals_against': [],
                'match_dates': [],
                'match_results': []  # 1 for win, 0.5 for draw, 0 for loss
            }
        
        # Process each match chronologically
        for idx, row in df.iterrows():
            home_team = row['home_team_name']
            away_team = row['away_team_name']
            match_date = row['date_GMT']
            
            home_stats = team_stats[home_team]
            away_stats = team_stats[away_team]
            
            # Calculate features using historical data (before this match)
            
            # 1. Corner averages for different windows
            for window in [3, 6, 10]:
                if len(home_stats['corners_for']) >= window:
                    df.loc[idx, f'home_corners_avg_{window}'] = np.mean(home_stats['corners_for'][-window:])
                    df.loc[idx, f'home_corners_against_avg_{window}'] = np.mean(home_stats['corners_against'][-window:])
                
                if len(away_stats['corners_for']) >= window:
                    df.loc[idx, f'away_corners_avg_{window}'] = np.mean(away_stats['corners_for'][-window:])
                    df.loc[idx, f'away_corners_against_avg_{window}'] = np.mean(away_stats['corners_against'][-window:])
            
            # 2. Goal averages for different windows
            for window in [3, 6, 10]:
                if len(home_stats['goals_for']) >= window:
                    df.loc[idx, f'home_goals_avg_{window}'] = np.mean(home_stats['goals_for'][-window:])
                    df.loc[idx, f'home_goals_against_avg_{window}'] = np.mean(home_stats['goals_against'][-window:])
                
                if len(away_stats['goals_for']) >= window:
                    df.loc[idx, f'away_goals_avg_{window}'] = np.mean(away_stats['goals_for'][-window:])
                    df.loc[idx, f'away_goals_against_avg_{window}'] = np.mean(away_stats['goals_against'][-window:])
            
            # 3. Corner trends (recent vs older performance)
            if len(home_stats['corners_for']) >= 6:
                recent_home = np.mean(home_stats['corners_for'][-3:])
                older_home = np.mean(home_stats['corners_for'][-6:-3])
                df.loc[idx, 'home_corner_trend'] = recent_home - older_home
            
            if len(away_stats['corners_for']) >= 6:
                recent_away = np.mean(away_stats['corners_for'][-3:])
                older_away = np.mean(away_stats['corners_for'][-6:-3])
                df.loc[idx, 'away_corner_trend'] = recent_away - older_away
            
            # 4. Team form (win rate)
            for window in [3, 6]:
                if len(home_stats['match_results']) >= window:
                    df.loc[idx, f'home_form_{window}'] = np.mean(home_stats['match_results'][-window:])
                
                if len(away_stats['match_results']) >= window:
                    df.loc[idx, f'away_form_{window}'] = np.mean(away_stats['match_results'][-window:])
            
            # 5. Corner variance (predictability)
            if len(home_stats['corners_for']) >= 6:
                df.loc[idx, 'home_corner_variance_6'] = np.var(home_stats['corners_for'][-6:])
            
            if len(away_stats['corners_for']) >= 6:
                df.loc[idx, 'away_corner_variance_6'] = np.var(away_stats['corners_for'][-6:])
            
            # 6. High/Low corner game rates
            if len(home_stats['corners_for']) >= 8:
                home_total_corners = [(home_stats['corners_for'][i] + home_stats['corners_against'][i]) 
                                    for i in range(len(home_stats['corners_for']))]
                high_corner_games = sum(1 for total in home_total_corners[-8:] if total >= 12)
                low_corner_games = sum(1 for total in home_total_corners[-8:] if total <= 8)
                df.loc[idx, 'home_high_corner_rate_8'] = high_corner_games / 8
                df.loc[idx, 'home_low_corner_rate_8'] = low_corner_games / 8
            
            if len(away_stats['corners_for']) >= 8:
                away_total_corners = [(away_stats['corners_for'][i] + away_stats['corners_against'][i]) 
                                    for i in range(len(away_stats['corners_for']))]
                high_corner_games = sum(1 for total in away_total_corners[-8:] if total >= 12)
                low_corner_games = sum(1 for total in away_total_corners[-8:] if total <= 8)
                df.loc[idx, 'away_high_corner_rate_8'] = high_corner_games / 8
                df.loc[idx, 'away_low_corner_rate_8'] = low_corner_games / 8
            
            # 7. Attacking tendency and defensive solidity
            if len(home_stats['corners_for']) >= 5:
                df.loc[idx, 'home_attacking_tendency'] = np.mean(home_stats['corners_for'][-5:])
                df.loc[idx, 'home_defensive_solidity'] = 1 / (1 + np.mean(home_stats['corners_against'][-5:]))
            
            if len(away_stats['corners_for']) >= 5:
                df.loc[idx, 'away_attacking_tendency'] = np.mean(away_stats['corners_for'][-5:])
                df.loc[idx, 'away_defensive_solidity'] = 1 / (1 + np.mean(away_stats['corners_against'][-5:]))
            
            # 8. Head-to-head statistics
            h2h_key = f"{home_team}_vs_{away_team}"
            h2h_reverse_key = f"{away_team}_vs_{home_team}"
            
            h2h_data = h2h_stats.get(h2h_key, []) + h2h_stats.get(h2h_reverse_key, [])
            if h2h_data:
                recent_h2h = h2h_data[-3:]  # Last 3 meetings
                df.loc[idx, 'h2h_avg_corners'] = np.mean([match['total_corners'] for match in recent_h2h])
                df.loc[idx, 'h2h_matches_count'] = len(h2h_data)
            
            # 9. Rest days (if we have match dates)
            if len(home_stats['match_dates']) > 0:
                last_home_match = home_stats['match_dates'][-1]
                home_rest = (match_date - last_home_match).days
                df.loc[idx, 'home_rest_days'] = min(home_rest, 30)  # Cap at 30 days
            
            if len(away_stats['match_dates']) > 0:
                last_away_match = away_stats['match_dates'][-1]
                away_rest = (match_date - last_away_match).days
                df.loc[idx, 'away_rest_days'] = min(away_rest, 30)
            
            # 10. Expected corners calculation
            home_att = df.loc[idx, 'home_corners_avg_6'] if df.loc[idx, 'home_corners_avg_6'] > 0 else 5.0
            home_def = df.loc[idx, 'home_corners_against_avg_6'] if df.loc[idx, 'home_corners_against_avg_6'] > 0 else 4.5
            away_att = df.loc[idx, 'away_corners_avg_6'] if df.loc[idx, 'away_corners_avg_6'] > 0 else 4.5
            away_def = df.loc[idx, 'away_corners_against_avg_6'] if df.loc[idx, 'away_corners_against_avg_6'] > 0 else 5.0
            
            expected_corners = (home_att + away_def + away_att + home_def) / 2
            df.loc[idx, 'expected_corners_base'] = expected_corners
            
            # UPDATE STATS AFTER CALCULATING FEATURES (NO DATA LEAKAGE)
            
            home_corners = row['home_team_corner_count']
            away_corners = row['away_team_corner_count']
            home_goals = row['home_team_goal_count']
            away_goals = row['away_team_goal_count']
            total_corners = home_corners + away_corners
            
            # Determine match result for form calculation
            if home_goals > away_goals:
                home_result, away_result = 1, 0  # Home win
            elif home_goals < away_goals:
                home_result, away_result = 0, 1  # Away win
            else:
                home_result, away_result = 0.5, 0.5  # Draw
            
            # Update home team stats
            home_stats['corners_for'].append(home_corners)
            home_stats['corners_against'].append(away_corners)
            home_stats['goals_for'].append(home_goals)
            home_stats['goals_against'].append(away_goals)
            home_stats['match_dates'].append(match_date)
            home_stats['match_results'].append(home_result)
            
            # Update away team stats
            away_stats['corners_for'].append(away_corners)
            away_stats['corners_against'].append(home_corners)
            away_stats['goals_for'].append(away_goals)
            away_stats['goals_against'].append(home_goals)
            away_stats['match_dates'].append(match_date)
            away_stats['match_results'].append(away_result)
            
            # Update head-to-head stats
            if h2h_key not in h2h_stats:
                h2h_stats[h2h_key] = []
            
            h2h_stats[h2h_key].append({
                'total_corners': total_corners,
                'home_corners': home_corners,
                'away_corners': away_corners,
                'date': match_date
            })
            
            # Keep only recent h2h data (last 5 years)
            cutoff_date = match_date - pd.DateOffset(years=5)
            h2h_stats[h2h_key] = [match for match in h2h_stats[h2h_key] 
                                 if match['date'] >= cutoff_date]
        
        print(f"Created {len(temporal_features)} temporal features")
        return df
    
    def optimize_hyperparameters(self, X, y, model_type='regression', n_trials=100):
        """Auto-tune hyperparameters using Optuna"""
        
        def objective(trial):
            if model_type == 'regression':
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'mae',
                    'max_depth': trial.suggest_int('max_depth', 4, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 3.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                    'random_state': 42,
                    'n_jobs': -1,
                    'tree_method': 'hist'
                }
                
                model = xgb.XGBRegressor(**params)
                
                # Use TimeSeriesSplit for proper cross-validation
                tscv = TimeSeriesSplit(n_splits=5)
                scores = cross_val_score(model, X, y, cv=tscv, scoring='r2', n_jobs=1)
                
                return scores.mean()
                
            else:  # classification
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': trial.suggest_int('max_depth', 4, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 3.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
                    'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 3.0),
                    'random_state': 42,
                    'n_jobs': -1,
                    'tree_method': 'hist'
                }
                
                model = xgb.XGBClassifier(**params)
                
                tscv = TimeSeriesSplit(n_splits=5)
                scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=1)
                
                return scores.mean()
        
        # Create study with appropriate direction
        direction = 'maximize'
        study = optuna.create_study(direction=direction, 
                                  sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        print(f"Best {model_type} parameters found:")
        print(f"Best score: {study.best_value:.4f}")
        
        return study.best_params
    
    def create_preprocessing_pipeline(self, df):
        """Create preprocessing pipeline with one-hot encoding"""
        print("Creating preprocessing pipeline...")
        
        # Identify categorical and numerical features
        categorical_features = ['season_period', 'home_team_name', 'away_team_name']
        
        # Filter categorical features that exist in the data
        categorical_features = [col for col in categorical_features if col in df.columns]
        
        # Get numerical features (excluding target variables and other non-features)
        exclude_cols = ['total_corners', 'corner_difference', 'home_corner_ratio', 
                       'date_GMT', 'home_win', 'draw', 'away_win'] + \
                      [f'over_{line}_corners' for line in self.corner_lines]
        
        numerical_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                            if col not in exclude_cols and not col.endswith('_count')]
        
        # Limit team encoding to avoid high cardinality issues
        if 'home_team_name' in categorical_features:
            team_counts = df['home_team_name'].value_counts()
            frequent_teams = team_counts[team_counts >= 10].index.tolist()[:50]  # Top 50 teams
            
            df['home_team_name'] = df['home_team_name'].apply(
                lambda x: x if x in frequent_teams else 'Other'
            )
            df['away_team_name'] = df['away_team_name'].apply(
                lambda x: x if x in frequent_teams else 'Other'
            )
        
        print(f"Categorical features: {len(categorical_features)}")
        print(f"Numerical features: {len(numerical_features)}")
        
        # Store for later use
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                 categorical_features)
            ],
            remainder='drop'
        )
        
        self.preprocessor = preprocessor
        
        return preprocessor
    
    def load_and_clean_data(self, csv_file_path):
        """Load and clean the football match data"""
        print("Loading and cleaning data...")
        
        df = pd.read_csv(csv_file_path)
        print(f"Original data shape: {df.shape}")
        
        # Convert date and sort chronologically
        df['date_GMT'] = pd.to_datetime(df['date_GMT'], errors='coerce')
        df = df.dropna(subset=['date_GMT'])
        df = df.sort_values('date_GMT').reset_index(drop=True)
        
        # Clean data
        df = self.clean_data(df)
        
        # Engineer enhanced features
        df = self.engineer_enhanced_features(df)
        
        print(f"Cleaned data shape: {df.shape}")
        return df
    
    def prepare_features_and_targets(self, df):
        """Prepare features and targets with proper temporal handling"""
        print("Preparing features and targets...")
        
        # Add temporal features
        df = self.create_advanced_temporal_features_no_leakage(df)
        
        # Remove early matches without sufficient historical data
        min_matches_for_prediction = 50
        df_modeling = df.iloc[min_matches_for_prediction:].copy()
        print(f"Using {len(df_modeling)} matches for modeling (after temporal feature creation)")
        
        # Create preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline(df_modeling)
        
        # Prepare targets
        targets = {'total_corners': df_modeling['total_corners']}
        for line in self.corner_lines:
            targets[f'over_{line}_corners'] = df_modeling[f'over_{line}_corners']
        
        return df_modeling, preprocessor, targets
    
    def train_models_with_auto_tuning(self, df_modeling, preprocessor, targets):
        """Train models with automatic hyperparameter tuning"""
        print(f"\n{'='*80}")
        print("TRAINING MODELS WITH AUTO-HYPERPARAMETER TUNING")
        print(f"{'='*80}")
        
        # Prepare feature matrix
        feature_columns = self.numerical_features + self.categorical_features
        X_raw = df_modeling[feature_columns].copy()
        
        # Handle missing values before preprocessing
        for col in self.numerical_features:
            if col in X_raw.columns:
                X_raw[col] = X_raw[col].fillna(X_raw[col].median())
        
        for col in self.categorical_features:
            if col in X_raw.columns:
                X_raw[col] = X_raw[col].fillna('Unknown')
        
        # Transform features
        X_processed = preprocessor.fit_transform(X_raw)
        
        results = {}
        
        # 1. Train total corners regression model
        print(f"\n{'='*60}")
        print("TRAINING TOTAL CORNERS REGRESSION MODEL")
        print(f"{'='*60}")
        
        y_regression = targets['total_corners']
        
        print("Auto-tuning hyperparameters for regression...")
        best_params_reg = self.optimize_hyperparameters(X_processed, y_regression, 
                                                       model_type='regression', n_trials=50)
        
        # Feature selection
        feature_selector_reg = SelectKBest(score_func=mutual_info_regression, k=min(40, X_processed.shape[1]))
        X_selected_reg = feature_selector_reg.fit_transform(X_processed, y_regression)
        
        print(f"Selected {X_selected_reg.shape[1]} features for regression")
        
        # Train final model with best parameters
        final_model_reg = xgb.XGBRegressor(**best_params_reg)
        
        # Cross-validation with TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores_reg = cross_val_score(final_model_reg, X_selected_reg, y_regression, 
                                       cv=tscv, scoring='r2', n_jobs=-1)
        
        print(f"Cross-validation R²: {cv_scores_reg.mean():.4f} (±{cv_scores_reg.std() * 2:.4f})")
        
        # Train on full data
        final_model_reg.fit(X_selected_reg, y_regression)
        
        # Store model and components
        self.models['total_corners'] = {
            'model': final_model_reg,
            'feature_selector': feature_selector_reg,
            'params': best_params_reg
        }
        
        # Final evaluation
        split_idx = int(len(X_selected_reg) * 0.8)
        X_train_reg = X_selected_reg[:split_idx]
        X_test_reg = X_selected_reg[split_idx:]
        y_train_reg = y_regression.iloc[:split_idx]
        y_test_reg = y_regression.iloc[split_idx:]
        
        test_model_reg = xgb.XGBRegressor(**best_params_reg)
        test_model_reg.fit(X_train_reg, y_train_reg)
        y_pred_reg = test_model_reg.predict(X_test_reg)
        
        mae_reg = mean_absolute_error(y_test_reg, y_pred_reg)
        r2_reg = r2_score(y_test_reg, y_pred_reg)
        rmse_reg = np.sqrt(np.mean((y_test_reg - y_pred_reg) ** 2))
        
        print(f"\nFinal Regression Performance:")
        print(f"R²: {r2_reg:.4f}")
        print(f"MAE: {mae_reg:.3f}")
        print(f"RMSE: {rmse_reg:.3f}")
        
        results['regression'] = {'r2': r2_reg, 'mae': mae_reg, 'rmse': rmse_reg}
        
        # 2. Train classification models for corner lines
        print(f"\n{'='*60}")
        print("TRAINING CORNER LINE CLASSIFICATION MODELS")
        print(f"{'='*60}")
        
        for line in self.corner_lines:
            target_name = f'over_{line}_corners'
            if target_name not in targets:
                continue
                
            print(f"\nTraining model for Over {line} corners...")
            
            y_clf = targets[target_name]
            
            # Skip if insufficient class diversity
            if y_clf.nunique() < 2:
                print(f"Skipping {target_name} - insufficient class diversity")
                continue
            
            print(f"Auto-tuning hyperparameters for Over {line}...")
            best_params_clf = self.optimize_hyperparameters(X_processed, y_clf, 
                                                           model_type='classification', 
                                                           n_trials=30)
            
            # Adjust scale_pos_weight based on class distribution
            pos_ratio = y_clf.mean()
            best_params_clf['scale_pos_weight'] = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1
            
            # Feature selection
            feature_selector_clf = SelectKBest(score_func=mutual_info_classif, 
                                             k=min(30, X_processed.shape[1]))
            X_selected_clf = feature_selector_clf.fit_transform(X_processed, y_clf)
            
            # Train final model
            final_model_clf = xgb.XGBClassifier(**best_params_clf)
            
            # Cross-validation
            cv_scores_clf = cross_val_score(final_model_clf, X_selected_clf, y_clf, 
                                           cv=tscv, scoring='accuracy', n_jobs=-1)
            
            print(f"Over {line} - CV Accuracy: {cv_scores_clf.mean():.3f} (±{cv_scores_clf.std() * 2:.3f})")
            
            # Train on full data
            final_model_clf.fit(X_selected_clf, y_clf)
            
            # Store model
            self.models[target_name] = {
                'model': final_model_clf,
                'feature_selector': feature_selector_clf,
                'params': best_params_clf
            }
            
            results[target_name] = {'accuracy': cv_scores_clf.mean()}
        
        # Store preprocessing pipeline
        self.preprocessor = preprocessor
        
        return results
    
    def predict_all_corner_lines(self, match_data):
        """Predict corners for all lines"""
        if not self.models:
            raise ValueError("Models not trained yet!")
        
        predictions = {}
        
        # Prepare features
        feature_vector = self.prepare_prediction_features(match_data)
        
        # Total corners prediction
        if 'total_corners' in self.models:
            model_data = self.models['total_corners']
            model = model_data['model']
            feature_selector = model_data['feature_selector']
            
            X_selected = feature_selector.transform(feature_vector.reshape(1, -1))
            total_corners = model.predict(X_selected)[0]
            
            predictions['total_corners'] = {
                'prediction': round(total_corners, 2),
                'rounded': round(total_corners),
                'confidence_interval': (max(0, total_corners - 2), total_corners + 2)
            }
        
        # Corner line predictions
        predictions['corner_lines'] = {}
        
        for line in self.corner_lines:
            target_name = f'over_{line}_corners'
            if target_name not in self.models:
                continue
            
            model_data = self.models[target_name]
            model = model_data['model']
            feature_selector = model_data['feature_selector']
            
            X_selected = feature_selector.transform(feature_vector.reshape(1, -1))
            
            prob = model.predict_proba(X_selected)[0]
            prediction = model.predict(X_selected)[0]
            
            over_prob = prob[1] if len(prob) > 1 else prob[0]
            under_prob = 1 - over_prob
            
            predictions['corner_lines'][line] = {
                'prediction': 'Over' if prediction == 1 else 'Under',
                'over_probability': over_prob,
                'under_probability': under_prob,
                'confidence': max(over_prob, under_prob),
                'recommended': 'Over' if over_prob > 0.6 else 'Under' if under_prob > 0.6 else 'Skip',
                'edge': abs(over_prob - 0.5) * 2
            }
        
        return predictions
    
    def prepare_prediction_features(self, match_data):
        """Prepare features for prediction from match data"""
        # Create a row with the match data
        feature_row = {}
        
        # Fill numerical features with provided data or defaults
        for feature in self.numerical_features:
            if feature in match_data:
                feature_row[feature] = match_data[feature]
            else:
                # Provide reasonable defaults
                defaults = {
                    'home_ppg': 1.2, 'away_ppg': 1.1,
                    'home_xg': 1.4, 'away_xg': 1.3,
                    'home_odds': 2.0, 'draw_odds': 3.5, 'away_odds': 3.0,
                    'avg_corners_historical': 10.0,
                    'btts_pct': 0.55, 'over25_pct': 0.60
                }
                feature_row[feature] = defaults.get(feature, 0.0)
        
        # Fill categorical features
        for feature in self.categorical_features:
            if feature in match_data:
                feature_row[feature] = match_data[feature]
            else:
                defaults = {
                    'season_period': 'mid_season',
                    'home_team_name': 'Unknown',
                    'away_team_name': 'Unknown'
                }
                feature_row[feature] = defaults.get(feature, 'Unknown')
        
        # Create DataFrame and transform
        feature_df = pd.DataFrame([feature_row])
        
        # Handle missing values
        for col in self.numerical_features:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna(0.0)
        
        for col in self.categorical_features:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna('Unknown')
        
        # Transform using preprocessor
        feature_vector = self.preprocessor.transform(feature_df)
        
        return feature_vector.flatten()
    
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
            print(f"  Standard Deviation: {corners.std():.2f}")
            print(f"  Range: {corners.min():.0f} - {corners.max():.0f}")
            
            print(f"\nCorner Distribution:")
            for i in range(int(corners.min()), int(corners.max()) + 1):
                count = (corners == i).sum()
                pct = count / len(corners) * 100
                if count > 0:
                    print(f"  {i:2d} corners: {count:4d} matches ({pct:5.1f}%)")
            
            print(f"\nOver/Under Analysis for Popular Lines:")
            for line in [7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]:
                over_pct = (corners > line).mean() * 100
                under_pct = 100 - over_pct
                print(f"  Over {line:4.1f}: {over_pct:5.1f}% | Under {line:4.1f}: {under_pct:5.1f}%")
    
    def display_feature_importance(self, top_n=15):
        """Display feature importance for all models"""
        print(f"\n{'='*80}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*80}")
        
        for model_name, model_data in self.models.items():
            if 'model' in model_data and hasattr(model_data['model'], 'feature_importances_'):
                print(f"\nTop {top_n} Features for {model_name.replace('_', ' ').title()}:")
                print("-" * 60)
                
                importances = model_data['model'].feature_importances_
                
                # Get feature names (this is simplified - in practice you'd need to track feature names through preprocessing)
                feature_names = [f"feature_{i}" for i in range(len(importances))]
                
                # Sort by importance
                sorted_idx = np.argsort(importances)[::-1]
                
                for i in range(min(top_n, len(importances))):
                    idx = sorted_idx[i]
                    importance = importances[idx]
                    feature_name = feature_names[idx]
                    
                    bar_length = int(importance * 50)
                    bar = "█" * bar_length + "░" * (50 - bar_length)
                    print(f"{i+1:2d}. {feature_name:30s} {importance:6.3f} |{bar}|")
    
    def generate_betting_recommendations(self, predictions, match_info=None):
        """Generate intelligent betting recommendations"""
        print(f"\n{'='*80}")
        print("INTELLIGENT BETTING RECOMMENDATIONS")
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
            print(f"  Confidence Range: {total_pred['confidence_interval'][0]:.1f} - {total_pred['confidence_interval'][1]:.1f}")
        
        # Corner lines analysis
        if 'corner_lines' in predictions:
            print(f"\nCORNER LINES ANALYSIS:")
            print("-" * 90)
            print(f"{'Line':>6s} {'Prediction':>10s} {'Over %':>8s} {'Under %':>9s} {'Confidence':>11s} {'Recommendation':>14s} {'Edge':>6s}")
            print("-" * 90)
            
            recommendations = {'strong_bets': [], 'good_bets': [], 'avoid': []}
            
            for line in sorted(predictions['corner_lines'].keys()):
                pred_data = predictions['corner_lines'][line]
                
                over_pct = pred_data['over_probability'] * 100
                under_pct = pred_data['under_probability'] * 100
                confidence = pred_data['confidence']
                recommendation = pred_data['recommended']
                edge = pred_data['edge']
                
                print(f"{line:6.1f} {pred_data['prediction']:>10s} {over_pct:7.1f}% {under_pct:8.1f}% "
                      f"{confidence:10.3f} {recommendation:>14s} {edge:5.3f}")
                
                bet_info = {
                    'line': line,
                    'bet': recommendation,
                    'confidence': confidence,
                    'edge': edge,
                    'probability': over_pct if recommendation == 'Over' else under_pct
                }
                
                if recommendation != 'Skip':
                    if confidence >= 0.75 and edge >= 0.3:
                        recommendations['strong_bets'].append(bet_info)
                    elif confidence >= 0.65 and edge >= 0.2:
                        recommendations['good_bets'].append(bet_info)
                    else:
                        recommendations['avoid'].append(bet_info)
        
        # Display recommendations
        print(f"\n{'='*70}")
        print("BETTING STRATEGY RECOMMENDATIONS")
        print(f"{'='*70}")
        
        if recommendations['strong_bets']:
            print(f"\n🔥 STRONG BETS (High Confidence):")
            for bet in sorted(recommendations['strong_bets'], key=lambda x: x['confidence'], reverse=True):
                print(f"  • {bet['bet']} {bet['line']} corners - Confidence: {bet['confidence']:.3f} "
                      f"({bet['probability']:.1f}% chance)")
        
        if recommendations['good_bets']:
            print(f"\n✅ GOOD BETS (Medium Confidence):")
            for bet in sorted(recommendations['good_bets'], key=lambda x: x['confidence'], reverse=True):
                print(f"  • {bet['bet']} {bet['line']} corners - Confidence: {bet['confidence']:.3f} "
                      f"({bet['probability']:.1f}% chance)")
        
        if not any([recommendations['strong_bets'], recommendations['good_bets']]):
            print(f"\n⚠️  NO HIGH-CONFIDENCE BETS IDENTIFIED")
            print(f"   Consider avoiding corner bets for this match.")
        
        print(f"\n💡 RISK MANAGEMENT TIPS:")
        print(f"  • Only bet when you have a clear edge over bookmaker odds")
        print(f"  • Use proper bankroll management (1-3% per bet)")
        print(f"  • Track your results to validate model performance")
        print(f"  • Consider market timing and line shopping")
        
        return recommendations


def main():
    """Enhanced main function with improved corner prediction"""
    predictor = EnhancedFootballCornerPredictor()
    
    # Update this path to your actual CSV file
    csv_file_path = r"/home/teodor/Pulpit/szwecja rożne/combined_sezony.csv"
    
    try:
        print("🏈 Enhanced Football Corner Prediction System with Auto-Tuning")
        print("=" * 80)
        
        # Load and clean data
        df = predictor.load_and_clean_data(csv_file_path)
        
        # Analyze data
        predictor.analyze_corner_data(df)
        
        # Prepare features and targets
        df_modeling, preprocessor, targets = predictor.prepare_features_and_targets(df)
        
        print(f"\nFinal dataset for modeling:")
        print(f"  Matches: {len(df_modeling):,}")
        print(f"  Features: {len(predictor.numerical_features + predictor.categorical_features)}")
        print(f"  Date range: {df_modeling['date_GMT'].min().strftime('%Y-%m-%d')} to {df_modeling['date_GMT'].max().strftime('%Y-%m-%d')}")
        
        # Train models with auto-tuning
        training_results = predictor.train_models_with_auto_tuning(df_modeling, preprocessor, targets)
        
        # Display results
        print(f"\n{'='*80}")
        print("TRAINING RESULTS SUMMARY")
        print(f"{'='*80}")
        
        for model_name, metrics in training_results.items():
            print(f"\n{model_name.replace('_', ' ').title()}:")
            for metric, value in metrics.items():
                print(f"  {metric.upper()}: {value:.4f}")
        
        # Display feature importance
        predictor.display_feature_importance(top_n=20)
        
        # Initialize chatbot
        chatbot = predictor.initialize_chatbot()
        
        print(f"\n{'='*80}")
        print("SYSTEM READY - STARTING INTERACTIVE CHATBOT")
        print(f"{'='*80}")
        print(f"✅ Models trained successfully!")
        print(f"🤖 Chatbot initialized and ready for interaction!")
        print(f"💬 You can now chat with the AI to get personalized corner predictions!")
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


if __name__ == "__main__":
    main()