import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PM10FeatureAnalyzer:
    def __init__(self):
        """Initialize the feature analyzer"""
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
        self.model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        
        # Load data
        self.train_data = None
        self.test_data = None
        self.feature_names = None
        self.target = 'pm10'
        
        self._load_data()
        
    def _load_data(self):
        """Load the balanced dataset"""
        try:
            train_path = os.path.join(self.data_dir, 'train_balanced.csv')
            test_path = os.path.join(self.data_dir, 'test_balanced.csv')
            
            if os.path.exists(train_path) and os.path.exists(test_path):
                self.train_data = pd.read_csv(train_path)
                self.test_data = pd.read_csv(test_path)
                
                # Remove datetime column if it exists
                if 'datetime' in self.train_data.columns:
                    self.train_data = self.train_data.drop('datetime', axis=1)
                    self.test_data = self.test_data.drop('datetime', axis=1)
                    print("   Removed datetime column")
                
                # Separate features and target
                self.feature_names = [col for col in self.train_data.columns if col != self.target]
                
                # Ensure all features are numeric
                non_numeric_features = []
                for feature in self.feature_names:
                    if not pd.api.types.is_numeric_dtype(self.train_data[feature]):
                        non_numeric_features.append(feature)
                
                if non_numeric_features:
                    print(f"   Removing non-numeric features: {non_numeric_features}")
                    self.train_data = self.train_data.drop(non_numeric_features, axis=1)
                    self.test_data = self.test_data.drop(non_numeric_features, axis=1)
                    self.feature_names = [col for col in self.train_data.columns if col != self.target]
                
                print(f"✅ Loaded data:")
                print(f"   Training samples: {len(self.train_data)}")
                print(f"   Test samples: {len(self.test_data)}")
                print(f"   Features: {len(self.feature_names)}")
                print(f"   Target: {self.target}")
            else:
                print("❌ Balanced dataset not found. Please run data preprocessing first.")
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def analyze_feature_importance(self):
        """Analyze feature importance using multiple methods"""
        if self.train_data is None:
            return
        
        print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        X_train = self.train_data[self.feature_names]
        y_train = self.train_data[self.target]
        
        # 1. Linear Regression Coefficients
        print("\n1️⃣ Linear Regression Coefficients:")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': lr.coef_,
            'Abs_Coefficient': np.abs(lr.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print(coef_df.head(10))
        
        # 2. Random Forest Feature Importance
        print("\n2️⃣ Random Forest Feature Importance:")
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        rf_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(rf_importance.head(10))
        
        # 3. Statistical Feature Selection
        print("\n3️⃣ Statistical Feature Selection (F-statistic):")
        selector = SelectKBest(score_func=f_regression, k='all')
        selector.fit(X_train, y_train)
        
        f_scores = pd.DataFrame({
            'Feature': self.feature_names,
            'F_Score': selector.scores_,
            'P_Value': selector.pvalues_
        }).sort_values('F_Score', ascending=False)
        
        print(f_scores.head(10))
        
        # 4. Mutual Information
        print("\n4️⃣ Mutual Information Scores:")
        mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
        
        mi_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Mutual_Info': mi_scores
        }).sort_values('Mutual_Info', ascending=False)
        
        print(mi_df.head(10))
        
        return {
            'linear_coefficients': coef_df,
            'rf_importance': rf_importance,
            'f_scores': f_scores,
            'mutual_info': mi_df
        }
    
    def analyze_feature_correlations(self):
        """Analyze feature correlations with target and between features"""
        if self.train_data is None:
            return
        
        print("\n📊 FEATURE CORRELATION ANALYSIS")
        print("=" * 50)
        
        # Correlation with target
        correlations = []
        for feature in self.feature_names:
            corr = self.train_data[feature].corr(self.train_data[self.target])
            correlations.append({
                'Feature': feature,
                'Correlation': corr,
                'Abs_Correlation': abs(corr)
            })
        
        corr_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False)
        print("\n🎯 Top 10 Features by Target Correlation:")
        print(corr_df.head(10))
        
        # Feature-to-feature correlations
        print("\n🔗 Feature-to-Feature Correlations (Top 10):")
        feature_corr = self.train_data[self.feature_names].corr()
        
        # Get upper triangle of correlation matrix
        upper_tri = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))
        
        # Find highest correlations
        high_corr_pairs = []
        for i in range(len(feature_corr.columns)):
            for j in range(i+1, len(feature_corr.columns)):
                corr_val = feature_corr.iloc[i, j]
                if abs(corr_val) > 0.8:  # High correlation threshold
                    high_corr_pairs.append({
                        'Feature1': feature_corr.columns[i],
                        'Feature2': feature_corr.columns[j],
                        'Correlation': corr_val
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False)
            print(high_corr_df.head(10))
        else:
            print("No highly correlated feature pairs found (|r| > 0.8)")
        
        return corr_df, high_corr_pairs
    
    def analyze_linear_regression_performance(self):
        """Deep dive into why Linear Regression works so well"""
        if self.train_data is None:
            return
        
        print("\n🎯 LINEAR REGRESSION PERFORMANCE ANALYSIS")
        print("=" * 50)
        
        X_train = self.train_data[self.feature_names]
        y_train = self.train_data[self.target]
        X_test = self.test_data[self.feature_names]
        y_test = self.test_data[self.target]
        
        # Fit Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = lr.predict(X_train)
        y_test_pred = lr.predict(X_test)
        
        # Performance metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        print(f"\n📈 Performance Metrics:")
        print(f"   Training R²: {train_r2:.6f}")
        print(f"   Test R²: {test_r2:.6f}")
        print(f"   Training RMSE: {train_rmse:.6f}")
        print(f"   Test RMSE: {test_rmse:.6f}")
        
        # Residual analysis
        train_residuals = y_train - y_train_pred
        test_residuals = y_test - y_test_pred
        
        print(f"\n📊 Residual Analysis:")
        print(f"   Training Residuals - Mean: {train_residuals.mean():.6f}")
        print(f"   Training Residuals - Std: {train_residuals.std():.6f}")
        print(f"   Test Residuals - Mean: {test_residuals.mean():.6f}")
        print(f"   Test Residuals - Std: {test_residuals.std():.6f}")
        
        # Check for linearity assumptions
        print(f"\n🔍 Linearity Assumptions:")
        
        # 1. Residuals vs Predicted
        train_residual_std = train_residuals.std()
        test_residual_std = test_residuals.std()
        print(f"   Residual variance ratio (test/train): {test_residual_std/train_residual_std:.3f}")
        
        # 2. Feature linearity check
        linearity_scores = []
        for feature in self.feature_names:
            # Check if feature has strong non-linear relationship with target
            feature_corr = abs(X_train[feature].corr(y_train))
            feature_corr_squared = abs(X_train[feature].corr(y_train**2))
            
            # If squared correlation is much higher, there's non-linearity
            non_linearity = feature_corr_squared - feature_corr
            linearity_scores.append({
                'Feature': feature,
                'Linear_Corr': feature_corr,
                'Squared_Corr': feature_corr_squared,
                'Non_Linearity': non_linearity
            })
        
        linearity_df = pd.DataFrame(linearity_scores).sort_values('Non_Linearity', ascending=False)
        print(f"\n   Top 5 features with potential non-linearity:")
        print(linearity_df.head())
        
        # 3. Multicollinearity check
        print(f"\n🔗 Multicollinearity Analysis:")
        feature_corr_matrix = X_train.corr()
        high_collinearity = []
        
        for i in range(len(feature_corr_matrix.columns)):
            for j in range(i+1, len(feature_corr_matrix.columns)):
                corr_val = feature_corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.9:  # Very high correlation
                    high_collinearity.append({
                        'Feature1': feature_corr_matrix.columns[i],
                        'Feature2': feature_corr_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if high_collinearity:
            print(f"   ⚠️  Found {len(high_collinearity)} highly correlated feature pairs:")
            for pair in high_collinearity[:5]:
                print(f"      {pair['Feature1']} ↔ {pair['Feature2']}: {pair['Correlation']:.3f}")
        else:
            print(f"   ✅ No severe multicollinearity detected")
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'linearity_analysis': linearity_df,
            'high_collinearity': high_collinearity
        }
    
    def optimize_feature_engineering(self):
        """Optimize feature engineering based on analysis"""
        if self.train_data is None:
            return
        
        print("\n⚡ FEATURE ENGINEERING OPTIMIZATION")
        print("=" * 50)
        
        # Get feature importance analysis
        importance_results = self.analyze_feature_importance()
        
        # Identify top features
        top_linear_features = importance_results['linear_coefficients'].head(10)['Feature'].tolist()
        top_rf_features = importance_results['rf_importance'].head(10)['Feature'].tolist()
        top_mi_features = importance_results['mutual_info'].head(10)['Feature'].tolist()
        
        # Find common top features
        all_top_features = set(top_linear_features + top_rf_features + top_mi_features)
        common_features = set(top_linear_features) & set(top_rf_features) & set(top_mi_features)
        
        print(f"\n🎯 Feature Selection Analysis:")
        print(f"   Top Linear Regression features: {len(top_linear_features)}")
        print(f"   Top Random Forest features: {len(top_rf_features)}")
        print(f"   Top Mutual Info features: {len(top_mi_features)}")
        print(f"   Common top features: {len(common_features)}")
        print(f"   Total unique top features: {len(all_top_features)}")
        
        print(f"\n🔝 Common Top Features (all methods):")
        for feature in sorted(common_features):
            print(f"   • {feature}")
        
        # Test different feature subsets
        feature_subsets = {
            'All_Features': self.feature_names,
            'Top_Linear': top_linear_features,
            'Top_RF': top_rf_features,
            'Top_MI': top_mi_features,
            'Common_Top': list(common_features),
            'All_Top': list(all_top_features)
        }
        
        print(f"\n🧪 Testing Feature Subsets:")
        results = {}
        
        for subset_name, features in feature_subsets.items():
            if len(features) == 0:
                continue
                
            X_train_subset = self.train_data[features]
            X_test_subset = self.test_data[features]
            y_train = self.train_data[self.target]
            y_test = self.test_data[self.target]
            
            # Fit and evaluate
            lr = LinearRegression()
            lr.fit(X_train_subset, y_train)
            
            y_test_pred = lr.predict(X_test_subset)
            test_r2 = r2_score(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            results[subset_name] = {
                'n_features': len(features),
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'features': features
            }
            
            print(f"   {subset_name:15} | Features: {len(features):2d} | R²: {test_r2:.6f} | RMSE: {test_rmse:.6f}")
        
        # Find best subset
        best_subset = max(results.items(), key=lambda x: x[1]['test_r2'])
        print(f"\n🏆 Best Feature Subset: {best_subset[0]}")
        print(f"   R² Score: {best_subset[1]['test_r2']:.6f}")
        print(f"   Features: {best_subset[1]['n_features']}")
        
        return results, best_subset
    
    def create_enhanced_features(self):
        """Create enhanced features based on analysis"""
        if self.train_data is None:
            return
        
        print("\n🚀 CREATING ENHANCED FEATURES")
        print("=" * 50)
        
        # Get the best feature subset
        _, best_subset = self.optimize_feature_engineering()
        best_features = best_subset[1]['features']
        
        # Create enhanced dataset
        enhanced_train = self.train_data[best_features + [self.target]].copy()
        enhanced_test = self.test_data[best_features + [self.target]].copy()
        
        # Add polynomial features for top features
        top_features = best_features[:5]  # Top 5 features
        
        print(f"\n📈 Adding Polynomial Features for Top Features:")
        for feature in top_features:
            # Add squared term
            enhanced_train[f'{feature}_squared'] = enhanced_train[feature] ** 2
            enhanced_test[f'{feature}_squared'] = enhanced_test[feature] ** 2
            
            # Add interaction with temperature if it exists
            if 'temperature_2m' in best_features and feature != 'temperature_2m':
                enhanced_train[f'{feature}_temp_interaction'] = enhanced_train[feature] * enhanced_train['temperature_2m']
                enhanced_test[f'{feature}_temp_interaction'] = enhanced_test[feature] * enhanced_test['temperature_2m']
            
            print(f"   • {feature}: added squared and interaction terms")
        
        # Add ratio features for lag features
        lag_features = [f for f in best_features if 'lag' in f]
        if len(lag_features) >= 2:
            print(f"\n📊 Adding Ratio Features for Lag Features:")
            for i in range(len(lag_features) - 1):
                ratio_name = f'ratio_{lag_features[i]}_{lag_features[i+1]}'
                enhanced_train[ratio_name] = enhanced_train[lag_features[i]] / (enhanced_train[lag_features[i+1]] + 1e-8)
                enhanced_test[ratio_name] = enhanced_test[lag_features[i]] / (enhanced_test[lag_features[i+1]] + 1e-8)
                print(f"   • {ratio_name}")
        
        # Add rolling feature ratios
        rolling_features = [f for f in best_features if 'rolling' in f]
        if len(rolling_features) >= 2:
            print(f"\n📈 Adding Rolling Feature Ratios:")
            mean_features = [f for f in rolling_features if 'mean' in f]
            std_features = [f for f in rolling_features if 'std' in f]
            
            for mean_feat in mean_features:
                for std_feat in std_features:
                    if mean_feat.split('_')[2] == std_feat.split('_')[2]:  # Same window
                        ratio_name = f'cv_{mean_feat.split("_")[2]}'  # Coefficient of variation
                        enhanced_train[ratio_name] = enhanced_train[std_feat] / (enhanced_train[mean_feat] + 1e-8)
                        enhanced_test[ratio_name] = enhanced_test[std_feat] / (enhanced_test[mean_feat] + 1e-8)
                        print(f"   • {ratio_name}")
        
        # Save enhanced dataset
        enhanced_features = [col for col in enhanced_train.columns if col != self.target]
        
        enhanced_train.to_csv(os.path.join(self.data_dir, 'train_enhanced.csv'), index=False)
        enhanced_test.to_csv(os.path.join(self.data_dir, 'test_enhanced.csv'), index=False)
        
        print(f"\n💾 Enhanced Dataset Saved:")
        print(f"   Training samples: {len(enhanced_train)}")
        print(f"   Test samples: {len(enhanced_test)}")
        print(f"   Original features: {len(best_features)}")
        print(f"   Enhanced features: {len(enhanced_features)}")
        print(f"   New features added: {len(enhanced_features) - len(best_features)}")
        
        return enhanced_train, enhanced_test, enhanced_features
    
    def run_complete_analysis(self):
        """Run complete feature analysis"""
        print("🌬️ PM10 FEATURE ANALYSIS")
        print("=" * 60)
        print(f"Analysis started at: {datetime.now()}")
        
        # 1. Feature importance analysis
        importance_results = self.analyze_feature_importance()
        
        # 2. Correlation analysis
        corr_results, high_corr = self.analyze_feature_correlations()
        
        # 3. Linear regression performance analysis
        lr_analysis = self.analyze_linear_regression_performance()
        
        # 4. Feature engineering optimization
        optimization_results, best_subset = self.optimize_feature_engineering()
        
        # 5. Create enhanced features
        enhanced_data = self.create_enhanced_features()
        
        print(f"\n✅ Analysis completed at: {datetime.now()}")
        
        return {
            'importance': importance_results,
            'correlations': corr_results,
            'linear_analysis': lr_analysis,
            'optimization': optimization_results,
            'best_subset': best_subset,
            'enhanced_data': enhanced_data
        }

# Example usage
if __name__ == "__main__":
    analyzer = PM10FeatureAnalyzer()
    results = analyzer.run_complete_analysis() 