"""
Feature Engineering Module
Preprocessing, feature scaling, encoding, and feature creation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from pathlib import Path
import argparse
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering and preprocessing pipeline"""
    
    def __init__(self):
        self.scaler = None
        self.encoders = {}
        self.categorical_features = None
        self.numerical_features = None
        
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """Fit preprocessors to data"""
        
        # Identify feature types
        self.categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        self.numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target/id columns
        for col in ['churn', 'retained', 'treatment', 'cate', 'customer_id']:
            if col in self.numerical_features:
                self.numerical_features.remove(col)
        
        logger.info(f"Numerical features ({len(self.numerical_features)}): {self.numerical_features[:5]}...")
        logger.info(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
        
        # Fit scalers
        self.scaler = StandardScaler()
        self.scaler.fit(df[self.numerical_features])
        
        # Fit label encoders for categorical
        for col in self.categorical_features:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.encoders[col] = le
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations to data"""
        df_transformed = df.copy()
        
        # Scale numerical features
        df_transformed[self.numerical_features] = self.scaler.transform(df[self.numerical_features])
        
        # Encode categorical features
        for col in self.categorical_features:
            df_transformed[col] = self.encoders[col].transform(df[col].astype(str))
        
        return df_transformed
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(df).transform(df)
    
    def save(self, path: str) -> None:
        """Save preprocessors"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'encoders': self.encoders,
                'categorical_features': self.categorical_features,
                'numerical_features': self.numerical_features
            }, f)
        logger.info(f"Preprocessors saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FeatureEngineer':
        """Load preprocessors"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        fe = cls()
        fe.scaler = data['scaler']
        fe.encoders = data['encoders']
        fe.categorical_features = data['categorical_features']
        fe.numerical_features = data['numerical_features']
        logger.info(f"Preprocessors loaded from {path}")
        return fe


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction and polynomial features"""
    df_new = df.copy()
    
    # Key interactions for causal estimation
    if 'age' in df.columns and 'tenure_months' in df.columns:
        df_new['age_tenure_interaction'] = df['age'] * df['tenure_months']
    
    if 'annual_income' in df.columns and 'purchase_frequency' in df.columns:
        df_new['income_frequency_interaction'] = df['annual_income'] * df['purchase_frequency']
    
    if 'engagement_score' in df.columns and 'tenure_months' in df.columns:
        df_new['engagement_tenure_interaction'] = df['engagement_score'] * df['tenure_months']
    
    # Polynomial features
    if 'engagement_score' in df.columns:
        df_new['engagement_score_squared'] = df['engagement_score'] ** 2
    
    if 'annual_income' in df.columns:
        df_new['income_log'] = np.log1p(df['annual_income'])
    
    logger.info(f"Created {len(df_new.columns) - len(df.columns)} interaction features")
    
    return df_new


def create_aggregation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregate features"""
    df_new = df.copy()
    
    # Behavioral aggregates
    if 'email_opens_last_30days' in df.columns and 'website_visits_last_30days' in df.columns:
        df_new['total_engagement_events'] = (
            df['email_opens_last_30days'] + df['website_visits_last_30days']
        )
    
    # Customer lifetime indicators
    if 'purchase_frequency' in df.columns and 'avg_order_value' in df.columns:
        df_new['estimated_customer_value'] = df['purchase_frequency'] * df['avg_order_value']
    
    # Risk indicators
    if 'cart_abandonment_rate' in df.columns and 'complaint_count' in df.columns:
        df_new['churn_risk_score'] = (
            df['cart_abandonment_rate'] * 0.5 + 
            np.minimum(df['complaint_count'] / 5, 1) * 0.5
        )
    
    logger.info(f"Created {len(df_new.columns) - len(df.columns)} aggregate features")
    
    return df_new


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values"""
    df_new = df.copy()
    
    for col in df_new.columns:
        if df_new[col].isnull().any():
            if df_new[col].dtype in [np.float64, np.int64]:
                df_new[col].fillna(df_new[col].median(), inplace=True)
            else:
                df_new[col].fillna(df_new[col].mode()[0], inplace=True)
            logger.info(f"Filled {col}")
    
    return df_new


def main():
    parser = argparse.ArgumentParser(description='Feature engineering pipeline')
    parser.add_argument('--input', type=str, default='data/raw/synthetic_retail.csv',
                       help='Input CSV file')
    parser.add_argument('--output', type=str, default='data/processed/features_engineered.csv',
                       help='Output CSV file')
    parser.add_argument('--preprocessor', type=str, default='models/preprocessor.pkl',
                       help='Save preprocessor to this path')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Create new features
    df = create_interaction_features(df)
    df = create_aggregation_features(df)
    
    # Identify features for preprocessing (exclude target and id)
    exclude_cols = {'churn', 'retained', 'treatment', 'cate', 'customer_id'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Prepare features and targets
    X = df[feature_cols].copy()
    y = df[['churn', 'retained', 'treatment', 'cate']].copy()
    
    # Feature engineering and scaling
    fe = FeatureEngineer()
    X_transformed = fe.fit_transform(X)
    
    # Save preprocessor
    fe.save(args.preprocessor)
    
    # Combine transformed features with targets
    result = pd.concat([
        pd.DataFrame(X_transformed, columns=feature_cols),
        y.reset_index(drop=True)
    ], axis=1)
    
    # Add customer_id
    result['customer_id'] = df['customer_id'].values
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    
    logger.info(f"Features engineered: {X_transformed.shape[1]} features")
    logger.info(f"Output saved to {args.output}")
    print("\nFeature Engineering Summary:")
    print(f"Original columns: {len(df.columns)}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Final dataset shape: {result.shape}")


if __name__ == '__main__':
    main()
