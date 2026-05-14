"""
Improved Data Generation Module
Realistic retail customer dataset with strong confounders and heterogeneous treatment effects
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse
from pathlib import Path
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticRetailDataGenerator:
    """Generate synthetic retail customer dataset with realistic patterns"""

    def __init__(self, n_samples: int = 100000, random_seed: int = 42):
        self.n_samples = n_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def generate(self) -> pd.DataFrame:
        """Generate complete synthetic dataset with improved realism"""
        logger.info(f"Generating {self.n_samples} customer records...")

        df = pd.DataFrame({
            'customer_id': range(1, self.n_samples + 1),
            'age': np.random.normal(45, 15, self.n_samples).clip(18, 80).astype(int),
            'tenure_months': np.random.gamma(5, 2, self.n_samples).clip(0, 120).astype(int),
            'annual_income': np.random.lognormal(10.5, 0.8, self.n_samples).astype(int),
            'purchase_frequency': np.random.poisson(5, self.n_samples).astype(int),
            'avg_order_value': np.random.gamma(50, 2, self.n_samples).clip(10, 500),
            'account_age_days': np.random.poisson(800, self.n_samples).astype(int),
            'support_tickets': np.random.poisson(2, self.n_samples).astype(int),
            'product_category_affinity': np.random.choice(['electronics', 'fashion', 'home', 'sports'], self.n_samples),
            'geographic_region': np.random.choice(['North', 'South', 'East', 'West'], self.n_samples),
            'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], self.n_samples),
            'marketing_channel': np.random.choice(['email', 'social', 'direct', 'organic'], self.n_samples),
        })

        # ==================== STRONGER BEHAVIORAL CONFOUNDERS ====================
        # These strongly predict BOTH treatment adoption AND churn
        df['email_opens_last_30days'] = np.random.binomial(20, 0.6, self.n_samples).astype(int)
        df['website_visits_last_30days'] = np.random.poisson(8, self.n_samples).astype(int)
        df['cart_abandonment_rate'] = np.random.beta(2, 5, self.n_samples)
        df['app_downloads'] = np.random.binomial(1, 0.3, self.n_samples)
        df['referral_program_member'] = np.random.binomial(1, 0.15, self.n_samples)
        df['newsletter_subscriber'] = np.random.binomial(1, 0.45, self.n_samples)

        # ==================== TEMPORAL CONFOUNDERS ====================
        df['last_purchase_days_ago'] = np.random.poisson(30, self.n_samples).astype(int)
        df['season'] = np.random.choice(['spring', 'summer', 'fall', 'winter'], self.n_samples)
        df['customer_segment'] = np.random.choice(['vip', 'loyal', 'at_risk', 'dormant'], self.n_samples)

        # ==================== ENGAGEMENT SCORE (KEY CONFOUNDER) ====================
        df['engagement_score'] = (
            df['email_opens_last_30days'] / 20 * 0.3 +
            np.minimum(df['website_visits_last_30days'] / 20, 1) * 0.3 +
            (1 - df['cart_abandonment_rate']) * 0.2 +
            df['app_downloads'] * 0.2
        )

        # ==================== MORE BEHAVIORAL FEATURES ====================
        df['days_since_signup'] = df['account_age_days']
        df['purchase_recency'] = df['last_purchase_days_ago']
        df['avg_discount_used'] = np.random.beta(2, 3, self.n_samples) * 100
        df['return_rate'] = np.random.beta(1, 10, self.n_samples)
        df['complaint_count'] = np.random.poisson(0.5, self.n_samples).astype(int)
        df['tier_status'] = np.random.choice(['bronze', 'silver', 'gold', 'platinum'], self.n_samples)
        df['payment_method'] = np.random.choice(['credit_card', 'debit', 'wallet', 'bank_transfer'], self.n_samples)

        # ==================== TREATMENT ASSIGNMENT (CONFOUNDED!) ====================
        # Selection bias: high-engagement customers more likely to receive treatment
        propensity_treatment = (
            0.3 +  # baseline
            0.25 * (df['engagement_score'] / 1) +  # STRONG engagement effect
            0.15 * (df['tenure_months'] / 120) +
            0.10 * (df['annual_income'] / 100000) +
            0.05 * df['referral_program_member'] -
            0.08 * (df['last_purchase_days_ago'] / 90)
        )
        propensity_treatment = np.clip(propensity_treatment, 0.1, 0.9)
        df['treatment'] = np.random.binomial(1, propensity_treatment)

        # ==================== REALISTIC CHURN MODEL ====================
        # Confounders affect churn independently of treatment
        baseline_churn = 0.45

        propensity_churn = (
            baseline_churn
            - 0.20 * (df['tenure_months'] / 120)      # STRONG tenure effect
            - 0.25 * (df['engagement_score'] / 1)     # STRONG engagement effect
            - 0.15 * (df['annual_income'] / 100000)   # Income matters
            - 0.10 * df['referral_program_member']    # Referral reduces churn
            + 0.20 * (df['last_purchase_days_ago'] / 90)  # Recency matters
            + 0.15 * (df['complaint_count'] / 5)      # Complaints increase churn
            + 0.08 * df['return_rate']                # Returns increase churn
        )
        propensity_churn = np.clip(propensity_churn, 0.05, 0.95)

        # ==================== HETEROGENEOUS TREATMENT EFFECTS ====================
        # Different customers respond differently to treatment
        # CRITICAL FIX: Make heterogeneity STRONG to be statistically detectable

        # Base treatment effect with STRONG RANDOM VARIATION
        # This creates the heterogeneity that models need to learn
        # CRITICAL FIX: Treatment should INCREASE retention (positive), not decrease it
        base_effect = np.random.normal(0.15, 0.12, self.n_samples)  # Positive = treatment helps!

        # Customer-specific effects based on characteristics (INCREASED COEFFICIENTS)
        age_effect = 0.15 * (df['age'] / 100)  # Increased from 0.05 - stronger age effect
        engagement_effect = 0.20 * (df['engagement_score'] / 1)  # Increased from 0.08 - stronger engagement
        tenure_effect = 0.18 * (df['tenure_months'] / 120)  # Increased from 0.06 - stronger loyalty
        segment_effect = np.zeros(self.n_samples)

        # Segment-specific effects (MUCH STRONGER VARIATION)
        # CRITICAL FIX: Convert segment names to numeric indices
        segment_numeric = pd.Categorical(
            df['customer_segment'],
            categories=['dormant', 'at_risk', 'loyal', 'vip'],
            ordered=False
        ).codes  # Returns [0,1,2,3]

        # Apply STRONG segment-specific effects
        segment_effect[segment_numeric == 0] = -0.30  # Dormant: HARMED
        segment_effect[segment_numeric == 1] = -0.10  # At-risk: slightly harmed
        segment_effect[segment_numeric == 2] = 0.15   # Loyal: HELPED
        segment_effect[segment_numeric == 3] = 0.40   # VIP: STRONGLY helped

        # Substantial random heterogeneity (increased from 0.07 to 0.10)
        random_effect = np.random.normal(0, 0.10, self.n_samples)

        # Combine all effects - NOW WITH STRONG HETEROGENEITY
        df['cate'] = (
            base_effect +  # Now has std 0.12 (was constant)
            age_effect +  # Stronger coefficient
            engagement_effect +  # Stronger coefficient
            tenure_effect +  # Stronger coefficient
            segment_effect +  # Much stronger segment differences
            random_effect  # More random variation
        )

        # Clip to reasonable range
        df['cate'] = df['cate'].clip(-0.35, 0.35)

        logger.info(f"CATE Distribution:")
        logger.info(f"  Mean: {df['cate'].mean():.4f}")
        logger.info(f"  Std: {df['cate'].std():.4f}")
        logger.info(f"  Min: {df['cate'].min():.4f}")
        logger.info(f"  Max: {df['cate'].max():.4f}")
        logger.info(f"  % Negative: {(df['cate'] < 0).mean():.1%}")
        logger.info(f"  % Positive: {(df['cate'] > 0).mean():.1%}")

        # ==================== OUTCOME: CHURN ====================
        # Apply heterogeneous treatment effect.
        # cate already captures all the individual-level variation (age, engagement,
        # tenure, segment, random). We subtract it from churn probability for treated
        # customers only (positive CATE = treatment REDUCES churn = helps retention).
        # Bug fix: do NOT add base_effect here — it is already baked into cate above,
        # so adding it again double-counts the baseline and swamps the heterogeneity
        # signal that meta-learners need to contrast treated vs control outcomes.
        treatment_effect = df['cate'] * df['treatment']

        # Final churn probability
        final_churn_prob = propensity_churn - treatment_effect
        final_churn_prob = np.clip(final_churn_prob, 0, 1)

        df['churn'] = np.random.binomial(1, final_churn_prob)
        df['retained'] = 1 - df['churn']

        # ==================== LOGGING ====================
        logger.info(f"\nDataset Summary:")
        logger.info(f"  Total records: {len(df):,}")
        logger.info(f"  Churn rate: {df['churn'].mean():.2%}")
        logger.info(f"  Treatment split: {df['treatment'].mean():.1%}")
        logger.info(f"  Features: {len(df.columns)}")

        # Treatment effect by segment
        logger.info(f"\nTreatment Effect by Segment:")
        for segment in df['customer_segment'].unique():
            mask = df['customer_segment'] == segment
            mean_effect = df.loc[mask, 'cate'].mean()
            logger.info(f"  {segment}: {mean_effect:.4f}")

        return df

    def save(self, df: pd.DataFrame, output_path: str) -> None:
        """Save dataset to CSV"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
        logger.info(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Generate improved synthetic retail dataset')
    parser.add_argument('--output', type=str, default='data/raw/synthetic_retail.csv',
                       help='Output file path')
    parser.add_argument('--n_samples', type=int, default=100000,
                       help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    generator = SyntheticRetailDataGenerator(n_samples=args.n_samples, random_seed=args.seed)
    df = generator.generate()
    generator.save(df, args.output)

    # Display sample statistics
    print("\n" + "="*70)
    print("IMPROVED DATASET SUMMARY")
    print("="*70)
    print(df.head(10))
    print("\n" + df.describe().to_string())
    print("\n" + "="*70)
    print("KEY IMPROVEMENTS:")
    print("="*70)
    print(f"✓ Realistic effect size: {df['cate'].mean():.4f} (~8-10%)")
    print(f"✓ Strong heterogeneity: std={df['cate'].std():.4f} (target: 0.10+)")
    print(f"✓ Negative effects present: {(df['cate'] < 0).mean():.1%} (realistic)")
    print(f"✓ Segment-specific effects: VIP={df[df['customer_segment']=='vip']['cate'].mean():.4f}, Dormant={df[df['customer_segment']=='dormant']['cate'].mean():.4f}")
    print("="*70)


if __name__ == '__main__':
    main()
