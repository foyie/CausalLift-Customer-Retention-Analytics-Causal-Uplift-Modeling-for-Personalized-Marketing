"""
Fully Fixed Causal Estimation Module
Compatible with latest EconML API (0.14.0+)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from econml.metalearners import TLearner, XLearner
from econml.dml import CausalForestDML
import logging
from pathlib import Path
import pickle
import argparse
from typing import Dict, Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedCausalEstimator:
    """Enhanced causal inference estimator with correct EconML API"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.estimators = {}

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for causal estimation"""

        # Features (X): all except targets, treatment, and derived composites.
        # Drop engagement_score: it's a linear combination of email_opens,
        # website_visits, cart_abandonment_rate, and app_downloads which are
        # already in X. Including it causes severe multicollinearity — the model
        # learns to use it as a confounder proxy instead of a CATE modifier,
        # which collapses predicted heterogeneity toward the mean.
        # Drop engineered interaction/aggregate features too: they were built
        # for propensity/outcome prediction, not heterogeneity estimation.
        exclude_cols = {
            'churn', 'retained', 'treatment', 'cate', 'customer_id',
            # Composite that duplicates raw features already in X:
            'engagement_score',
            # Engineered features from feature_engineering.py that are
            # products/logs of raw features — they collapse variance:
            'age_tenure_interaction', 'income_frequency_interaction',
            'engagement_tenure_interaction', 'engagement_score_squared',
            'income_log', 'total_engagement_events',
            'estimated_customer_value', 'churn_risk_score',
        }
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].values

        # Treatment (T)
        T = df['treatment'].values

        # Outcome (Y): use retained as the outcome (1 = retained, 0 = churned)
        y = df['retained'].values

        logger.info(f"Data shapes - X: {X.shape}, T: {T.shape}, y: {y.shape}")
        logger.info(f"Feature columns used ({len(feature_cols)}): {feature_cols}")
        logger.info(f"Treatment split: {T.mean():.1%} in treatment, {(1-T.mean()):.1%} in control")
        logger.info(f"Outcome mean (retention): {y.mean():.1%}")

        return X, T, y

    def estimate_t_learner(self, X: np.ndarray, T: np.ndarray, y: np.ndarray) -> 'ImprovedCausalEstimator':
        """
        Improved T-Learner with diverse base learners
        Uses DIFFERENT learners for control and treatment groups
        """
        logger.info("=" * 70)
        logger.info("Training Improved T-Learner")
        logger.info("=" * 70)

        try:
            # Use DIFFERENT learners for each group for better diversity
            estimator = TLearner(
                models=[
                    xgb.XGBRegressor(
                        max_depth=4,
                        n_estimators=100,
                        learning_rate=0.15,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=self.random_state,
                        verbosity=0
                    ),  # Control group learner
                    xgb.XGBRegressor(
                        max_depth=6,
                        n_estimators=100,
                        learning_rate=0.10,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=self.random_state,
                        verbosity=0
                    ),  # Treatment group learner (different params)
                ]
            )

            estimator.fit(y, T, X=X)
            self.estimators['t_learner'] = estimator

            # Get CATE predictions
            cate_t_learner = estimator.effect(X)

            logger.info(f"T-Learner CATE Statistics:")
            logger.info(f"  Shape: {cate_t_learner.shape}")
            logger.info(f"  Mean: {cate_t_learner.mean():.4f}")
            logger.info(f"  Std: {cate_t_learner.std():.4f}")
            logger.info(f"  Min: {cate_t_learner.min():.4f}")
            logger.info(f"  Max: {cate_t_learner.max():.4f}")
            logger.info(f"  % Negative: {(cate_t_learner < 0).mean():.1%}")

            self.models['t_learner'] = {
                'estimator': estimator,
                'cate': cate_t_learner,
                'type': 'T-Learner'
            }
        except Exception as e:
            logger.error(f"Error in T-Learner: {str(e)}")
            raise

        return self

    def estimate_x_learner(self, X: np.ndarray, T: np.ndarray, y: np.ndarray) -> 'ImprovedCausalEstimator':
        """
        Improved X-Learner with cross-fitting and robust propensity estimation
        FIXED: Using correct API (no propensity_models parameter)
        """
        logger.info("=" * 70)
        logger.info("Training Improved X-Learner")
        logger.info("=" * 70)

        try:
            # X-Learner with corrected API (current EconML versions)
            # propensity_models parameter was removed in newer versions
            estimator = XLearner(
                models=GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=self.random_state
                ),
                # Note: propensity_models removed - X-Learner estimates propensity internally
                cate_models=xgb.XGBRegressor(
                    max_depth=5,
                    n_estimators=100,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    verbosity=0
                )
            )

            estimator.fit(y, T, X=X)
            self.estimators['x_learner'] = estimator

            # Get CATE predictions
            cate_x_learner = estimator.effect(X)

            logger.info(f"X-Learner CATE Statistics:")
            logger.info(f"  Shape: {cate_x_learner.shape}")
            logger.info(f"  Mean: {cate_x_learner.mean():.4f}")
            logger.info(f"  Std: {cate_x_learner.std():.4f}")
            logger.info(f"  Min: {cate_x_learner.min():.4f}")
            logger.info(f"  Max: {cate_x_learner.max():.4f}")
            logger.info(f"  % Negative: {(cate_x_learner < 0).mean():.1%}")

            self.models['x_learner'] = {
                'estimator': estimator,
                'cate': cate_x_learner,
                'type': 'X-Learner'
            }
        except Exception as e:
            logger.error(f"Error in X-Learner: {str(e)}")
            raise

        return self

    def estimate_causal_forest(self, X: np.ndarray, T: np.ndarray, y: np.ndarray) -> 'ImprovedCausalEstimator':
        """
        CausalForestDML — the correct EconML class for DML-style nuisance residualization.

        econml.grf.CausalForest does NOT accept model_y/model_t — it is a
        lower-level GRF implementation. econml.dml.CausalForestDML is the
        high-level wrapper that:
          1. Fits model_y to estimate E[Y|X] and model_t to estimate E[T|X]
             via cross-fitting (k-fold to avoid overfitting nuisance models)
          2. Computes residuals: Y_res = Y - E[Y|X], T_res = T - E[T|X]
          3. Fits a causal forest on the residualized problem:
             Y_res = theta(X) * T_res + epsilon
        This separates confounding (partialled out in step 1-2) from
        heterogeneity (estimated in step 3), which is why it produces
        meaningful CATE variance instead of near-constant predictions.

        API differences from CausalForest:
        - fit(Y, T, X=X)  not  fit(X, T, Y)
        - effect(X)       not  predict(X)
        """
        logger.info("=" * 70)
        logger.info("Training CausalForestDML (with nuisance residualization)")
        logger.info("=" * 70)

        try:
            estimator = CausalForestDML(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=10,
                random_state=self.random_state,
                n_jobs=-1,
                cv=3,                  # 3-fold cross-fitting for nuisance models
                model_y=GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    subsample=0.8, random_state=self.random_state
                ),
                model_t=LogisticRegression(
                    C=1.0, max_iter=500, random_state=self.random_state
                ),
            )

            # CausalForestDML uses fit(Y, T, X=X) — note argument order
            estimator.fit(y, T, X=X)
            self.estimators['causal_forest'] = estimator

            # effect() is the correct prediction method for DML estimators
            cate_cf = estimator.effect(X).ravel()

            logger.info(f"CausalForestDML CATE Statistics:")
            logger.info(f"  Shape: {cate_cf.shape}")
            logger.info(f"  Mean: {cate_cf.mean():.4f}")
            logger.info(f"  Std: {cate_cf.std():.4f}")
            logger.info(f"  Min: {cate_cf.min():.4f}")
            logger.info(f"  Max: {cate_cf.max():.4f}")
            logger.info(f"  % Negative: {(cate_cf < 0).mean():.1%}")

            self.models['causal_forest'] = {
                'estimator': estimator,
                'cate': cate_cf,
                'type': 'CausalForestDML'
            }
        except Exception as e:
            logger.error(f"Error in CausalForestDML: {str(e)}")
            raise

        return self

    def estimate_ensemble_robust(self) -> np.ndarray:
        """
        Robust ensemble using weighted average
        """
        logger.info("=" * 70)
        logger.info("Computing Robust Ensemble")
        logger.info("=" * 70)

        if len(self.models) < 2:
            logger.warning("Less than 2 models for ensemble")
            return list(self.models.values())[0]['cate']

        # Get all CATEs
        cates = []
        names = []

        for name in ['t_learner', 'x_learner', 'causal_forest']:
            if name in self.models:
                cates.append(self.models[name]['cate'])
                names.append(name)

        if len(cates) == 0:
            logger.error("No models available for ensemble")
            raise ValueError("No models trained")

        # Calculate agreement metrics
        if len(cates) >= 2:
            agreement_1_2 = np.abs(cates[0] - cates[1]).mean()
            logger.info(f"Model Agreement (lower is better):")
            logger.info(f"  {names[0]} vs {names[1]}: {agreement_1_2:.4f}")

            if len(cates) >= 3:
                agreement_1_3 = np.abs(cates[0] - cates[2]).mean()
                agreement_2_3 = np.abs(cates[1] - cates[2]).mean()
                logger.info(f"  {names[0]} vs {names[2]}: {agreement_1_3:.4f}")
                logger.info(f"  {names[1]} vs {names[2]}: {agreement_2_3:.4f}")

        # Equal weight ensemble (simple and robust)
        ensemble_cate = np.mean(cates, axis=0)

        logger.info(f"\nEnsemble CATE Statistics:")
        logger.info(f"  Mean: {ensemble_cate.mean():.4f}")
        logger.info(f"  Std: {ensemble_cate.std():.4f}")
        logger.info(f"  Min: {ensemble_cate.min():.4f}")
        logger.info(f"  Max: {ensemble_cate.max():.4f}")
        logger.info(f"  % Negative: {(ensemble_cate < 0).mean():.1%}")

        self.models['ensemble'] = {
            'cate': ensemble_cate,
            'type': 'Ensemble (3 Models)'
        }

        return ensemble_cate

    def save(self, output_dir: str) -> None:
        """Save trained models"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for name, estimator in self.estimators.items():
            path = Path(output_dir) / f'{name}.pkl'
            with open(path, 'wb') as f:
                pickle.dump(estimator, f)
            logger.info(f"Saved {name} to {path}")


def compare_models(df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
    """Train and compare improved causal models"""

    logger.info("=" * 70)
    logger.info("STARTING IMPROVED CAUSAL ESTIMATION PIPELINE")
    logger.info("=" * 70)

    # Prepare data
    estimator = ImprovedCausalEstimator()
    X, T, y = estimator.prepare_data(df)

    # Train models
    logger.info("\n" + "=" * 70)
    estimator.estimate_t_learner(X, T, y)

    logger.info("\n" + "=" * 70)
    estimator.estimate_x_learner(X, T, y)

    logger.info("\n" + "=" * 70)
    estimator.estimate_causal_forest(X, T, y)

    logger.info("\n" + "=" * 70)
    # Ensemble
    estimator.estimate_ensemble_robust()

    # Save
    estimator.save(output_dir)

    # Create summary
    logger.info("\n" + "=" * 70)
    logger.info("CAUSAL ESTIMATION SUMMARY - IMPROVED RESULTS")
    logger.info("=" * 70)

    summary = {}
    for name, model in estimator.models.items():
        cate = model['cate']
        summary[name] = {
            'mean_effect': float(cate.mean()),
            'std_effect': float(cate.std()),
            'min_effect': float(cate.min()),
            'max_effect': float(cate.max()),
            'p10': float(np.percentile(cate, 10)),
            'p25': float(np.percentile(cate, 25)),
            'p50': float(np.percentile(cate, 50)),
            'p75': float(np.percentile(cate, 75)),
            'p90': float(np.percentile(cate, 90)),
            'pct_negative': float((cate < 0).mean())
        }

        logger.info(f"\n{name.upper()}")
        logger.info(f"  Mean effect: {summary[name]['mean_effect']:.4f}")
        logger.info(f"  Std effect: {summary[name]['std_effect']:.4f}")
        logger.info(f"  Range: [{summary[name]['min_effect']:.4f}, {summary[name]['max_effect']:.4f}]")
        logger.info(f"  Percentiles (p10, p25, p50, p75, p90):")
        logger.info(f"    {summary[name]['p10']:.4f}, {summary[name]['p25']:.4f}, {summary[name]['p50']:.4f}, {summary[name]['p75']:.4f}, {summary[name]['p90']:.4f}")
        logger.info(f"  % Negative: {summary[name]['pct_negative']:.1%}")

    # Save per-customer CATE predictions so validation.py can evaluate
    # model predictions instead of ground-truth oracle CATE.
    predictions_df = pd.DataFrame({'customer_id': df['customer_id'].values})
    if 'cate' in df.columns:
        predictions_df['true_cate'] = df['cate'].values
    for name in ['t_learner', 'x_learner', 'causal_forest', 'ensemble']:
        if name in estimator.models:
            predictions_df[f'{name}_cate'] = estimator.models[name]['cate']
    pred_path = Path(output_dir) / 'cate_predictions.csv'
    predictions_df.to_csv(pred_path, index=False)
    logger.info(f"\nPer-customer CATE predictions saved to {pred_path}")

    return summary, estimator.models


def main():
    parser = argparse.ArgumentParser(description='Improved causal estimation pipeline')
    parser.add_argument('--data', type=str, default='data/processed/features_engineered.csv',
                       help='Input engineered features CSV')
    parser.add_argument('--output', type=str, default='models/',
                       help='Output directory for models')

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Run improved causal estimation
    summary, models = compare_models(df, args.output)

    # Save summary
    summary_df = pd.DataFrame(summary).T
    summary_path = Path(args.output) / 'causal_summary.csv'
    summary_df.to_csv(summary_path)
    logger.info(f"\nSummary saved to {summary_path}")

    # Print summary table
    print("\n" + summary_df.to_string())

    logger.info("\n" + "=" * 70)
    logger.info("✓ IMPROVED CAUSAL ESTIMATION COMPLETE")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
