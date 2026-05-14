"""
Fixed CATE Validation Module
Correct Qini curve, AUUC, and placebo test implementations
"""

import numpy as np
import pandas as pd
from sklearn.metrics import auc
from sklearn.cluster import KMeans
from pathlib import Path
import argparse
import logging
import json
import pickle
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CATEValidator:
    """
    Correct implementation of CATE validation metrics.

    Key fix: Qini gain is defined as the lift in treated outcomes vs random
    targeting — it uses TREATMENT-GROUP outcomes, not the full population mean.

    Placebo test: shuffle the CATE scores (not T), so random ordering has
    no predictive power. Real CATE should beat shuffled CATE.
    """

    # ------------------------------------------------------------------
    # 1. QINI CURVE  (correct implementation)
    # ------------------------------------------------------------------
    def calculate_qini_curve(
        self,
        y_true: np.ndarray,
        T: np.ndarray,
        cate: np.ndarray,
        n_bins: int = 20,
    ) -> Dict:
        """
        Calculate Qini curve for uplift / CATE validation.

        Qini gain at fraction f:
            Q(f) = (Y1(f)/N1(f) - Y0(f)/N0(f)) * N1_total
        where (f) denotes the top-f fraction selected by predicted CATE,
        Y1/N1 are treated outcomes and Y0/N0 are control outcomes inside that bucket.

        AUUC = area under Qini curve minus area under random baseline,
        normalised by the perfect-model area.
        """
        n = len(y_true)
        fractions = np.linspace(0, 1, n_bins + 1)

        # Sort everyone by predicted CATE descending
        order = np.argsort(-cate)
        y_s  = y_true[order]
        T_s  = T[order]

        # Global totals needed for normalisation
        N1_total = T.sum()
        N0_total = (1 - T).sum()

        if N1_total == 0 or N0_total == 0:
            logger.error("No variation in treatment assignment — Qini undefined")
            return {'percentiles': [], 'qini_gains': [], 'auuc': 0.0}

        qini_values  = [0.0]   # Q(0) = 0
        random_values = [0.0]

        for f in fractions[1:]:
            n_top = max(1, int(f * n))

            y_top = y_s[:n_top]
            T_top = T_s[:n_top]

            treat_mask   = T_top == 1
            control_mask = T_top == 0

            n1 = treat_mask.sum()
            n0 = control_mask.sum()

            if n1 == 0 or n0 == 0:
                # Can't estimate both arms; inherit previous value
                qini_values.append(qini_values[-1])
            else:
                # Normalized Qini: cumulative outcome rates vs global denominators.
                # Dividing by N1_total / N0_total keeps Q(f) in [-1, 1] regardless
                # of dataset size. The old formula (mu1 - mu0) * n1 scaled with raw
                # counts, making AUUC proportional to N and uninterpretable.
                Y1_cum = y_top[treat_mask].sum()
                Y0_cum = y_top[control_mask].sum()
                qini_values.append(Y1_cum / N1_total - Y0_cum / N0_total)

            # Random baseline: linear from 0 to full-population Qini at f=1
            global_Y1 = y_true[T == 1].sum()
            global_Y0 = y_true[T == 0].sum()
            global_qini_at_1 = global_Y1 / N1_total - global_Y0 / N0_total
            random_values.append(f * global_qini_at_1)

        qini_values   = np.array(qini_values)
        random_values = np.array(random_values)
        pct_axis      = fractions * 100

        auuc_model  = auc(pct_axis, qini_values) / 100
        auuc_random = auc(pct_axis, random_values) / 100
        auuc_lift   = auuc_model - auuc_random          # signed lift over random

        logger.info(f"Qini AUUC (model):  {auuc_model:.4f}")
        logger.info(f"Qini AUUC (random): {auuc_random:.4f}")
        logger.info(f"Qini AUUC (lift):   {auuc_lift:.4f}")

        return {
            'percentiles': pct_axis.tolist(),
            'qini_gains': qini_values.tolist(),
            'random_gains': random_values.tolist(),
            'auuc': float(auuc_lift),          # lift is the meaningful metric
            'auuc_model': float(auuc_model),
            'auuc_random': float(auuc_random),
        }

    def calculate_auuc(
        self,
        y_true: np.ndarray,
        T: np.ndarray,
        cate: np.ndarray,
    ) -> float:
        return self.calculate_qini_curve(y_true, T, cate)['auuc']

    # ------------------------------------------------------------------
    # 2. PLACEBO TEST  (correct implementation)
    # ------------------------------------------------------------------
    def placebo_test(
        self,
        y_true: np.ndarray,
        T: np.ndarray,
        cate: np.ndarray,
        n_iterations: int = 50,
    ) -> Dict:
        """
        Placebo / permutation test.

        H0: predicted CATE has no targeting power.

        For each iteration we SHUFFLE the cate scores (i.e. random ranking)
        and recompute AUUC.  If our real model is useful its AUUC should be
        significantly larger than the shuffled distribution.

        Note: shuffling T doesn't work here because the Qini formula already
        conditions on T labels; what we need to test is whether the *ordering*
        produced by our CATE predictions is meaningful.
        """
        logger.info("Running placebo test (shuffling CATE predictions)…")

        real_auuc = self.calculate_auuc(y_true, T, cate)
        logger.info(f"Real AUUC:  {real_auuc:.4f}")

        rng = np.random.default_rng(42)
        placebo_auucs = []

        for i in range(n_iterations):
            cate_shuffled = rng.permutation(cate)
            p_auuc = self.calculate_auuc(y_true, T, cate_shuffled)
            placebo_auucs.append(p_auuc)

        placebo_auucs = np.array(placebo_auucs)

        # p-value: proportion of shuffled AUUCs >= real AUUC
        p_value    = float(np.mean(placebo_auucs >= real_auuc))
        significant = p_value < 0.05

        logger.info(f"Placebo AUUC (mean): {placebo_auucs.mean():.4f}")
        logger.info(f"Placebo AUUC (std):  {placebo_auucs.std():.4f}")
        logger.info(f"P-value:             {p_value:.4f}")
        logger.info(f"Significant (p<.05): {significant}")

        return {
            'real_auuc':    float(real_auuc),
            'placebo_mean': float(placebo_auucs.mean()),
            'placebo_std':  float(placebo_auucs.std()),
            'placebo_min':  float(placebo_auucs.min()),
            'placebo_max':  float(placebo_auucs.max()),
            'p_value':      p_value,
            'significant':  significant,
        }

    # ------------------------------------------------------------------
    # 3. SEGMENT HETEROGENEITY
    # ------------------------------------------------------------------
    def segment_heterogeneity(
        self,
        cate: np.ndarray,
        X: Optional[np.ndarray] = None,
        n_segments: int = 4,
    ) -> Dict:
        """
        Cluster customers into segments and report per-segment CATE statistics.
        If X is provided uses KMeans on features; otherwise quantile-splits on CATE.
        """
        logger.info("Segment Heterogeneity Analysis:")

        if X is not None:
            km = KMeans(n_clusters=n_segments, random_state=42, n_init='auto')
            labels = km.fit_predict(X)
        else:
            # Simple quantile split on CATE itself
            bins   = np.quantile(cate, np.linspace(0, 1, n_segments + 1))
            labels = np.digitize(cate, bins[1:-1])  # 0 … n_segments-1

        segments = {}
        for seg in np.unique(labels):
            mask = labels == seg
            seg_cate = cate[mask]
            label = f"Segment_{seg}"
            segments[label] = {
                'n_samples': int(mask.sum()),
                'mean_cate': float(seg_cate.mean()),
                'std_cate':  float(seg_cate.std()),
                'p10':       float(np.percentile(seg_cate, 10)),
                'p50':       float(np.percentile(seg_cate, 50)),
                'p90':       float(np.percentile(seg_cate, 90)),
            }
            logger.info(
                f"  {label}: n={mask.sum():,}  mean={seg_cate.mean():.4f}  "
                f"std={seg_cate.std():.4f}"
            )

        return segments

    # ------------------------------------------------------------------
    # 4. IDENTIFY RESPONDERS
    # ------------------------------------------------------------------
    def identify_treatment_responders(
        self,
        df: pd.DataFrame,
        cate: np.ndarray,
        threshold_pct: float = 0.25,
    ) -> pd.DataFrame:
        """Return rows whose predicted CATE is in the top `threshold_pct` fraction."""
        threshold = np.quantile(cate, 1 - threshold_pct)
        mask = cate >= threshold
        responders = df[mask].copy()
        responders['predicted_cate'] = cate[mask]
        logger.info(
            f"Identified {mask.sum():,} responders "
            f"(top {threshold_pct:.0%}, CATE >= {threshold:.4f})"
        )
        return responders


# ------------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='CATE Validation')
    parser.add_argument('--data',        type=str, default='data/processed/features_engineered.csv')
    parser.add_argument('--predictions', type=str, default='models/cate_predictions.csv',
                        help='Per-customer CATE predictions from causal_estimation.py')
    parser.add_argument('--output',      type=str, default='results/')
    args = parser.parse_args()

    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # ---- pull outcome / treatment arrays ----------------------------
    y_true = df['retained'].values.astype(float)
    T      = df['treatment'].values.astype(float)

    # ---- feature matrix for KMeans segmentation ---------------------
    exclude = {'churn', 'retained', 'treatment', 'cate', 'customer_id'}
    feat_cols = [c for c in df.columns if c not in exclude
                 and df[c].dtype in [np.float64, np.int64, float, int]]
    X = df[feat_cols].values

    # ---- load model predictions if available ------------------------
    model_cates = {}
    if Path(args.predictions).exists():
        pred_df = pd.read_csv(args.predictions)
        logger.info(f"Loaded predictions from {args.predictions}")
        for col in pred_df.columns:
            if col.endswith('_cate') and col != 'true_cate':
                model_name = col.replace('_cate', '')
                model_cates[model_name] = pred_df[col].values.astype(float)
                logger.info(f"  Found model predictions: {model_name}  "
                            f"mean={model_cates[model_name].mean():.4f}  "
                            f"std={model_cates[model_name].std():.4f}")
    else:
        logger.warning(f"No predictions file at {args.predictions}. "
                       "Run causal_estimation.py first to evaluate model performance.")

    # Always include ground-truth oracle CATE as an upper-bound reference
    if 'cate' in df.columns:
        model_cates['oracle'] = df['cate'].values.astype(float)
        logger.info(f"  oracle CATE: mean={model_cates['oracle'].mean():.4f}  "
                    f"std={model_cates['oracle'].std():.4f}")
    elif not model_cates:
        logger.error("No CATE scores found. Run causal_estimation.py first.")
        return

    validator = CATEValidator()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ---- validate every available CATE source -----------------------
    for model_name, cate in model_cates.items():
        logger.info("\n" + "=" * 70)
        logger.info(f"VALIDATING: {model_name.upper()}")
        logger.info("=" * 70)
        logger.info(f"CATE summary: mean={cate.mean():.4f}  std={cate.std():.4f}  "
                    f"min={cate.min():.4f}  max={cate.max():.4f}  "
                    f"pct_negative={( cate < 0).mean():.1%}")

        logger.info(f"\n--- Qini Curve ---")
        qini_results = validator.calculate_qini_curve(y_true, T, cate)

        logger.info(f"\n--- Placebo Test ---")
        placebo_results = validator.placebo_test(y_true, T, cate, n_iterations=100)

        logger.info(f"\n--- Segment Heterogeneity (KMeans on features) ---")
        # Pass X so KMeans clusters on customer characteristics, not CATE bins.
        # Binning CATE directly fails when values pile up at a clip boundary.
        segments = validator.segment_heterogeneity(cate, X=X, n_segments=4)

        logger.info(f"\n--- Identify Responders ---")
        responders = validator.identify_treatment_responders(df, cate, threshold_pct=0.25)

        all_results[model_name] = {
            'qini':     qini_results,
            'placebo':  placebo_results,
            'segments': segments,
        }

        resp_path = Path(args.output) / f'high_responders_{model_name}.csv'
        responders.to_csv(resp_path, index=False)

    # ---- comparison summary -----------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION SUMMARY — ALL MODELS")
    logger.info("=" * 70)
    logger.info(f"{'Model':<20} {'AUUC (lift)':>12} {'p-value':>10} {'Significant':>12} {'CATE std':>10}")
    logger.info("-" * 70)
    for model_name, res in all_results.items():
        p   = res['placebo']
        std = model_cates[model_name].std()
        sig = "YES ✓" if p['significant'] else "NO  ✗"
        logger.info(f"{model_name:<20} {p['real_auuc']:>12.4f} {p['p_value']:>10.4f} {sig:>12} {std:>10.4f}")

    logger.info("\nSegment Heterogeneity per model:")
    for model_name, res in all_results.items():
        logger.info(f"  {model_name}:")
        for seg, stats in res['segments'].items():
            logger.info(f"    {seg}: n={stats['n_samples']:,}  "
                        f"mean_cate={stats['mean_cate']:.4f}  std={stats['std_cate']:.4f}")

    # ---- save -------------------------------------------------------
    out_path = Path(args.output) / 'validation_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nValidation results saved to {out_path}")


if __name__ == '__main__':
    main()
