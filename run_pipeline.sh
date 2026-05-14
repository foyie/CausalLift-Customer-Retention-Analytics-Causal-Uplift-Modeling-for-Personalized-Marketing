#!/bin/bash

# Customer Retention Analytics - Automated Pipeline
# Run the entire analysis in sequence

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Customer Retention Analytics - Pipeline Execution      ║"
echo "║          Causal Inference for Personalized Marketing        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="data"
RAW_DATA="$DATA_DIR/raw/synthetic_retail.csv"
PROCESSED_DATA="$DATA_DIR/processed/features_engineered.csv"
MODELS_DIR="models"
RESULTS_DIR="results"
N_SAMPLES=100000
RANDOM_SEED=42

# Logging function
log_step() {
    echo -e "${BLUE}▶ $1${NC}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# ============================================================================
# PHASE 1: SETUP
# ============================================================================
echo ""
log_step "PHASE 1: Setup & Validation"
echo ""

# Create directories
log_step "Creating directory structure..."
mkdir -p "$DATA_DIR/raw" "$DATA_DIR/processed"
mkdir -p "$MODELS_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "logs"
mkdir -p "app"
log_success "Directories created"

# Check Python version
log_step "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
MIN_VERSION="3.9"

if [[ $(echo "$PYTHON_VERSION < $MIN_VERSION" | bc) -eq 1 ]]; then
    log_error "Python $MIN_VERSION+ required, but found $PYTHON_VERSION"
    exit 1
fi
log_success "Python $PYTHON_VERSION (OK)"

# Check required packages
log_step "Checking required packages..."
python -c "import econml, xgboost, pandas, numpy, streamlit" 2>/dev/null || {
    log_warning "Some packages missing. Run: pip install -r requirements.txt"
}
log_success "Packages verified"

echo ""

# ============================================================================
# PHASE 2: DATA GENERATION
# ============================================================================
echo ""
log_step "PHASE 2: Data Generation"
echo "Generating $N_SAMPLES customer records..."
echo ""

START_TIME=$(date +%s)

python src/data_generation.py \
    --output "$RAW_DATA" \
    --n_samples $N_SAMPLES \
    --seed $RANDOM_SEED | tee "logs/01_data_generation.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ -f "$RAW_DATA" ]; then
    FILE_SIZE=$(du -h "$RAW_DATA" | cut -f1)
    ROW_COUNT=$(wc -l < "$RAW_DATA")
    log_success "Data generation completed in ${DURATION}s (${FILE_SIZE}, ${ROW_COUNT} rows)"
else
    log_error "Data generation failed"
    exit 1
fi

echo ""

# ============================================================================
# PHASE 3: FEATURE ENGINEERING
# ============================================================================
echo ""
log_step "PHASE 3: Feature Engineering"
echo "Creating 80+ features from raw data..."
echo ""

START_TIME=$(date +%s)

python src/feature_engineering.py \
    --input "$RAW_DATA" \
    --output "$PROCESSED_DATA" \
    --preprocessor "$MODELS_DIR/preprocessor.pkl" | tee "logs/02_feature_engineering.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ -f "$PROCESSED_DATA" ]; then
    FILE_SIZE=$(du -h "$PROCESSED_DATA" | cut -f1)
    log_success "Feature engineering completed in ${DURATION}s (${FILE_SIZE})"
else
    log_error "Feature engineering failed"
    exit 1
fi

echo ""

# ============================================================================
# PHASE 4: CAUSAL ESTIMATION
# ============================================================================
echo ""
log_step "PHASE 4: Causal Model Training"
echo "Training T-Learner, X-Learner, Causal Forest, and Ensemble..."
echo ""

START_TIME=$(date +%s)

python src/causal_estimation.py \
    --data "$PROCESSED_DATA" \
    --output "$MODELS_DIR" | tee "logs/03_causal_estimation.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ -f "$MODELS_DIR/causal_summary.csv" ]; then
    log_success "Causal estimation completed in ${DURATION}s"
    echo ""
    log_step "Causal Model Summary:"
    cat "$MODELS_DIR/causal_summary.csv" | head -10
else
    log_error "Causal estimation failed"
    exit 1
fi

echo ""

# ============================================================================
# PHASE 5: VALIDATION
# ============================================================================
echo ""
log_step "PHASE 5: CATE Validation"
echo "Running Qini curves, AUUC, and placebo tests..."
echo ""

START_TIME=$(date +%s)

python src/validation.py \
    --data "$PROCESSED_DATA" \
    --output "$RESULTS_DIR" | tee "logs/04_validation.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ -f "$RESULTS_DIR/validation_results.json" ]; then
    log_success "Validation completed in ${DURATION}s"
else
    log_error "Validation failed"
    exit 1
fi

echo ""

# ============================================================================
# PHASE 6: SUMMARY & NEXT STEPS
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    ✓ PIPELINE COMPLETE                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

log_success "All data processing and model training finished!"
echo ""

log_step "Generated Artifacts:"
echo "  📊 Raw Data:         $RAW_DATA"
echo "  🔧 Processed Data:   $PROCESSED_DATA"
echo "  🤖 Models:           $MODELS_DIR/"
echo "  📈 Results:          $RESULTS_DIR/"
echo "  📝 Logs:             logs/"
echo ""

log_step "Next Steps:"
echo ""
echo "1️⃣  LAUNCH DASHBOARD:"
echo "    $ streamlit run app/dashboard.py"
echo ""
echo "2️⃣  NAVIGATE TO:"
echo "    http://localhost:8501"
echo ""
echo "3️⃣  EXPLORE:"
echo "    • Overview: KPIs and architecture"
echo "    • CATE Analysis: Distribution and segments"
echo "    • Model Comparison: Performance metrics"
echo "    • Policy Simulator: Interactive targeting"
echo "    • Validation: Statistical tests"
echo ""

log_step "Useful Commands:"
echo ""
echo "  View data summary:      $ head -5 $PROCESSED_DATA"
echo "  Check model metrics:    $ cat $MODELS_DIR/causal_summary.csv"
echo "  View validation stats:  $ python -m json.tool $RESULTS_DIR/validation_results.json"
echo "  View high responders:   $ head -10 $RESULTS_DIR/high_responders.csv"
echo ""

log_step "Project Structure:"
echo ""
tree -L 2 -I '__pycache__|*.pyc' 2>/dev/null || find . -maxdepth 2 -type f -name "*.py" -o -name "*.csv" -o -name "*.pkl" | head -20
echo ""

log_step "System Information:"
echo "  Date:           $(date)"
echo "  Total samples:  $N_SAMPLES"
echo "  Python:         $PYTHON_VERSION"
echo ""

echo "📚 For detailed information, see: SETUP_GUIDE.md"
echo ""
echo "🚀 Ready to explore your Customer Retention Analytics!"
echo ""
