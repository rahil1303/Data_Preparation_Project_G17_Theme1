# AutoML-Clean: Autonomous Data Quality Monitoring System

Production-grade data quality monitoring and remediation for ML pipelines.

## Overview

This system implements a two-layer validation approach for detecting and fixing data corruption in real-time:
- **Layer 1**: Fast, obvious checks (schema, types, ranges)
- **Layer 2**: Deep pattern analysis (triggered on accuracy drops >15%)

Tested across 3 review datasets: Amazon, Goodreads, Steam.

## Architecture
```
Incoming Data → Layer 1 (Quick) → Model → Accuracy Check
                                              ↓
                                    Dropped >15%?
                                              ↓
                                    Layer 2 (Deep Analysis)
                                              ↓
                                    Pattern Detection → Fix
```

## Project Structure
```
.
├── src/
│   ├── validation/
│   │   ├── layer1_quick.py      # Fast validation checks
│   │   └── layer2_deep.py       # Deep pattern analysis
│   ├── corruption/
│   │   └── inject.py            # Corruption simulation
│   ├── models/
│   │   └── train.py             # Model training
│   └── utils/
│       ├── metrics.py           # Evaluation metrics
│       └── logger.py            # Logging utilities
├── data/
│   ├── raw/                     # Original datasets
│   ├── processed/               # Clean data
│   └── corrupted/               # Corrupted samples
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Unit tests
├── logs/                        # System logs
├── results/                     # Plots and metrics
├── main.py                      # Main entry point
└── requirements.txt
```

## Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start
```bash
# Run the monitoring system
python main.py --config configs/default.yaml

# Or run in notebook
jupyter notebook notebooks/demo.ipynb
```

## Features

- ✅ Two-layer validation (fast + deep)
- ✅ Multi-dataset support (Amazon, Goodreads, Steam)
- ✅ 13 corruption types (JENGA + custom)
- ✅ Automated remediation strategies
- ✅ Real-time monitoring simulation
- ✅ Cross-dataset vulnerability analysis

## Corruption Types Supported

**JENGA-based:**
- Missing values (MCAR, MAR, MNAR)
- Swapped values
- Scaling errors
- Gaussian noise
- Encoding errors
- Broken characters

**Custom:**
- Label noise
- Duplicates
- Fake reviews
- Schema violations
- Outliers
- Temporal corruption
- Burst patterns

## Results

Coming soon...

## Team

- Rahil S
- [Team members]

## Course

Data Preparation 2026 - UvA Master Computer Science

## License

MIT
