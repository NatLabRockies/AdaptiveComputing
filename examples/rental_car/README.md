# Rental Car Active Learning Example

This example demonstrates active learning techniques applied to the NLR rental car optimization model using both SMT and SOOGO surrogates. The goal is to efficiently explore the 4D parameter space and quantify and reduce uncertainty using adaptive sampling.

## Overview

The rental car model optimizes EV fleet management across four key parameters:
- **utility_rate**: Charging behavior (Moderate vs Aggressive)
- **solution**: Storage configuration (Grid, Storage 25%, 50%, 75%, 100%)
- **demand**: Daily EV demand (10 to 10,000 vehicles)
- **soc_mean**: State of charge target (25%, 35%, 45%, 55%)

Two surrogate approaches are available:
- **SMT surrogate** (`active_learning_smt.py`): Uses native mixed types (categorical + ordered variables)
- **SOOGO surrogate** (`active_learning_soogo.py`): Uses continuous variables with interpolation

## Prerequisites

### 1. AdaptiveComputing Environment Setup

First, ensure you have the AdaptiveComputing environment properly set up:

```bash
# Navigate to the AdaptiveComputing root directory
cd /path/to/AdaptiveComputing_1.0/AdaptiveComputing

# Create and activate the conda environment
conda env create -f environment.yaml
conda activate AC

# Install AdaptiveComputing in development mode
pip install -e .
```

### 2. NLR Rental Car Model Setup

Clone the required NLR genesis model repository:

```bash
# Clone the rental car model repository
git clone https://github.NLR.gov/Genesis/model-aeroportal-rental-car

# The repository should be cloned to:
# /path/to/AdaptiveComputing_1.0/AdaptiveComputing/examples/rental_car/model-aeroportal-rental-car
```

### 3. Environment Configuration

The rental car model requires specific environment variables for HERO backend access. 

#### Step 1: Create environment file
Copy the template and edit with your credentials:

```bash
# From the rental_car directory
cp env_template.txt env.txt
```

#### Step 2: Edit env.txt
Open `env.txt` and fill in your actual NLR credentials:

```bash
nano env.txt  # or use your preferred editor
```

Replace the placeholder values with your actual:
- NLR username  
- NLR password
- API token (if required)
- Any other required credentials

#### Step 3: Source environment variables
Before running any active learning scripts:

```bash
# Source the environment file
source env.txt

# Verify variables are set (optional)
echo $NLR_USERNAME  # Should show your username
```

**Important**: 
- Never commit `env.txt` to version control
- The `env.txt` file is already in `.gitignore`
- Always source `env.txt` before running the active learning scripts

## Usage

### SMT Surrogate (Recommended for Discrete Optimization)

```bash
# Basic run with default parameters
python active_learning_smt.py

# Custom configuration
python active_learning_smt.py --n-initial 15 --n-bayes-opt 20
```

**Features:**
- Native support for mixed variable types
- Exact evaluation at discrete points  
- Global uncertainty tracking over 520 discrete combinations
- No interpolation required

### SOOGO Surrogate (Continuous Variable Approach)

```bash
# Basic run with default parameters  
python active_learning_soogo.py

# Custom configuration
python active_learning_soogo.py --n-initial 15 --n-bayes-opt 20
```

**Features:**
- Continuous variables with interpolation
- Bilinear interpolation for smooth cost surfaces
- Sampled uncertainty tracking over continuous space
- Better for sensitivity analysis

## Command Line Arguments

Both scripts support the following arguments:

- `--n-initial N`: Number of initial LHS samples (default: 10)
- `--n-bayes-opt N`: Number of Bayesian optimization iterations (default: 15)

Example:
```bash
python active_learning_smt.py --n-initial 20 --n-bayes-opt 25
```

## Output

Each script generates:

1. **Console Output**: 
   - Iteration-by-iteration variance reduction statistics
   - Total evaluation counts and reduction percentages
   - Performance summaries

2. **Visualization**: 
   - Sample distribution plot with legend (initial vs adaptive samples)
   - Relative average variance reduction over iterations
   - Relative maximum variance reduction over iterations

3. **Return Value**: 
   - Trained `ActiveLoopDriver` object for further analysis

## Understanding the Results

### Sample Distribution Plot
- **Triangles**: Initial LHS samples  
- **Circles**: Adaptive samples
- **Blue**: Moderate utility rate
- **Red**: Aggressive utility rate

### Variance Reduction Plots
- Show how uncertainty decreases as more samples are added
- Values relative to initial uncertainty (1.0 = initial, 0.0 = no uncertainty)
- Steeper drops indicate more efficient uncertainty reduction

## File Organization

```
rental_car/
├── README.md                    # This file
├── env_template.txt            # Template for environment variables
├── env.txt                     # Your actual credentials (do not commit!)
├── active_learning_smt.py      # SMT surrogate implementation
├── active_learning_soogo.py    # SOOGO surrogate implementation
└── model-aeroportal-rental-car/ # Cloned NLR model repository
```

## Important Notes

- **Each evaluation calls the actual HERO simulation** - expect longer runtimes
- **HERO backend requires NLR network access** - ensure proper VPN connection if needed
- **Start with small sample sizes** for testing (e.g., `--n-initial 5 --n-bayes-opt 5`)
- **SMT approach is generally more efficient** for this discrete optimization problem
- **Always source env.txt first** before running any scripts

## Troubleshooting

### Common Issues

1. **Environment variables not set**:
   ```bash
   source env.txt
   ```

2. **HERO connection errors**:
   - Check VPN connection to NLR network
   - Verify credentials in `env.txt`
   - Test with smaller sample sizes first

3. **Import errors**:
   ```bash
   # Ensure AdaptiveComputing is properly installed
   pip install -e /path/to/AdaptiveComputing
   
   # Verify model repository is cloned in the right location
   ls model-aeroportal-rental-car/
   ```

4. **Memory issues with large sample sizes**:
   - Reduce `--n-initial` and `--n-bayes-opt` values
   - Consider using the SOOGO approach for continuous optimization

## Next Steps

After running the examples:

1. **Analyze results**: Compare SMT vs SOOGO performance for your specific use case
2. **Customize parameters**: Adjust variable ranges or add new parameters  
3. **Extend acquisition functions**: Try different acquisition strategies
4. **Scale up**: Increase sample sizes for production runs
5. **Integration**: Incorporate results into broader optimization workflows

For questions or issues, refer to the AdaptiveComputing documentation or contact the development team.