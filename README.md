![BSD-3-Clause License](https://img.shields.io/badge/license-BSD--3--Clause-blue)

# Adaptive Computing

The Adaptive Computing (AC) software stack supports goal-based computing, for which a simulation workload is created on the fly, adapting to the results of calculations. Application-specific code defines an objective, which may be to solve an optimization problem or to train a surrogate model with minimal uncertainty. Then, the AC driver decides where in the design parameter space to run simulations to best achieve that objective. This process is iterative and online; as new data is returned from simulations, the AC driver chooses new simulations to run. The AC driver can strategically run simulations on distributed hardware resources (including high performance computing machines, cloud resources, and edge devices) to maximize throughput and obey resource constraints.

AC also provides an **MCP (Model Context Protocol) server** (`ac_mcp/`) that exposes surrogate modelling and Bayesian optimisation as callable tools for AI agents. The MCP interface supports stateful simulation orchestration — an agent can create an experiment, submit evaluation or optimization runs asynchronously, poll for results, and query the trained surrogate for predictions — all through a standard HTTP/SSE tool API. Experiment state (dataset, trained surrogate) is persisted to a per-application **experiment registry**, allowing agents to build up a library of investigations, reason across past results, and plan future experiments.

## Citation

If you use this project, please cite:

K. P. Griffin et al., "Adaptive Computing for Scale-Up Problems" in *Computing in Science & Engineering*, vol. 27, no. 01, pp. 28-38, Jan.-March 2025, doi: 10.1109/MCSE.2025.3555589.

URL: https://doi.ieeecomputersociety.org/10.1109/MCSE.2025.3555589

This software corresponds to the National Laboratory of the Rockies (NLR) software record SWR-24-106. https://doi.org/10.11578/dc.20250414.5

## Setup Instructions

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) should be installed on your system. HPC users can try

```bash
module load conda
```

### Quick Setup

**1. Clone the repository:**

```bash
git clone https://github.com/NatLabRockies/AdaptiveComputing.git
cd AdaptiveComputing
```

**2. Create and activate the conda environment:**

```bash
mamba env create -f environment.yml
mamba activate AC
```

**3. Install AdaptiveComputing with core dependencies:**

```bash
pip install -e .
```

**4. Install deep learning framework (for neural network surrogates):**

```bash
pip install git+https://github.com/NatLabRockies/tf-melt.git#egg=tfmelt
```

**5. Install Hero (optional, for distributed computing):**

If you plan to use Hero distributed computing features:

```bash
pip install git+https://github.com/NLR-hero/hero.git
```

**5b. Install agentic AI and MCP dependencies (optional, for LLM-driven workflows):**

If you plan to use AC with LangGraph-based agentic workflows or the AC MCP server (e.g., LLM agents that reason over the parameter space and submit simulations autonomously):

```bash
pip install langchain-openai langchain-core langgraph pydantic typing_extensions fastmcp
```

These packages enable:
- **LangGraph** — stateful multi-step agent graphs
- **LangChain** — LLM provider abstraction (OpenAI, Azure OpenAI, Anthropic, etc.)
- **Pydantic** — structured LLM output parsing
- **FastMCP** — MCP server and client for exposing AC capabilities as agent-callable tools

Most AC users do not need these packages. They are intentionally kept separate from the core `environment.yml` to avoid imposing heavy ML framework dependencies on users who only need AC's optimization and surrogate modeling capabilities. As agentic AI features mature in AC, a dedicated extras install target (e.g., `pip install -e ".[agents]"`) may be added.

**6. Set up Hero credentials (Hero users only):**

Set Hero credentials as environment variables in your local shell session or in a local, untracked environment file.

If you prefer using the helper file, copy the template and edit it:

```bash
cp adaptive_computing/hero_utils/set_hero_env_vars_template.py adaptive_computing/hero_utils/set_hero_env_vars.py
nano adaptive_computing/hero_utils/set_hero_env_vars.py
```

Replace the placeholder values with your actual Hero credentials (ask your Hero admin for these values). Do not commit secrets.

**7. Run the tests:**

```bash
python -m pytest
```
   
## Examples

The `examples/` directory contains practical demonstrations of AC functionality:

### Getting Started - Hero Introduction

**Start here**: [examples/hero/](examples/hero/) provides a simple introduction to Hero framework concepts:
- **controller.py**: Demonstrates HeroDataset API with different sample addition methods
- **controller_noAC.py**: Shows direct Hero API usage with detailed queue monitoring
- **worker.py**: Simple local worker that processes tasks with basic calculations
- No HPC access required - runs entirely on your local machine
- Perfect for learning Hero concepts and testing your setup

### Advanced - HPC Production Workflows

**Scale up**: [examples/hero_HPC_managers/](examples/hero_HPC_managers/) demonstrates production HPC workflows:
- **Adaptive Computing strategies**: Offline training, offline inference, and online inference
- Automated HPC managers with SSH and SLURM integration
- Real molecular dynamics simulations for conductivity calculation
- Multi-cluster job distribution and error handling
- Requires HPC access and SSH configuration

### Other Examples

- [examples/bayesian_1d_sf/](examples/bayesian_1d_sf/): Single-fidelity Bayesian optimization with 4 surrogate model options:
  - **SMT_GP**: Surrogate Modeling Toolbox's implementation of Gaussian Process (recommended starting point). Also, supports multi-fidelity.
  - **SOOGO_GP**: Alternative Gaussian Process implementation with different optimization algorithms
  - **TFMELT_BNN**: TensorFlow-based Bayesian Neural Network for uncertainty quantification
  - **TFMELT_MDN**: TensorFlow-based Mixture Density Network for multi-modal uncertainty
  - Neural network approaches (BNN/MDN) excel with larger datasets and complex functions
- [examples/bayesian_2d_sf/](examples/bayesian_2d_sf/): Multi-dimensional optimization
- [examples/cpp_embedding/](examples/cpp_embedding/): C++ integration
- [examples/query_sf/](examples/query_sf/): Simple query examples

**Recommended learning path**: Start with `examples/hero/` to understand the basics, then explore `examples/hero_HPC_managers/` for production deployment.

## Developer instructions

Development workflows, testing expectations, and pull request guidelines are documented in CONTRIBUTING.md.

For security vulnerability reporting, see SECURITY.md.

## License

This project is licensed under the BSD-3-Clause license. See LICENSE.txt for details.
