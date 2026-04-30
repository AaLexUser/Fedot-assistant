<div align="center">

<img src="./docs/FEDOT-ASSISTANT-logo.svg" width="600" alt="FEDOT.ASSISTANT logo">


[![Acknowledgement ITMO](https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/ITMO_badge.svg)](https://itmo.ru/)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AaLexUser/Fedot-assistant)

**LLM-based multi-AutoML Orchestrator.**
</div>


FEDOT.ASSISTANT is an LLM-based prototype for next-generation AutoML. It combines the power of Large Language Models with automated machine learning techniques to enhance data analysis and pipeline building processes.

## 📁 Workspace Layout

```
fedot-assistant/
├── packages/
│   ├── fedotllm/           # Core package library
│   └── shared/             # Shared
│
├── apps/
│   ├── frontend/           # Streamlit UI application
│   └── server/            # FastAPI backend server
├── examples/              # Example tasks and datasets
└── docs/                  # Documentation
```

## 🆕 What's New

- CAAFE integration: LLM-driven feature engineering for tabular classification tasks. Enabled by default via `feature_transformers.enabled_models: [CAAFE]`. Requires installing the optional dependency group and setting an API key (see below).

## 💾 Installation

1. Install uv (A fast Python package installer and resolver):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:

```bash
git clone https://github.com/AaLexUser/Fedot-assistant.git
cd Fedot-assistant
```

3. Create a virtual environment:

```bash
uv venv --python 3.10
source .venv/bin/activate  # On Unix/macOS
# Or on Windows:
# .venv\Scripts\activate
```

4. Install all dependencies:

```bash
# Install all workspace packages
uv sync --all-packages
```

## 🔧 Configuration

### Environment Setup

Set your OpenAI API key:

```bash
export FEDOTLLM_LLM_API_KEY="your-api-key-here"
```

Optional (CAAFE feature generation uses its own LLM settings; you can also put these into a `.env` file):

```bash
export CAAFE_LLM_API_KEY="your-api-key-here"
# Optional overrides (defaults work with the example config)
export CAAFE_LLM_MODEL="openai/gemini-2.0-flash"
export CAAFE_LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
```

### Configuration Options

The system uses YAML configuration files you can customize. The default configuration is located at `packages/fedotllm/src/fedotllm/configs/default.yaml`. You can create your own configuration file and specify it using the `--config-path` option.

## 🚀 Quick Start

### Running the Applications

#### Frontend (Streamlit UI)

```bash
fedotllm-ui
```

#### Backend (FastAPI Server)

```bash
fedotllm-server
```


#### CLI Usage

```bash
fedotllm run /path/to/your/task/directory

# Use specific presets
fedotllm run /path/to/your/task/directory --presets best_quality

# Custom configuration
fedotllm run /path/to/your/task/directory --config-path config.yaml

# Override specific settings
fedotllm run /path/to/your/task/directory -o automl.enabled=fedot,time_limit=7200
```

### Enable or configure CAAFE (optional)

CAAFE performs LLM-driven feature engineering and currently supports classification tasks only.

```bash
# Ensure optional deps are installed
uv sync --group caafe

# Run with CAAFE enabled (default), customizing parameters on the fly
fedotllm run /path/to/task \
  -o "feature_transformers.enabled_models=[CAAFE]" \
  -o "feature_transformers.models.CAAFE.num_iterations=5" \
  -o "feature_transformers.models.CAAFE.optimization_metric=roc" \
  -o "feature_transformers.models.CAAFE.eval_model=lightgdm"  # or tab_pfn
```

### Task Directory Structure

Your task directory should contain:

- Training data file (e.g., `train.csv`)
- Test data file (e.g., `test.csv`)
- Sample submission file (e.g., `sample_submission.csv`)
- Task description file (e.g., `descriptions.txt`)

### Quality Presets

- `medium_quality`: Fast execution with good performance
- `best_quality`: Maximum accuracy (default)

## 📦 Package Usage

You can also use the FedotLLM package directly in your Python code:

```python
from fedotllm import run_assistant

# Run a task programmatically
task, assistant = run_assistant(
    task_path="/path/to/task",
    config_overrides=["time_limit=600"],
    presets="best_quality",
)
```

## 🙌 Acknowledgement

Our implementation adapts code from [AutoGluon Assistant](https://github.com/autogluon/autogluon-assistant). We thank authors of this project for providing high quality open source code!
