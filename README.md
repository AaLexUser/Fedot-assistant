# FEDOT.ASSISTANT

<p align="center">
  <img src="./docs/FEDOT-ASSISTANT-logo.svg" width="600" alt="FEDOT.ASSISTANT logo">
</p>

[![Acknowledgement ITMO](https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/ITMO_badge.svg)](https://itmo.ru/)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AaLexUser/Fedot-assistant)

FEDOT.ASSISTANT is an LLM-based prototype for next-generation AutoML. It combines the power of Large Language Models with automated machine learning techniques to enhance data analysis and pipeline building processes.

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

3. Create a new virtual environment and activate it:

```bash
uv venv --python 3.10
source .venv/bin/activate  # On Unix/macOS
# Or on Windows:
# .venv\Scripts\activate
```

4. Install dependencies:

```bash
uv sync
```

## 🔧 Configuration

### Environment Setup

Set your OpenAI API key:

```bash
export FEDOTLLM_LLM_API_KEY="your-api-key-here"
```

### Configuration Options

The system uses YAML configuration files you can customize. The default configuration is located at `fedotllm/configs/default.yaml`. You can create your own configuration file and specify it using the `--config-path` option.

## 🚀 Quick Start

### Basic Usage

```bash
# Run with default settings
fedotllm /path/to/your/task/directory

# Use specific presets
fedotllm /path/to/your/task/directory --presets best_quality

# Custom configuration
fedotllm /path/to/your/task/directory --config-path config.yaml

# Override specific settings
fedotllm /path/to/your/task/directory -o automl.enabled=fedot -o time_limit=7200
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

## 🙌 Acknowledgement
Our implementation adapts code from [AutoGluon Assistant](https://github.com/autogluon/autogluon-assistant). We thank authors of this project for providing high quality open source code!