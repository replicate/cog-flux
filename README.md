# FLUX

## Introduction

Flux is a rectified flow transformer.

## Installation

Install via

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e '.[all]'
```

### Models

We are offering three models:
- `flux-schnell` step-distilled variant (quantized and non-quantized)
- `flux-dev` guidance-distilled variant (quantized and non-quantized)
- `flux-pro` the base model, available via API

## Usage

For interactive sampling run
```bash
python -m flux --name <name> --loop
```
Or to generate a single sample run
```bash
python -m flux --name <name> \
  --height <height> --width <width> \
  --prompt "<prompt>"
```

To use the quantized model, add the flag `--quantize_flow` to the above commands.

Run the streamlit demo via
```bash
streamlit run demo_st.py
```
