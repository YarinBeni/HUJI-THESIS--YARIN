#!/bin/bash

# Setup script for SAELens/TransformerLens environment
echo "🔧 Setting up conda environment for SAELens analysis..."

# Create conda environment from yml
conda env create -f environment.yml

echo "✅ Environment created. Activate with: conda activate torso-sae"
echo ""
echo "To activate and test:"
echo "  conda activate torso-sae"
echo "  python -c 'import transformer_lens; import sae_lens; print(\"✅ All packages installed\")'"
