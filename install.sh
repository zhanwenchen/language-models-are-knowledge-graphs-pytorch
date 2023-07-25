GNUMAKEFLAGS=-j12 MAKEFLAGS=-j12 HOMEBREW_VERBOSE=1 arch -arm64 brew install --build-from-source rust apache-arrow openblas

conda create -n llm2kg
conda activate llm2kg
conda install python=3.10

MPLSETUPCFG=/Users/zhanwenchen/mplsetup.cfg pip install --no-binary :all: --only-binary torch,torchvision,torchaudio,flair transformers datasets torch torchvision torchaudio tqdm radboud-el

# Spacy
# For Linux:
LDFLAGS="-L/opt/homebrew/opt/openblas/lib" CPPFLAGS="-I/opt/homebrew/opt/openblas/include" PKG_CONFIG_PATH="/opt/homebrew/opt/openblas/lib/pkgconfig" pip install --no-binary :all: spacy
# For Apple, install spacy like so:
# pip install -U pip setuptools wheel
pip install --no-binary :all: 'spacy[apple]'

python -m spacy download en_core_web_md

# TOKENIZERS_PARALLELISM=false python extract.py examples/bob_dylan.txt bert-large-cased-bob_dynlan.jsonl --language_model bert-large-cased --use_cuda false
TOKENIZERS_PARALLELISM=true python extract.py examples/bob_dylan.txt bert-large-cased-bob_dynlan.jsonl --language_model bert-large-cased --device mps