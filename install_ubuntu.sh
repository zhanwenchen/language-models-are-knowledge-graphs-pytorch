conda create -n llm2kg
conda activate llm2kg
conda install python=3.10

pip install --no-binary :all: --only-binary torch,torchvision,torchaudio,flair transformers datasets torch torchvision torchaudio tqdm radboud-el

# Spacy
# For Ubuntu:
pip install --no-binary :all: spacy cupy-cuda11x
python -m cupyx.tools.install_library --cuda 11.x --library cutensor
python -m cupyx.tools.install_library --cuda 11.x --library nccl
python -m cupyx.tools.install_library --cuda 11.x --library cudnn

python -m spacy download en_core_web_md

# TOKENIZERS_PARALLELISM=false python extract.py examples/bob_dylan.txt bert-large-cased-bob_dynlan.jsonl --language_model bert-large-cased --use_cuda false
TOKENIZERS_PARALLELISM=true python extract.py examples/bob_dylan.txt bert-large-cased-bob_dynlan.jsonl --language_model bert-large-cased --device mps