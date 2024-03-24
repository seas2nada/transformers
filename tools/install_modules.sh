. ./activate_python.sh
pip install --upgrade git+https://github.com/huggingface/transformers.git
pip install datasets[audio] evaluate jiwer bitsandbytes
pip install accelerate -U
pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git@main
