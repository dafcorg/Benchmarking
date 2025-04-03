

https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch.distributed as dist; print(dist.is_nccl_available())" #true
pip install nvidia-ml-py3



python simulador_topologia.py --gpus 4 --matrix_size 4096 --topologia ring
