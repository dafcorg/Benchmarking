#!/bin/bash

# Stop the script on error / # Para o script em caso de erro /
set -e  # Detener el script en caso de error

echo "Configuring the machine for MLPerf Training - SSD on Ubuntu 24.04"

#1 Update system and dependencies / Actualizar sistema y dependencias / Atualizar sistema e dependências
echo "Update system ------------"
sudo apt update && sudo apt upgrade -y
sudo apt install -y ca-certificates curl gnupg lsb-release wget tar git python3 python3-pip python3-venv software-properties-common || { echo "❌ Error: Failed to install system dependencies."; exit 1; }

echo ":) System updated and dependencies installed."

#2 Add NVIDIA repository and update / Agregar el repositorio de NVIDIA y actualizar / Adicione repositório NVIDIA e atualize
echo "Adding the NVIDIA repository------------"
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update

#3 Install NVIDIA and CUDA drivers / Instalar los controladores NVIDIA y CUDA / Instale drivers NVIDIA e CUDA
echo "Installing NVIDIA and CUDA drivers----------"
sudo apt install -y nvidia-driver-535 nvidia-utils-535 nvidia-cuda-toolkit || { echo "❌ Error: Failed to install CUDA and drivers."; exit 1; }

#4 Verify CUDA installation / Verificar instalación de CUDA / Verifique a instalação do CUDA
echo "Verifying CUDA --------"
nvidia-smi || { echo "❌ Error: NVIDIA-SMI doesn't detect the GPU. Check your NVIDIA drivers."; exit 1; }
nvcc --version || { echo "❌ Error: CUDA is not installed correctly."; exit 1; }
echo " :) CUDA OK"

#5 Install Docker 
echo "Installing Docker---------"
sudo apt-get install -y docker.io || { echo "❌ Error: Failed to install Docker."; exit 1; }
docker --version || { echo "❌ Error: Docker did not install correctly."; exit 1; }

echo " :) Docker installed successfully."

#6 Install NVIDIA-Docker
echo "Installing NVIDIA-Docker--------"
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-docker-keyring.gpg \
   && echo "deb [signed-by=/usr/share/keyrings/nvidia-docker-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/ubuntu24.04/$(dpkg --print-architecture) /" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-docker2 || { echo "❌ Error: Failed to install NVIDIA-Docker."; exit 1; }
sudo systemctl restart docker

# Verify NVIDIA-Docker
docker run --rm --gpus all nvidia/cuda:12.3.0-base nvidia-smi || { echo "❌ Error: NVIDIA-Docker bug."; exit 1; }

echo ":) NVIDIA-Docker installed successfully."

#7 Clone the MLPerf Training repository 
echo "Cloning MLPerf Training-------"
git clone https://github.com/mlcommons/training.git ~/mlperf_training || { echo "❌ Error: Failed to clone MLPerf Training."; exit 1; }
cd ~/mlperf_training/single_stage_detector || { echo "❌ Error: The 'single_stage_detector' folder was not found."; exit 1; }

echo ":) MLPerf Training cloned successfully."

#8 Create folders for data and models
echo "Creating data folders and models---------"
mkdir -p ~/mlperf_data ~/mlperf_logs ~/mlperf_models || { echo "❌ Error: The required folders could not be created."; exit 1; }

echo ":) Folders created successfully."

#9 Build the Docker image
echo "Building the Docker image for MLPerf SSD---------"
sudo docker build -t mlperf/single_stage_detector . || { echo "❌ Error: Docker image build failed."; exit 1; }

echo ":) Docker image created successfully."

#10 Download the OpenImages-v6 dataset
echo "Downloading the OpenImages-v6 dataset--------"
cd ~/mlperf_training/single_stage_detector/scripts
pip install fiftyone || { echo "❌ Error: Failed to install fiftyone."; exit 1; }
./download_openimages_mlperf.sh -d ~/mlperf_data || { echo "❌ Error: Failed to download dataset."; exit 1; }

echo ":) Dataset downloaded successfully."

#11 Download the pre-trained model (ResNeXt50_32x4d)
echo "Downloading the pretrained model ResNeXt50_32x4d -----"
bash download_backbone.sh || { echo "❌ Error: Failed to download pre-trained model."; exit 1; }

echo ":) Pre-trained model downloaded successfully."


echo "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣠⣄⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⠶⠛⠉⠁⠀⠀⠀⠉⠛⠶⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⣤⣤⣤⣤⣤⣤⠶⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⢦⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣶⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠶⣤⣀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣀⣠⣾⡋⢻⡄⠀⣤⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⡀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠉⠛⢦⣄⠀⠀⠀
⠀⣠⣴⠖⠛⠋⠉⠁⠀⠹⣬⣻⠆⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⢠⡄⠀⠀⠀⠀⠀⠈⢷⠀⠀⠀⠀⠹⣆⣀⣀⡀⠀⠀⠀⠈⠳⢦⡀
⢸⡟⠉⣴⠀⠀⠀⠀⢀⣤⢟⣿⠀⠀⠀⠀⠀⠀⠀⠀⣆⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⣀⣼⠷⠀⠀⠀⢀⣽⠀⠈⠉⠛⠒⠶⠶⠒⠛⠁
⠈⠳⣆⡀⠀⣀⣤⣶⠿⠟⠋⠉⠳⢦⣤⣀⣠⡾⠋⠀⠀⢠⣾⣀⣤⡤⠶⠚⣿⡿⣿⡄⡄⣰⠖⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠈⠉⠉⠉⠁⠀⠀⠀⠀⠀⠀⢀⣴⣿⢀⠀⣠⡴⠚⠉⠀⠀⠀⠀⠀⠀⠈⢿⣧⣸⣇⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢻⣯⡾⢸⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀"
