#!/bin/bash

set -e  # Detener el script en caso de error

echo "🔧 Preparando el entorno para MLPerf Training - SSD en Ubuntu 24.04..."
echo "🔧 Preparing the environment for MLPerf Training - SSD on Ubuntu 24.04..."
echo "🔧 Preparando o ambiente para o MLPerf Training - SSD no Ubuntu 24.04..."

# 1️⃣ Instalar dependencias básicas
echo "📦 Instalando dependencias del sistema..."
echo "📦 Installing system dependencies..."
echo "📦 Instalando dependências do sistema..."
sudo apt update && sudo apt install -y \
    python3 python3-pip python3-venv \
    ca-certificates curl gnupg lsb-release git wget tar

# 2️⃣ Instalar CUDA 12.3 (Compatible con Ubuntu 24.04)
echo "🎛️ Instalando CUDA 12.3..."
echo "🎛️ Installing CUDA 12.3..."
echo "🎛️ Instalando CUDA 12.3..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-ubuntu2404-12-3-local_12.3.0-535.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-3-local_12.3.0-535.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# 3️⃣ Verificar instalación de CUDA
echo "✅ CUDA instalado correctamente."
echo "✅ CUDA installed correctly."
echo "✅ CUDA instalado corretamente."
nvidia-smi  # Verifica que la GPU está activa

# 4️⃣ Instalar Docker
echo "🐳 Instalando Docker..."
echo "🐳 Installing Docker..."
echo "🐳 Instalando Docker..."
sudo apt-get install -y docker.io

# 5️⃣ Instalar NVIDIA-Docker
echo "🚀 Instalando NVIDIA-Docker..."
echo "🚀 Installing NVIDIA-Docker..."
echo "🚀 Instalando NVIDIA-Docker..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-docker-keyring.gpg \
   && echo "deb [signed-by=/usr/share/keyrings/nvidia-docker-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/ubuntu24.04/$(dpkg --print-architecture) /" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-docker2
sudo systemctl restart docker

# 6️⃣ Clonar el repositorio MLPerf Training
echo "📂 Clonando MLPerf Training..."
echo "📂 Cloning MLPerf Training..."
echo "📂 Clonando MLPerf Training..."
git clone https://github.com/mlcommons/training.git mlperf_training
cd mlperf_training/single_stage_detector

# 7️⃣ Construir la imagen Docker
echo "🐳 Construyendo la imagen Docker para SSD..."
echo "🐳 Building the Docker image for SSD..."
echo "🐳 Construindo a imagem Docker para SSD..."
docker build -t mlperf/single_stage_detector .

# 8️⃣ Descargar el dataset OpenImages-v6
echo "📥 Descargando el dataset OpenImages-v6..."
echo "📥 Downloading the OpenImages-v6 dataset..."
echo "📥 Baixando o dataset OpenImages-v6..."
cd scripts
pip install fiftyone  # Dependencia necesaria para descargar el dataset
./download_openimages_mlperf.sh -d ~/mlperf_data

# 9️⃣ Descargar el modelo backbone preentrenado (ResNeXt50_32x4d)
echo "📥 Descargando el modelo preentrenado ResNeXt50_32x4d..."
echo "📥 Downloading the pre-trained model ResNeXt50_32x4d..."
echo "📥 Baixando o modelo pré-treinado ResNeXt50_32x4d..."
bash download_backbone.sh

echo "✅ ¡Alistamiento completado! Ahora puedes ejecutar el entrenamiento con run.sh"
echo "✅ Setup complete! You can now run the training with run.sh"
echo "✅ Configuração concluída! Agora você pode executar o treinamento com run.sh"
