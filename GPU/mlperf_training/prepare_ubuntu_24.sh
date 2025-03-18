#!/bin/bash

set -e  # Detener el script en caso de error

echo "🔧 Configurando la máquina para MLPerf Training - SSD en Ubuntu 24.04..."

# 1️⃣ Actualizar sistema y dependencias
echo "📦 Actualizando el sistema..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y ca-certificates curl gnupg lsb-release wget tar git python3 python3-pip python3-venv || { echo "❌ Error: Fallo al instalar dependencias del sistema."; exit 1; }

echo "✅ Sistema actualizado y dependencias instaladas."

# 2️⃣ Instalar CUDA 12.3 (Compatible con Ubuntu 24.04)
echo "🎛️ Instalando CUDA 12.3..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin || { echo "❌ Error: Fallo al descargar el archivo de configuración de CUDA."; exit 1; }
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-ubuntu2404-12-3-local_12.3.0-535.86.10-1_amd64.deb || { echo "❌ Error: Fallo al descargar el paquete de CUDA."; exit 1; }
sudo dpkg -i cuda-repo-ubuntu2404-12-3-local_12.3.0-535.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update && sudo apt-get -y install cuda || { echo "❌ Error: Fallo al instalar CUDA."; exit 1; }

# 3️⃣ Verificar instalación de CUDA
echo "✅ CUDA instalado correctamente."
nvidia-smi || { echo "❌ Error: NVIDIA-SMI no detecta la GPU. Verifica los drivers de NVIDIA."; exit 1; }

# 4️⃣ Instalar Docker
echo "🐳 Instalando Docker..."
sudo apt-get install -y docker.io || { echo "❌ Error: Fallo al instalar Docker."; exit 1; }
docker --version || { echo "❌ Error: Docker no se instaló correctamente."; exit 1; }

echo "✅ Docker instalado correctamente."

# 5️⃣ Instalar NVIDIA-Docker
echo "🚀 Instalando NVIDIA-Docker..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-docker-keyring.gpg \
   && echo "deb [signed-by=/usr/share/keyrings/nvidia-docker-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/ubuntu24.04/$(dpkg --print-architecture) /" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-docker2 || { echo "❌ Error: Fallo al instalar NVIDIA-Docker."; exit 1; }
sudo systemctl restart docker

# Verificar NVIDIA-Docker
docker run --rm --gpus all nvidia/cuda:12.3.0-base nvidia-smi || { echo "❌ Error: Fallo en NVIDIA-Docker."; exit 1; }

echo "✅ NVIDIA-Docker instalado correctamente."

# 6️⃣ Clonar el repositorio MLPerf Training
echo "📂 Clonando MLPerf Training..."
git clone https://github.com/mlcommons/training.git ~/mlperf_training || { echo "❌ Error: Fallo al clonar MLPerf Training."; exit 1; }
cd ~/mlperf_training/single_stage_detector || { echo "❌ Error: No se encontró la carpeta 'single_stage_detector'."; exit 1; }

echo "✅ MLPerf Training clonado correctamente."

# 7️⃣ Crear carpetas para datos y modelos
echo "📁 Creando carpetas de datos y modelos..."
mkdir -p ~/mlperf_data ~/mlperf_logs ~/mlperf_models || { echo "❌ Error: No se pudieron crear las carpetas necesarias."; exit 1; }

echo "✅ Carpetas creadas correctamente."

# 8️⃣ Construir la imagen Docker
echo "🐳 Construyendo la imagen Docker para MLPerf SSD..."
sudo docker build -t mlperf/single_stage_detector . || { echo "❌ Error: Fallo en la construcción de la imagen Docker."; exit 1; }

echo "✅ Imagen Docker creada correctamente."

# 9️⃣ Descargar el dataset OpenImages-v6
echo "📥 Descargando el dataset OpenImages-v6..."
cd ~/mlperf_training/single_stage_detector/scripts
pip install fiftyone || { echo "❌ Error: Fallo al instalar fiftyone."; exit 1; }
./download_openimages_mlperf.sh -d ~/mlperf_data || { echo "❌ Error: Fallo al descargar el dataset."; exit 1; }

echo "✅ Dataset descargado correctamente."

# 🔟 Descargar el modelo preentrenado (ResNeXt50_32x4d)
echo "📥 Descargando el modelo preentrenado ResNeXt50_32x4d..."
bash download_backbone.sh || { echo "❌ Error: Fallo al descargar el modelo preentrenado."; exit 1; }

echo "✅ Modelo preentrenado descargado correctamente."

echo "🎯 ¡Configuración completa! Ahora puedes ejecutar el entrenamiento con run.sh"
