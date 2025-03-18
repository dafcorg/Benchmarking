#!/bin/bash

set -e  # Detener el script en caso de error / Stop script on error / Parar o script em caso de erro

# Número de repeticiones / Number of repetitions / Número de repetições
NUM_RUNS=5  

# Directorios base / Base directories / Diretórios base
BASE_DATA_DIR=~/mlperf_data
BASE_LOG_DIR=~/mlperf_logs
BASE_MODEL_DIR=~/mlperf_models

# Crear directorios base si no existen / Create base directories if they don't exist / Criar diretórios base se não existirem
mkdir -p $BASE_LOG_DIR $BASE_MODEL_DIR

# Archivo de registro de tiempos y métricas / Log file for times and metrics / Arquivo de registro de tempos e métricas
METRICS_FILE="$BASE_LOG_DIR/training_metrics.csv"
echo "Run, Start Time, End Time, Duration (s), Final mAP" > $METRICS_FILE

# Loop de ejecuciones / Execution loop / Loop de execuções
for i in $(seq 1 $NUM_RUNS); do
  echo "🚀 Ejecutando entrenamiento #$i de $NUM_RUNS..."
  echo "🚀 Running training #$i of $NUM_RUNS..."
  echo "🚀 Executando treinamento #$i de $NUM_RUNS..."

  # Directorios específicos para cada ejecución / Specific directories for each execution / Diretórios específicos para cada execução
  RUN_LOG_DIR="$BASE_LOG_DIR/run_$i"
  RUN_MODEL_DIR="$BASE_MODEL_DIR/run_$i"
  
  mkdir -p $RUN_LOG_DIR $RUN_MODEL_DIR

  # Registrar tiempo de inicio / Log start time / Registrar tempo de início
  START_TIME=$(date +%s)

  # Ejecutar entrenamiento dentro de Docker / Run training inside Docker / Executar treinamento dentro do Docker
  docker run --rm -it \
    --gpus=all \
    --ipc=host \
    -v $BASE_DATA_DIR:/datasets/open-images-v6-mlperf \
    -v $RUN_LOG_DIR:/workspace/logs \
    -v $RUN_MODEL_DIR:/workspace/models \
    mlperf/single_stage_detector bash -c "
      cd /workspace/single_stage_detector;
      source config_DGXA100_001x08x032.sh;
      ./run_and_time.sh
    "

  # Registrar tiempo de finalización / Log end time / Registrar tempo de finalização
  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))

  # Evaluar el modelo entrenado y extraer la métrica mAP / Evaluate trained model and extract mAP metric / Avaliar o modelo treinado e extrair a métrica mAP
  FINAL_MAP=$(docker run --rm -it \
    --gpus=all \
    -v $RUN_MODEL_DIR:/workspace/models \
    mlperf/single_stage_detector bash -c "
      cd /workspace/single_stage_detector;
      python train.py --eval-batch-size 16 --test-only --resume /workspace/models/checkpoint.pth | grep -oP 'mAP: \K[0-9.]+'
    ")

  # Guardar resultados en CSV / Save results in CSV / Salvar resultados em CSV
  echo "$i, $(date -d @$START_TIME +'%Y-%m-%d %H:%M:%S'), $(date -d @$END_TIME +'%Y-%m-%d %H:%M:%S'), $DURATION, $FINAL_MAP" >> $METRICS_FILE

  echo " Entrenamiento #$i completado. Duración: $DURATION segundos. mAP: $FINAL_MAP"
  echo " Training #$i completed. Duration: $DURATION seconds. mAP: $FINAL_MAP"
  echo " Treinamento #$i concluído. Duração: $DURATION segundos. mAP: $FINAL_MAP"
done

echo " Todas las ejecuciones han finalizado. Resultados en: $METRICS_FILE"
echo " All executions have finished. Results in: $METRICS_FILE"
echo " Todas as execuções foram concluídas. Resultados em: $METRICS_FILE"
