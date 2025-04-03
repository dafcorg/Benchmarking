import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
import argparse
import csv
import random
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("log.txt")
CSV_PATH = Path("tiempos.csv")

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    with open(LOG_PATH, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")

def write_csv(rank, compute_time, comm_time, total_time):
    header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        if header:
            writer.writerow(["rank", "compute_time", "comm_time", "total_time"])
        writer.writerow([rank, f"{compute_time:.4f}", f"{comm_time:.4f}", f"{total_time:.4f}"])

def run(rank, world_size, matrix_size, topologia, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    log(f"[Rank {rank}] Inicializando proceso...")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(0) if device.type == "cuda" else None
        log(f"[Rank {rank}] Usando dispositivo: {device}")

        A = torch.randn(matrix_size, matrix_size, device=device)
        B = torch.randn(matrix_size, matrix_size, device=device)

        torch.cuda.synchronize() if device.type == "cuda" else None
        log(f"[Rank {rank}] Iniciando multiplicación de matrices...")
        start_compute = time.time()
        C = A @ B
        torch.cuda.synchronize() if device.type == "cuda" else None
        end_compute = time.time()
        log(f"[Rank {rank}] Multiplicación terminada.")

        tensor = torch.ones(1, device=device) * rank
        dist.barrier()
        torch.cuda.synchronize() if device.type == "cuda" else None
        log(f"[Rank {rank}] Iniciando comunicación ({topologia})...")
        start_comm = time.time()

        if topologia == "ring":
            next_rank = (rank + 1) % world_size
            prev_rank = (rank - 1 + world_size) % world_size
            dist.send(tensor=tensor, dst=next_rank)
            recv_tensor = torch.zeros(1, device=device)
            dist.recv(tensor=recv_tensor, src=prev_rank)
            log(f"[Rank {rank}] Recibido de Rank {prev_rank}: {recv_tensor.item()}")

        elif topologia == "star":
            if rank == 0:
                for i in range(1, world_size):
                    recv_tensor = torch.zeros(1, device=device)
                    dist.recv(tensor=recv_tensor, src=i)
                    log(f"[Rank 0] Recibido de Rank {i}: {recv_tensor.item()}")
                for i in range(1, world_size):
                    dist.send(tensor=tensor, dst=i)
            else:
                dist.send(tensor=tensor, dst=0)
                recv_tensor = torch.zeros(1, device=device)
                dist.recv(tensor=recv_tensor, src=0)
                log(f"[Rank {rank}] Recibido de Rank 0: {recv_tensor.item()}")

        elif topologia == "all_to_all":
            for i in range(world_size):
                if i != rank:
                    dist.send(tensor=tensor.clone(), dst=i)
            for i in range(world_size):
                if i != rank:
                    recv_tensor = torch.zeros(1, device=device)
                    dist.recv(tensor=recv_tensor, src=i)
                    log(f"[Rank {rank}] Recibido de Rank {i}: {recv_tensor.item()}")

        torch.cuda.synchronize() if device.type == "cuda" else None
        end_comm = time.time()
        log(f"[Rank {rank}] Comunicación terminada.")

        compute_time = end_compute - start_compute
        comm_time = end_comm - start_comm
        total_time = compute_time + comm_time
        log(f"[Rank {rank}] ⏱️ Tiempos -> Cómputo: {compute_time:.4f}s | Comunicación: {comm_time:.4f}s | Total: {total_time:.4f}s")
        write_csv(rank, compute_time, comm_time, total_time)

    finally:
        log(f"[Rank {rank}] Finalizando proceso y liberando recursos.")
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--matrix_size", type=int, default=2048)
    parser.add_argument("--topologia", type=str, choices=["ring", "star", "all_to_all"], default="ring")
    args = parser.parse_args()

    # Limpiar archivos previos
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    if CSV_PATH.exists():
        CSV_PATH.unlink()

    # Elegir puerto aleatorio libre entre 12350-12400
    port = random.randint(12350, 12400)
    log(f"Usando puerto dinámico: {port}")

    mp.spawn(run, args=(args.gpus, args.matrix_size, args.topologia, port), nprocs=args.gpus, join=True)

if __name__ == "__main__":
    main()
