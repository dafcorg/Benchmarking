import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
import argparse

def run(rank, world_size, matrix_size, topologia):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Simula carga de trabajo (matmul)
    A = torch.randn(matrix_size, matrix_size, device=device)
    B = torch.randn(matrix_size, matrix_size, device=device)

    torch.cuda.synchronize()
    start_compute = time.time()
    C = A @ B
    torch.cuda.synchronize()
    end_compute = time.time()

    # Comunicación simulada
    tensor = torch.ones(1, device=device) * rank
    dist.barrier()  # sincroniza antes de comunicación
    torch.cuda.synchronize()
    start_comm = time.time()

    if topologia == "ring":
        next_rank = (rank + 1) % world_size
        prev_rank = (rank - 1 + world_size) % world_size

        dist.send(tensor=tensor, dst=next_rank)
        recv_tensor = torch.zeros(1, device=device)
        dist.recv(tensor=recv_tensor, src=prev_rank)

    elif topologia == "star":
        if rank == 0:
            for i in range(1, world_size):
                recv_tensor = torch.zeros(1, device=device)
                dist.recv(tensor=recv_tensor, src=i)
            for i in range(1, world_size):
                dist.send(tensor=tensor, dst=i)
        else:
            dist.send(tensor=tensor, dst=0)
            recv_tensor = torch.zeros(1, device=device)
            dist.recv(tensor=recv_tensor, src=0)

    elif topologia == "all_to_all":
        recv_tensors = []
        for i in range(world_size):
            if i != rank:
                send_tensor = tensor.clone()
                dist.send(tensor=send_tensor, dst=i)
        for i in range(world_size):
            if i != rank:
                recv_tensor = torch.zeros(1, device=device)
                dist.recv(tensor=recv_tensor, src=i)
                recv_tensors.append(recv_tensor)

    torch.cuda.synchronize()
    end_comm = time.time()

    # Resultados
    compute_time = end_compute - start_compute
    comm_time = end_comm - start_comm
    total_time = compute_time + comm_time

    print(f"[Rank {rank}] Compute: {compute_time:.4f}s | Comm: {comm_time:.4f}s | Total: {total_time:.4f}s")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=4, help="Número de GPUs a simular")
    parser.add_argument("--matrix_size", type=int, default=4096, help="Tamaño de la matriz cuadrada (NxN)")
    parser.add_argument("--topologia", type=str, choices=["ring", "star", "all_to_all"], default="ring")
    args = parser.parse_args()

    mp.spawn(run, args=(args.gpus, args.matrix_size, args.topologia), nprocs=args.gpus, join=True)

if __name__ == "__main__":
    main()
