import os
import time
import cProfile
import pstats
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


def run(rank, world_size, data, max_epochs):
    # Set up the environment variables for master address and port
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Create the model and move it to the appropriate device
    model = SimpleModel()
    model = DDP(model)

    # Create the dataset and dataloader
    dataset = TensorDataset(*data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=256, sampler=sampler, num_workers=4)

    # Training loop
    for epoch in range(max_epochs):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            optimizer.zero_grad()
            outputs = model(batch[0])
            loss_fn = nn.MSELoss()
            loss = loss_fn(outputs, batch[1])
            loss.backward()
            optimizer.step()

    # Clean up
    dist.destroy_process_group()


def train_worker(rank, world_size, data, max_epochs):
    torch.set_num_threads(1)  # Limit each worker to a single thread
    run(rank, world_size, data, max_epochs)


def main():
    data = (torch.randn(1024*4, 10), torch.randn(1024*4, 1))

    n_cpus = 4
    world_size = n_cpus

    start_time = time.time()
    mp.spawn(train_worker,
             args=(world_size, data, 10),
             nprocs=n_cpus,
             join=True)

    end_time = time.time()
    print(f"Training time with {n_cpus} CPUs: {end_time - start_time} seconds")


if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(10)  # Print top 10 functions by cumulative time
