#####################################
# Generic utility functions
#####################################

from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import DistributedOptimizer

def parallelize_model(model):
    """The name of this function is a bit of a misnomer - it implements
    data parallelism, but operates on the model object"""
    
    init_process_group(backend='nccl', world_size=torch.cuda.device_count(), init_method='...')

    model = DDP(model, device_ids=[i], output_device=i)