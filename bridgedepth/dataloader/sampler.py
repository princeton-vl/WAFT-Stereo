from torch.utils.data.sampler import Sampler

from ..utils import dist_utils as comm


class InferenceSampler(Sampler):
    """
    Produces indices for inference across all workers.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, size):
        """
        Args:
            size (int): the total number of data on the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_size = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_size[:rank])
        end = min(sum(shard_size[: rank + 1]), total_size)
        return range(begin, end)
    
    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)