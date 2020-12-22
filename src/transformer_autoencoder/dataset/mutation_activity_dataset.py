
from .seq_dataset import ManyToOneDataset
from ..gen_dataset.gen_mutation_activity_dataset import MutationDatasetGenerator

class MutationActivityDataset(ManyToOneDataset):
    """Sub-class of ManyToOneDataset, specifying the generator class"""
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, 
            **kwargs,
            dataset_generator_class=MutationDatasetGenerator
        )