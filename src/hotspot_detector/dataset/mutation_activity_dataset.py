
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

    def visualize_activity_data(self, training_data_file="./training_data.npy"):
        training_data = np.load(training_data_file)
        a = np.array(list(training_data.values()))[:,1]
        a = a.astype(float)
        plt.hist(a)
        plt.xlabel('Activity')
        plt.ylabel('Frequency')
        plt.show()
        plt.save_fig('training_data.png')