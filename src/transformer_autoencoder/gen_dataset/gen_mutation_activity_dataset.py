#######################################################################################################
# Code to generate the activity dataset for a transformer predictor
#######################################################################################################

import numpy as np
import re
import matplotlib.pyplot as plt
import os
import copy

from .gen_dataset import SeqDatasetGenerator, SeqDatasetMetadata

class MutationDatasetGenerator(SeqDatasetGenerator):
    def __init__(self, n_seq=int(1e5), seq_length=200, label_prefix='label', **kwargs):
        super().__init__(**kwargs)

        self.label_file = os.path.join(kwargs['dataset_dir'], label_prefix + '.txt')
        self.n_seq = n_seq
        self.seq_length = seq_length


    def prepare_dataset(self):
        super().prepare_dataset()
        super().split_other_dataset_train_val_test(self.label_file)

    def prepare_sequences_file(self):
        self.gen_activity_data()

    def define_hotspots(self, n_beneficial=10, n_detrimental=10, n_flexible=30):

        indices = np.arange(self.seq_length)
        # Get beneficial indices
        beneficial_indices = np.random.choice(indices, size=n_beneficial, replace=False)
        remaining_indices = np.setxor1d(indices, beneficial_indices)
        # Get detrimental indices
        detrimental_indices = np.random.choice(remaining_indices, size=n_detrimental, replace=False)
        remaining_indices = np.setxor1d(remaining_indices, detrimental_indices)
        # Get flexible indices
        flexible_indices = np.random.choice(remaining_indices, size=n_flexible, replace=False)

        return beneficial_indices, detrimental_indices, flexible_indices

    def assign_activity(self, mutation_sites, beneficial_indices, detrimental_indices, flexible_indices):

        # Assign mutation impacts
        impact = np.random.uniform(low = -0.1,high = 0.1,size = 200) #most have no impact 
        impact[beneficial_indices] = np.random.uniform(low = 0,high = 0.6,size = 10)
        impact[detrimental_indices] = np.random.uniform(low = -0.6,high = 0.,size = 10)
        impact[flexible_indices] = np.random.uniform(low = -0.3,high = 0.3,size = 30)
        
        #assess the activity:
        activity = max(0, 1 + np.sum(impact[mutation_sites]))

        return activity

    def gen_activity_data(self, seq=np.array(list('MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLI'))):

        aminoacids = "ACDEFGHIKLMNPQRSTVWY"

        beneficial_indices, detrimental_indices, flexible_indices = self.define_hotspots()

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        # Log metadata for train/val/test split
        self.log_sequence_data(np.ones(self.n_seq)*self.seq_length)

        with open(self.seq_file, 'w') as seq_file:
            with open(self.label_file, 'w') as label_file:
                # Iterate for full training data:
                for i in range(self.n_seq):
                    # create a mutation:
                    mutation_sites  = np.random.randint(0, self.seq_length, 10)
                    # assess the activity:
                    activity = self.assign_activity(mutation_sites, beneficial_indices, detrimental_indices, flexible_indices)

                    mut_values = np.random.randint(0,20,10)
                    mut_sequence = copy.deepcopy(seq)
                    mut_sequence[mutation_sites] = [ aminoacids[mv] for mv in mut_values ]
                    mut_sequence = ''.join(mut_sequence)
                    
                    seq_file.write(mut_sequence + os.linesep)
                    label_file.write(str(activity) + os.linesep)