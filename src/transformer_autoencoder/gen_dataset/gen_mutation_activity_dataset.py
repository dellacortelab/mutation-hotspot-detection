#######################################################################################################
# Code to generate the activity dataset for a transformer predictor
#######################################################################################################

import numpy as np
import re
import matplotlib.pyplot as plt

from .gen_dataset import DatasetGenerator

class MutationDatasetGenerator(DatasetGenerator):
    def __init__(self):
        super().__init__()

    def prepare_dataset(self):
        self.gen_activity_data()

    def define_hotspots(seq_l=200, n_beneficial=10, n_detrimental=10, n_flexible=30):

        indices = np.arange(seq_l)
        # Get beneficial indices
        beneficial_indices = np.random.choice(indices, size=n_beneficial, replace=False)
        remaining_indices = np.setxor1d(indices, beneficial_indices)
        # Get detrimental indices
        detrimental_indices = np.random.choice(remaining_indices, size=n_detrimental, replace=False)
        remaining_indices = np.setxor1d(remaining_indices, detrimental_indices)
        # Get flexible indices
        flexible_indices = np.random.choice(remaining_indices, size=n_flexible, replace=False)

        return beneficial_indices, detrimental_indices, flexible_indices


    def assign_activity(mutation_sites, beneficial_indices, detrimental_indices, flexible_indices):

        # Assign mutation impacts
        impact = np.random.uniform(low = -0.1,high = 0.1,size = 200) #most have no impact 
        impact[beneficial_indices] = np.random.uniform(low = 0,high = 0.6,size = 10)
        impact[detrimental_indices] = np.random.uniform(low = -0.6,high = 0.,size = 10)
        impact[flexible_indices] = np.random.uniform(low = -0.3,high = 0.3,size = 30)
        
        #assess the activity:
        activity = max(0, 1 + np.sum(impact[mutation_sites]))

        return activity


    def gen_activity_data(output_file="./training_data.npy"):
        aminoacids = "ACDEFGHIKLMNPQRSTVWY"

        beneficial_indices, detrimental_indices, flexible_indices = define_hotspots()

        # Iterate for full training data:
        training_data = {}
        for i in range(100*1000):
            # create a mutation:
            mutation_sites  = np.random.randint(0,200,10)
            # assess the activity:
            activity = assign_activity(mutation_sites, beneficial_indices, detrimental_indices, flexible_indices)

            mut_values = np.random.randint(0,20,10)
            mut_sequence = seq
            mut_sequence[mut_sites] = [ aminoacids[mv] for mv in mut_values ]
            mut_sequence = ''.join(mut_sequence)
            # Add an index to string-float map to the dictionary
            training_data[i] = [mut_sequence, activity]

        #save or load data
        np.save(output_file, training_data)