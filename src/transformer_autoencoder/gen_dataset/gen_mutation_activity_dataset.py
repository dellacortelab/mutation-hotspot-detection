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
    def __init__(
            self, 
            n_seq=int(1e5), 
            label_prefix='label', 
            base_seq=np.array(list('MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLI')), 
            amino_acids=np.array(list('ACDEFGHIKLMNPQRSTVWY')),
            **kwargs
        ):

        super().__init__(**kwargs)

        self.label_file = os.path.join(kwargs['dataset_dir'], label_prefix + '.txt')
        self.n_seq = n_seq
        self.base_seq = base_seq
        self.amino_acids = amino_acids


    def prepare_dataset(self):
        super().prepare_dataset()
        super().split_other_dataset_train_val_test(self.label_file)

    def prepare_sequences_file(self):
        self.gen_activity_data()

    def define_hotspots(self, n_beneficial=10, n_detrimental=10, n_flexible=30, save=True):
        """Define beneficial, detrimental, and flexible hotspots for a sequence
        Args:
            n_beneficial (int): the number of beneficial indices
            n_detrimental (int): the number of detrimental indices
            n_flexible (int): the number of flexible indices
            save (bool): whether to save the indices
        Returns:
            beneficial_indices ((seq_length) np.ndarray): the beneficial indices
            detrimental_indices ((seq_length) np.ndarray): the detrimental indices
            flexible_indices ((seq_length) np.ndarray): the flexible indices
        """
        indices = np.arange(len(self.base_seq))
        # Get beneficial indices
        beneficial_indices = np.random.choice(indices, size=n_beneficial, replace=False)
        remaining_indices = np.setxor1d(indices, beneficial_indices)
        # Get detrimental indices
        detrimental_indices = np.random.choice(remaining_indices, size=n_detrimental, replace=False)
        remaining_indices = np.setxor1d(remaining_indices, detrimental_indices)
        # Get flexible indices
        flexible_indices = np.random.choice(remaining_indices, size=n_flexible, replace=False)

        if save:
            save_file = os.path.join(self.dataset_dir, 'good_bad_flex_indices.npz')
            np.savez(save_file, beneficial=beneficial_indices, detrimental=detrimental_indices, flexible=flexible_indices)

        return beneficial_indices, detrimental_indices, flexible_indices

    def assign_activity(self, mut_sites, mut_values, beneficial_indices, detrimental_indices, flexible_indices):
        """Assign a number representing the enzyme activity of the sequence, based on whether mutations
        occur at beneficial locations, detrimental locations, flexible locations, or neutral locations.
        """
        # For O(1) lookup
        beneficial_indices, detrimental_indices, flexible_indices = set(beneficial_indices), set(detrimental_indices), set(flexible_indices)
        # At flexible sites, polar hydrophilic are beneficial, hydrophobic are detrimental
        hydrophobics = 'AILMFCV'
        hydrophilics = 'RNDQEHK'

        # Start with a baseline positive activity
        activity = 1.
        for mut_site, mut_value in zip(mut_sites, mut_values):
            # import pdb; pdb.set_trace()
            if mut_site in beneficial_indices:
                if self.simple_data:
                    activity += .6
                else:
                    activity += np.random.uniform(low = 0.3, high = 0.6)
            elif mut_site in detrimental_indices:
                if self.simple_data:
                    activity += -.6
                else:
                    activity += np.random.uniform(low = -0.6, high = -0.3)    
            # The way we handl flexible indices isn't great - we only
            # give a hydrophilic/hydrophobic bonus/penalty if it is a 
            # mutated residue, not for all flexible residues. A better 
            # way to do this would be to assign a value for all flexible
            # residues. We could expect better embeddings in this case,
            # and make a cool distance matrix for the embeddings
            elif mut_site in flexible_indices:
                if mut_value in hydrophilics:
                    if self.simple_data:
                        activity += .6
                    else:
                        activity += np.random.uniform(low = 0.3, high = 0.6)
                elif mut_value in hydrophobics:
                    if self.simple_data:
                        activity += -.6
                    else:
                        activity += np.random.uniform(low = -0.6, high = -0.3)
                else:
                    if self.simple_data:
                        pass
                    else:
                        activity += np.random.uniform(low = -0.3, high = 0.3)

        # # Assign mutation impacts
        # impact = np.random.uniform(low = -0.1, high = 0.1, size = 200) #most have no impact 
        # impact[beneficial_indices] = np.random.uniform(low = 0.3, high = 0.6, size = 10)
        # impact[detrimental_indices] = np.random.uniform(low = -0.6, high = -0.3, size = 10)

        # for idx in flexible_indices:
        #     if mut_sequence[idx] in hydrophilics:
        #         impact[idx] = np.random.uniform(low = 0.3, high = 0.6)
        #     elif mut_sequence[idx] in hydrophobics:
        #         impact[idx] = np.random.uniform(low = -0.6, high = -0.3)
        #     else:
        #         impact[idx] = np.random.uniform(low = -0.3, high = 0.3)
        
        #assess the activity:
        activity = max(0, activity)

        return activity

    def gen_activity_data(
        self, 
        n_mutations=10,
    ):
        beneficial_indices, detrimental_indices, flexible_indices = self.define_hotspots()
        # Log metadata for train/val/test split
        self.log_sequence_data(np.ones(self.n_seq)*len(self.base_seq))

        with open(self.seq_file, 'w') as seq_file:
            with open(self.label_file, 'w') as label_file:

                # Create lookup table for amino acid indices for O(1) lookup when setting mutant residues
                amino_acid_indices = { character:idx for idx, character in enumerate(self.amino_acids) }

                # Iterate for full training data:
                for i in range(self.n_seq):

                    # create a mutation:
                    mut_sequence = copy.deepcopy(self.base_seq)
                    mut_sites  = np.random.randint(0, len(self.base_seq), n_mutations)
                    current_amino_acids = self.base_seq[mut_sites]

                    # Choose mutant values from amino acids not equal to the current amino acid
                    mut_values = np.empty(n_mutations, dtype=str)
                    for i, current_amino_acid in enumerate(current_amino_acids):
                        current_amino_acid_idx = amino_acid_indices[current_amino_acid]
                        eligible_mutations = np.empty( len(self.amino_acids) - 1, dtype=str)
                        eligible_mutations[:current_amino_acid_idx] = self.amino_acids[:current_amino_acid_idx]
                        eligible_mutations[current_amino_acid_idx:] = self.amino_acids[current_amino_acid_idx+1:]
                        mut_values[i] = np.random.choice(eligible_mutations)

                    mut_sequence[mut_sites] = mut_values
                    # assess the activity:
                    activity = self.assign_activity(
                        mut_sites=mut_sites,
                        mut_values=mut_values, 
                        beneficial_indices=beneficial_indices, 
                        detrimental_indices=detrimental_indices, 
                        flexible_indices=flexible_indices
                        )
                    mut_sequence = ''.join(mut_sequence)
                    
                    seq_file.write(mut_sequence + os.linesep)
                    label_file.write(str(activity) + os.linesep)