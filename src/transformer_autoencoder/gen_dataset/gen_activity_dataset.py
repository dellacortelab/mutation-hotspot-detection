

# Autoencoder experiment

def synthetic_kcat_score(seq):
    """Assign a score simulating catalytic activity based on an input sequence"""
    key_seq_1 = #common seq - see tokenizer
    pattern_1 = re.compile(key_seq_1)
    cnt_1 = len(re.findall(pattern_1, key_seq_1))
    
    key_seq_2 = #common seq - see tokenizer
    pattern_2 = re.compile(key_seq_2)
    cnt_2 = len(re.findall(pattern_2, key_seq_2))
    
    # "Score" is the sum of the matching sequences plus some randomness
    score = cnt_1 + cnt_2 + (.5 * np.random.randn())

def gen_kcat_outputs(output_file="/data/uniparc/synthetic_data/sequence_scores.txt"):
    """Create file with scores for sequences"""

    if not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file))

    with open(seq_file, "r") as inp_file:
        with open(output_file, "w") as out_file:
            for line in inp_file:
                score = synthetic_kcat_score(line)
                out_file.write(score + '\n')