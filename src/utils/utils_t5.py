import os


def compute_okvqa_accuracy(batch_generations, correct_answers_str, partially_correct_answers_str):
    
    score = 0
    total = 0
    
    for gen, ca_str, pca_str in zip(batch_generations, correct_answers_str, partially_correct_answers_str):

        # Extract lists of answers
        ca = ca_str.split("\t")
        pca = pca_str.split("\t")
        total += 1
        
        if gen in ca:
            score += 1.0
        elif gen in pca:
            score += 0.6
    
    if total > 0:
        return score / total
    else:
        return 0.0



def generate_run_name(model, output_path, run_name=None):
    
    if run_name is not None:
        return run_name

    counter = 1
    candidate = f"{model}_run{counter}"
    
    while filename_already_used(os.path.join(output_path, candidate)):
        counter += 1
        candidate = f"{model}_run{counter}"

    return candidate


def generate_prediction_filename(model, output_path, run_name=None):
    
    if not filename_already_used(os.path.join(output_path, run_name)):
        return run_name + ".json"

    counter = 1
    candidate = f"{model}_run{counter}"
    
    while filename_already_used(os.path.join(output_path, candidate)):
        counter += 1
        candidate = f"{model}_run{counter}"

    return candidate + ".json"

        
def filename_already_used(filename):
    return os.path.exists(filename) or  os.path.exists(filename + ".ckpt") or  os.path.exists(filename + ".json")