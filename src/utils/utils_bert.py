import os


def vqa_accuracy(list_of_answers, ans, soft_score=True):
    if soft_score:
        soft_acc = [min(1, (list_of_answers[:i] + list_of_answers[(i+1):]).count(ans) / 3) for i in range(len(list_of_answers))]
        return sum(soft_acc) / len(list_of_answers)
    else:
        return min(1, list_of_answers.count(ans) / 3)


def generate_ckpt_filename(model, cap_type, dataset, output_path, pretraining_task=None, run_name=None):
    
    if run_name is not None and not os.path.exists(os.path.join(output_path, run_name)):
        return run_name + ".ckpt"

    if pretraining_task is not None:
        trained_on = f"{pretraining_task}_{dataset}"
    else:
        trained_on = dataset
    filename = f"{model}_{cap_type}_{trained_on}"

    extension = "ckpt"

    counter = 1
    candidate = f"{filename}_run{counter}.{extension}"    
    
    while os.path.exists(os.path.join(output_path, candidate)):
        counter += 1
        candidate = f"{filename}_run{counter}.{extension}"
    
    return candidate
        