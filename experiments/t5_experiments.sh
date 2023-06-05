
#!/bin/bash
output_path="$1"
id="$2"

declare -a FinetuningDatasets=("okvqa_v1.1" "vqa_v2")
declare -a OkvqaDatasets=("okvqa_v1.1" "okvqa_v1.0")
 

python3 src/t5_train.py --model "t5-small" --do_train --do_predict --fast_version  \
    --output_path $output_path \
    --run_name "t5-small_run_${id}" --max_steps 18003

python3 src/t5_train.py --model "t5-base" --do_train --do_predict --fast_version  \
    --output_path $output_path \
    --run_name "t5-base_run_${id}" --max_steps 7892

python3 src/t5_train.py --model "t5-large" --do_train --do_predict --fast_version  \
    --output_path $output_path \
    --run_name "t5-large_run_${id}" --max_steps 5034

python3 src/t5_train.py --model "t5-3b" --do_train --do_predict --fast_version  \
    --output_path $output_path \
    --batch_size 8 --accumulate_grad_batches 7 \
    --run_name "t5-3b_run_${id}" --max_steps 3882

python3 src/t5_train.py --model "t5-11b" --do_train --do_predict \
    --output_path $output_path \
    --gpus 4 --batch_size 2 --accumulate_grad_batches 7 --deepspeed \
    --run_name "t5-11b_run_${id}" --max_steps 1 --max_steps 381
