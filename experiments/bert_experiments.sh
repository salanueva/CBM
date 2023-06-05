
#!/bin/bash
output_path="$1"
id="$2"

declare -a FinetuningDatasets=("okvqa_v1.1" "vqa_v2")
declare -a OkvqaDatasets=("okvqa_v1.1" "okvqa_v1.0")


# Train all models in 2 tasks, one fine-tuning per task (okvqa v1.1 and vqa v2)
for dataset in ${FinetuningDatasets[@]}; do

    python src/bert_train.py --model "bertq" --dataset $dataset --evaluate --output_path $output_path --run_name "bertq_oscar_${dataset}_run_${id}"

    python src/bert_train.py --model "cbm" --dataset $dataset --evaluate --output_path $output_path --run_name "cbm_oscar_${dataset}_run_${id}"

    python src/bert_train.py --model "mmbert" --dataset $dataset --evaluate --output_path $output_path --run_name "mmbert_oscar_${dataset}_run_${id}"
    
    python src/bert_train.py --model "cbm+mmbert" --dataset $dataset --evaluate --output_path $output_path --run_name "cbm+mmbert_oscar_${dataset}_run_${id}"
    
    python src/bert_eval_late_fusion.py --dataset $dataset --ckpt_cbm "${output_path}/cbm_oscar_${dataset}_run_${id}.ckpt" \
        --ckpt_mmbert "${output_path}/mmbert_oscar_${dataset}_run_${id}.ckpt"

done

# Fine-tune all vqa models in okvqa v1.0 and okvqa_v1.1
for dataset in ${OkvqaDatasets[@]}; do

    python src/bert_train.py --model "bertq" --dataset $dataset --evaluate --output_path $output_path \
        --ckpt "${output_path}/bertq_oscar_vqa_v2_run_${id}.ckpt"  --run_name "bertq_oscar_vqa_v2_${dataset}_run_${id}"

    python src/bert_train.py --model "cbm" --dataset $dataset --evaluate --output_path $output_path \
        --ckpt "${output_path}/cbm_oscar_vqa_v2_run_${id}.ckpt"  --run_name "cbm_oscar_vqa_v2_${dataset}_run_${id}"

    python src/bert_train.py --model "mmbert" --dataset $dataset --evaluate --output_path $output_path \
        --ckpt "${output_path}/mmbert_oscar_vqa_v2_run_${id}.ckpt"  --run_name "mmbert_oscar_vqa_v2_${dataset}_run_${id}"
    
    python src/bert_train.py --model "cbm+mmbert" --dataset $dataset --evaluate --output_path $output_path \
        --ckpt "${output_path}/cbm+mmbert_oscar_vqa_v2_run_${id}.ckpt"  --run_name "cbm+mmbert_oscar_vqa_v2_${dataset}_run_${id}"
    
    python src/bert_eval_late_fusion.py --dataset $dataset --ckpt_cbm "${output_path}/cbm_oscar_vqa_v2_${dataset}_run_${id}.ckpt" \
        --ckpt_mmbert "${output_path}/mmbert_oscar_vqa_v2_${dataset}_run_${id}.ckpt"


done


# Human captions
python src/bert_train.py --model "cbm" --dataset "okvqa_v1.1" --cap_type "human" --evaluate --output $output_path --run_name "cbm_human_okvqa_v1.1_run_${id}"