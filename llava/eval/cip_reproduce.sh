root_path="/workspace"

pope_question_path="${root_path}/POPE/output/sip_captions"
pope_result_path="${root_path}/LLaVA/output/sip_captions"

model="llava-v1.5-13b"
model_root="/workspace/huggingface"
data="coco"
img_root="/workspace/data/coco/val2014"
subsets=("adversarial" "random" "popular")

# make output directory
if [ ! -d "$pope_result_path" ]; then
    # The directory does not exist, create it
    mkdir -p "$pope_result_path"
else
    # The directory exists, skip
    :
fi

for subset in "${subsets[@]}"
do
    echo "------------- Running for subset: $subset -------------"

    question_file="${pope_question_path}/${data}/${data}_pope_llava-v1.5-13b_full_${subset}.json"
    answer_file="${pope_result_path}/${data}/${data}_pope_${model}_${subset}_ans.json"


    python3 model_vqa_loader_cip-sc-4.py \
        --model-path ${model_root}/${model} \
       --question-file ${question_file} \
       --image-folder ${img_root} \
       --answers-file ${answer_file} \
       --temperature 0 \
       --conv-mode vicuna_v1

    python3 eval_pope_standard.py \
        --annotation-file ${question_file} \
        --question-file ${answer_file} \
        --result-file ${answer_file}

done