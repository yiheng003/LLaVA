root="/workspace"
model_root="${root}/pretrained"
model='llava-v1.5-7b-voc-lora'
model_base="${root}/huggingface/llava-v1.5-7b"
scenarios=("coco_voc_unseen" "coco_voc_seen")
subsets=("adversarial" "random" "popular")
img_roots="${root}/data/coco/val2014"

for scenario in "${scenarios[@]}"
do
    echo "------------- Running for scenario: $scenario -------------"

    for subset in "${subsets[@]}"
    do
        echo "------------- Running for subset: $subset -------------"
        
        question_file="${root}/POPE/output/${scenario}/${scenario}_pope_${subset}.json"
        answer_file="${root}/POPE/output/${scenario}/${model}_${scenario}_pope_${subset}_ans.json"
        
        if test -e ${answer_file}; then
            python eval_pope_standard.py \
                --annotation-file ${question_file} \
                --question-file ${question_file} \
                --result-file ${answer_file}
        else
            python model_vqa.py \
                --model-path ${model_root}/${model} \
                --model-base ${model_base} \
                --question-file ${question_file} \
                --image-folder ${img_roots} \
                --answers-file ${answer_file}
            python eval_pope_standard.py \
                --annotation-file ${question_file} \
                --question-file ${question_file} \
                --result-file ${answer_file}
        fi
    done
done

