CUDA_VISIBLE_DEVICES="0,1,2,3"
root_path="/home/xingyun"
pope_question_path="${root_path}/POPE/output/sip_questions"
pope_result_path="${root_path}/POPE/output/sip_captions"
question_num=3000
models=("llava-v1.5-13b" "llava-v1.5-7b")
model_roots=("/media/drive_16TB/huggingface" "/media/drive_16TB/huggingface")
data_list=("coco" "gqa" "aokvqa")
img_roots=("/media/drive_16TB/data/coco/val2014" "/media/drive_16TB/data/gqa/images" "/media/drive_16TB/data/coco/val2014")
subsets=("adversarial" "random" "popular")

is_test="true"

# make output directory
if [ ! -d "$pope_result_path" ]; then
    # The directory does not exist, create it
    mkdir -p "$pope_result_path"
    echo "------------- Directory created: $pope_result_path -------------"
else
    # The directory exists, skip
    echo "------------- Directory exists: $pope_result_path, skipping -------------"
fi

for i in "${!models[@]}"
do
    model_root=${model_roots[$i]}
    model=${models[$i]}
    echo "------------- Running for model: $model -------------"

    for j in "${!data_list[@]}"
    do
        data=${data_list[$j]}
        img_root=${img_roots[$j]}
        echo "------------- Running for data: $data -------------"

        for subset in "${subsets[@]}"
        do
            echo "------------- Running for subset: $subset -------------"

            # allow testing the scripts with a smaller question sets
            if [ "$is_test" = "true" ]; then
                question_file="${pope_question_path}/${data}/${data}_pope_${subset}_small.json"
                echo "------------- Testing Mode -------------"
            else
                echo "------------- Generation Mode -------------"
                question_file="${pope_question_path}/${data}/${data}_pope_${subset}.json"
            fi

            output_file_pos="${pope_result_path}/${data}/${data}_pope_${model}_pos_${subset}.json"
            output_file_neg="${pope_result_path}/${data}/${data}_pope_${model}_neg_${subset}.json"
            output_file_combined="${pope_result_path}/${data}/${data}_pope_${model}_full_${subset}.json"

            python model_vqa_loader_self_correction_p0p1.py \
                --model-path ${model_root}/${model} \
                --question-file ${question_file} \
                --image-folder ${img_root} \
                --answers-file ${output_file_pos} \
                --temperature 0 \
                --conv-mode vicuna_v1 \
                --question_num ${question_num}

            python model_vqa_loader_self_correction_n0n1.py \
                --model-path ${model_root}/${model} \
                --question-file ${question_file} \
                --image-folder ${img_root} \
                --answers-file ${output_file_neg} \
                --temperature 0 \
                --conv-mode vicuna_v1 \
                --question_num ${question_num}

            python prepare_cip_sc_4shot.py \
                --input-file ${question_file} \
                --pos1 ${output_file_pos} \
                --neg1 ${output_file_neg} \
                --output-file ${output_file_combined} \
                --question_num ${question_num}

        done
    done
done