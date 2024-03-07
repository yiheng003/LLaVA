import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import re
import random

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.n_shot = 1
        self.mask_method = "masked_mean"
        self.pair_mode = "regular"

    def __getitem__(self, index):
        line = self.questions[index]
        # test image.
        image_file = line["image"]
        # question.
        qs = line["text"]
        # gt answer.
        gt = line["label"]
        # one-shot demonstrations.
        demo_1_path = line["p0_image"]
        demo_1_objects = list(set(line["p0_objects"]))
        # demo_1_description = 'The image features ' + ', '.join(demo_1_objects)
        demo_2_path = line["n0_image"]
        demo_2_objects = list(set(line["n0_objects"]))
        # demo_2_description = 'The image features ' + ', '.join(demo_2_objects)
        if self.n_shot == 1:
            demo_image_list = [
                [
                    Image.open(os.path.join(self.image_folder, demo_1_path)).convert('RGB'), 
                    Image.open(os.path.join(self.image_folder, demo_2_path)).convert('RGB')
                ]
            ]
            if self.model_config.mm_use_im_start_end:
                demo_instruct_list = [
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs, 
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                ]
            else:
                demo_instruct_list = [
                    DEFAULT_IMAGE_TOKEN + '\n' + qs, 
                    DEFAULT_IMAGE_TOKEN + '\n' + qs
                ]
        else:
            exit(0)
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            # "is there a tennis ball in the image?" -> "<image>\nis there a tennis ball in the image?"
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        conv = conv_templates[args.conv_mode].copy()

        # ! object name in question can be accessed here.
        pattern_1 = r"Is there a (.+?) in the"
        pattern_2 = r"Is there an (.+?) in the"
        match1 = re.search(pattern_1, qs)
        match2 = re.search(pattern_2, qs)
        if match1:
            object_name = match1.group(1)
        else:
            object_name = match2.group(1)

        image_list = []
        image_size_list = []

        neg_object_name = random.choice(demo_2_objects)

        # context.
        for idx in range(self.n_shot):
            conv.append_message(conv.roles[0], demo_instruct_list[idx])
            conv.append_message(conv.roles[1], f"There is a {object_name} in this image. So my answer is yes.\n")
            conv.append_message(conv.roles[0], demo_instruct_list[idx])
            conv.append_message(conv.roles[1], f"There is no {object_name} in this image. The image appears to feature a {neg_object_name} instead. So my answer is no.\n")

            image_list.append(demo_image_list[idx][0])
            image_list.append(demo_image_list[idx][1])

            image_size_list.append(demo_image_list[idx][0].size)
            image_size_list.append(demo_image_list[idx][1].size)

        # question.
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None) 
        '''
            [conv.system]
                "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            [conv.messages] 
                [
                ['USER', '<image>\nIs there a tennis ball in the image?'],
                ['ASSISTANT', 'Yes.\n'],
                ['USER', '<image>\nIs there a tennis ball in the image?'],
                ['ASSISTANT', 'No.\n'],
                ['USER', '<image>\nIs there a tennis ball in the image?'],
                ['ASSISTANT', None]
                ]
        '''
        image_list.append(image)
        image_size_list.append(image.size)    # ! test and demonstration image sizes are different.
        
        # prompt.
        prompt = conv.get_prompt()
        prompt = prompt.replace("</s>", "")   # ! manually replace </s> but need double check.
        print('==========')
        print(f"prompt: {prompt}".encode('unicode_escape'))
        print("gt: ", gt)
        
        image_tensor = process_images(image_list, self.image_processor, self.model_config)   # ! need double check.
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')    # ! need double check.
        return prompt, input_ids, image_tensor, image_size_list

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    prompt, input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    image_tensors = torch.squeeze(image_tensors, dim=0)
    image_sizes = tuple(image_sizes[0])
    return prompt, input_ids, image_tensors, image_sizes

# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=0):    # ----- pdb - xy
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (prompt, input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        label = line['label']
        image_path = line['image']
        pos_demonstration_image_path = line['p0_image']
        neg_demonstration_image_path = line['n0_image']
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        # print("model output_ids: ", output_ids)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print("model outputs: ", outputs)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "image": f'~/data/coco/val2014/{image_path}',
                                   "pos_image": f'~/data/coco/val2014/{pos_demonstration_image_path}',
                                   "neg_image": f'~/data/coco/val2014/{neg_demonstration_image_path}',
                                   "prompt": prompt[0],
                                   "text": outputs,
                                   "label": label,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}},indent=4) + "\n")
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--demonstration_file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
