import argparse
import torch
import os
import re
import json
from tqdm import tqdm
import shortuuid
import torch.nn.functional as F

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

    def __getitem__(self, index):
        line = self.questions[index]
        qs = line["text"]

        # ! object name in question can be accessed here.
        pattern = r"Is there (.+?) in the"
        object_name = re.search(pattern, qs).group(1)
        try:
            a_name, object_name = object_name.split(' ')[0], ', '.join(object_name.split(' ')[1:])
        except Exception:
            import pdb; pdb.set_trace()

        p0_image_file = line['p0_image']
        p1_image_file = line['p1_image']
        n0_image_file = line['n0_image']
        n1_image_file = line['n1_image']

        qs_n = f'describe this image.'
        qs_p = f'there is {a_name} {object_name} in this image. specify on the {object_name} you see and describe interactions of this object with other objects you see.'
        
        if self.model_config.mm_use_im_start_end:
            qs_n = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_n
        else:
            qs_n = DEFAULT_IMAGE_TOKEN + '\n' + qs_n

        if self.model_config.mm_use_im_start_end:
            qs_p = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_p
        else:
            qs_p = DEFAULT_IMAGE_TOKEN + '\n' + qs_p

        conv_n = conv_templates[args.conv_mode].copy()
        conv_n.append_message(conv_n.roles[0], qs_n)
        conv_n.append_message(conv_n.roles[1], None)
        prompt_n = conv_n.get_prompt()

        conv_p = conv_templates[args.conv_mode].copy()
        conv_p.append_message(conv_p.roles[0], qs_p)
        conv_p.append_message(conv_p.roles[1], None)
        prompt_p = conv_p.get_prompt()

        p0_image = Image.open(os.path.join(self.image_folder, p0_image_file)).convert('RGB')
        p1_image = Image.open(os.path.join(self.image_folder, p1_image_file)).convert('RGB')
        n0_image = Image.open(os.path.join(self.image_folder, n0_image_file)).convert('RGB')
        n1_image = Image.open(os.path.join(self.image_folder, n1_image_file)).convert('RGB')
        image_tensor = process_images([p0_image, p1_image, n0_image, n1_image], self.image_processor, self.model_config)


        input_ids_p = tokenizer_image_token(prompt_p, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids_n = tokenizer_image_token(prompt_n, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        len_p = input_ids_p.size(0)
        len_n = input_ids_n.size(0)
        pad_p = max(len_n, len_p) - len_p 
        pad_n = max(len_n, len_p) - len_n
        input_ids_p = F.pad(input_ids_p, (0, pad_p), 'constant', 0)
        input_ids_n = F.pad(input_ids_n, (0, pad_n), 'constant', 0)

        input_ids = torch.stack((input_ids_p, input_ids_p, input_ids_n, input_ids_n), dim=0)
        
        return input_ids, image_tensor, [p0_image.size, p1_image.size, n0_image.size, n1_image.size]

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    # input_ids = torch.stack(input_ids, dim=0)
    # image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids[0], image_tensors[0], image_sizes[0]


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=0):
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

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")][:args.question_num]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

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

        # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        p0_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        p1_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[1].strip()
        n0_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[2].strip()
        n1_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[3].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "p0_description": p0_output,
                                   "p1_description": p1_output,
                                   "n0_description": n0_output,
                                   "n1_description": n1_output,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--question_num", type=int, default=3000)
    args = parser.parse_args()

    eval_model(args)
