import os
import json
import time
import datetime
import torch
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from transformers import LlamaForCausalLM, LlamaTokenizerFast, AutoModel

base_path = ''
prompt_template = "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n"
response_split = "### Response:"

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for the script")
    parser.add_argument("--test_data_path", type=str, default="data/subgraph_mc/test.json", help="Path to the test data file")
    parser.add_argument("--embedding_path", type=str, default="", help="Path to the embedding file")
    parser.add_argument("--cuda", type=int, default=1, help="CUDA device to use")
    parser.add_argument("--log", type=str, default="log/test.json", help="Path to the log file")
    parser.add_argument('--ckpt_name', type=str, default="embeddings-final.pth")
    return parser.parse_args()


def load_test_dataset(path):
    test_dataset = json.load(open(path, "r"))
    return test_dataset

def inference_on_single_ckpt(test_data_path, embedding_path, cuda):
    test_dataset = load_test_dataset(test_data_path)
    kg_embeddings = torch.load(embedding_path)# .cuda(cuda)
    tokenizer = LlamaTokenizerFast.from_pretrained(base_path)
    model = LlamaForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16).cuda()
    print(kg_embeddings)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 128000
    model.config.eos_token_id = 128001
    model = model.eval()
    kg_embeddings = kg_embeddings.eval()
    result = []
    acc = 0
    max_token_num = 64 if "desc" in test_data_path else 1
    for data in tqdm(test_dataset):
        instruction = data["instruction"]
        input = data["input"]
        ans = data["output"]
        ids = data["embedding_ids"]
        ids = torch.LongTensor(ids).reshape(1, -1).cuda()
        prompt = prompt_template.format(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.cuda()
        token_embeds = model.model.embed_tokens(input_ids)
        knowledge_prompt = kg_embeddings(ids)
        input_embeds = torch.cat((knowledge_prompt.to(token_embeds.dtype), token_embeds), dim=1)
        generate_ids = model.generate(
            inputs_embeds=input_embeds, 
            max_new_tokens=max_token_num,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        response = response.split(response_split)[-1]
        # print(response + '\n')
        result.append(
            {
                "answer": ans,
                "predict": response
            }
        )
    for i in result:
        if i["answer"] == i["predict"]:
            acc += 1
    print("Test Results: ", acc / len(result))
    task_type = test_data_path.split('/')[1]
    model_type = embedding_path.split('/')[6] + embedding_path.split('/')[7]
    print("Test Task: ", task_type)
    print("Test Model: ", model_type)
    json.dump(result, open("test/{}-{}-{}.json".format(task_type, model_type, datetime.datetime.now()), "w"))




if __name__ == "__main__":
    args = parse_args()
    ckpt_paths = []
    print(os.listdir(args.embedding_path))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    print("Use Llama Model: ", base_path)
    for entry in os.listdir(args.embedding_path):
        full_path = os.path.join(args.embedding_path, entry)
        if os.path.isdir(full_path):
            ckpt_paths.append(os.path.join(full_path, args.ckpt_name))
    print(ckpt_paths)
    for ckpt in ckpt_paths:
        inference_on_single_ckpt(args.test_data_path, embedding_path=ckpt, cuda=args.cuda)
    