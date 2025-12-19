import torch
from transformers import AutoModel

def cal_delta_l1_norm(model_name_or_path, base_model="Qwen/Qwen1.5-7B"):
    model_name_1 = model_name_or_path
    model_1 = AutoModel.from_pretrained(model_name_1)

    model_name_2 = base_model
    model_2 = AutoModel.from_pretrained(model_name_2)

    assert len(list(model_1.parameters())) == len(list(model_2.parameters())), "The number of parameters of the two models does not match."

    delta_l1_norm = 0

    for param1, param2 in zip(model_1.parameters(), model_2.parameters()):
        delta_weight = param1 - param2
        delta_l1_norm += torch.sum(torch.abs(delta_weight))

    return delta_l1_norm.item()