import os
import subprocess
import pandas as pd
from glob import glob

current_dir = os.path.dirname(os.path.abspath(__file__))

merged_base_path = os.path.join(current_dir, "merged")


def create_config_file(model_name):
    template = f"""
import torch

from mmengine.config import read_base

with read_base():
    from .datasets.ceval.ceval_gen import ceval_datasets
    from .datasets.gsm8k.gsm8k_gen_small_train import gsm8k_datasets

    from .summarizers.example import summarizer

datasets = ceval_datasets + gsm8k_datasets

from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        path='{os.path.join(merged_base_path, model_name)}',
        tokenizer_path='{os.path.join(merged_base_path, model_name)}',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        model_kwargs=dict(device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True),
        max_seq_len=2048,
        abbr='{model_name}',
        max_out_len=1024,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|im_end|>', '<|im_start|>'],
    ),
]
"""
    config_file_path = f'../opencompass/configs/{model_name}.py'
    with open(config_file_path, 'w', encoding='utf-8') as f:
        f.write(template)
    return config_file_path

def create_config_file_mcq(model_name):
    template = f"""
import torch

from mmengine.config import read_base

with read_base():
    from .datasets.ceval.ceval_gen import ceval_datasets

    from .summarizers.example import summarizer

datasets = ceval_datasets

from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        path='{os.path.join(merged_base_path, model_name)}',
        tokenizer_path='{os.path.join(merged_base_path, model_name)}',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        model_kwargs=dict(device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True),
        max_seq_len=2048,
        abbr='{model_name}',
        max_out_len=2,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|im_end|>', '<|im_start|>'],
    ),
]
"""
    config_file_path = f'../opencompass/configs/{model_name}.py'
    with open(config_file_path, 'w', encoding='utf-8') as f:
        f.write(template)
    return config_file_path

def create_config_file_nomcq(model_name):
    template = f"""
import torch

from mmengine.config import read_base

with read_base():
    from .datasets.gsm8k.gsm8k_gen_small_train import gsm8k_datasets

    from .summarizers.example import summarizer

datasets = gsm8k_datasets

from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        path='{os.path.join(merged_base_path, model_name)}',
        tokenizer_path='{os.path.join(merged_base_path, model_name)}',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        model_kwargs=dict(device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True),
        max_seq_len=2048,
        abbr='{model_name}',
        max_out_len=1024,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|im_end|>', '<|im_start|>'],
    ),
]
"""
    config_file_path = f'../opencompass/configs/{model_name}.py'
    with open(config_file_path, 'w', encoding='utf-8') as f:
        f.write(template)
    return config_file_path

def get_latest_summary_csv():
    output_dir = '../opencompass/outputs/mm-mo-sota'
    timestamp_dirs = sorted(glob(os.path.join(output_dir, '*/')), key=os.path.getmtime, reverse=True)
    if not timestamp_dirs:
        raise FileNotFoundError("No output directories found.")
    
    latest_dir = timestamp_dirs[0].strip('/')
    timestamp = os.path.basename(latest_dir)
    summary_csv = os.path.join(latest_dir, 'summary', f'summary_{timestamp}.csv')
    
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"Summary CSV file not found: {summary_csv}")
    
    return summary_csv

def parse_summary_csv(model_name):
    summary_csv = get_latest_summary_csv()
    df = pd.read_csv(summary_csv)

    ceval_accuracy = df[(df['dataset'] == 'ceval') & (df['metric'] == 'naive_average')][model_name].values[0]
    gsm8k_accuracy = df[(df['dataset'] == 'gsm8k') & (df['metric'] == 'accuracy')][model_name].values[0]

    return ceval_accuracy, gsm8k_accuracy

def parse_summary_csv_mcq(model_name):
    summary_csv = get_latest_summary_csv()
    df = pd.read_csv(summary_csv)

    ceval_accuracy = df[(df['dataset'] == 'ceval') & (df['metric'] == 'naive_average')][model_name].values[0]

    return ceval_accuracy

def parse_summary_csv_nomcq(model_name):
    summary_csv = get_latest_summary_csv()
    df = pd.read_csv(summary_csv)

    gsm8k_accuracy = df[(df['dataset'] == 'gsm8k') & (df['metric'] == 'accuracy')][model_name].values[0]

    return gsm8k_accuracy


def opencompass_eval(model_name):
    config_file_path = create_config_file(model_name)
    
    try:
        subprocess.run(
            ['python', 'run.py', config_file_path.replace('../opencompass/', ''), 
             '-w', 'outputs/mm-mo-sota', '--max-num-workers', '2'],
            cwd='../opencompass',
            check=True,
            capture_output=False,
            # text=True,
            # env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
        )
        ceval_accuracy, gsm8k_accuracy = parse_summary_csv(model_name)

    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {e}")
        ceval_accuracy = None
        gsm8k_accuracy = None
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        ceval_accuracy = None
        gsm8k_accuracy = None

    return ceval_accuracy, gsm8k_accuracy

def opencompass_eval_mcq(model_name):
    config_file_path = create_config_file_mcq(model_name)
    
    try:
        subprocess.run(
            ['python', 'run.py', config_file_path.replace('../opencompass/', ''), 
             '-w', 'outputs/mm-mo-sota'],
            cwd='../opencompass',
            check=True,
            capture_output=False,
            # text=True,
            # env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
        )
        ceval_accuracy = parse_summary_csv_mcq(model_name)

    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {e}")
        ceval_accuracy = None
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        ceval_accuracy = None

    return ceval_accuracy

def opencompass_eval_nomcq(model_name):
    config_file_path = create_config_file_nomcq(model_name)
    
    try:
        subprocess.run(
            ['python', 'run.py', config_file_path.replace('../opencompass/', ''), 
             '-w', 'outputs/mm-mo-sota'],
            cwd='../opencompass',
            check=True,
            capture_output=False,
            # text=True,
            # env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
        )
        gsm8k_accuracy = parse_summary_csv_nomcq(model_name)

    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {e}")
        gsm8k_accuracy = None
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        gsm8k_accuracy = None

    return gsm8k_accuracy
