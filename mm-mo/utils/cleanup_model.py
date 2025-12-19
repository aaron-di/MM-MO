import os
import shutil

def cleanup_merged_models(model_dir):
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        print(f'Model directory {model_dir} has been deleted.')

model_dir = './merged'
cleanup_merged_models(model_dir)