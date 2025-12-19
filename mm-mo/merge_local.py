import subprocess

def run_mergekit_yaml(config_path, output_dir, options=None):
    command = [
        'mergekit-yaml',
        config_path,
        output_dir
    ]
    if options:
        command.extend(options)

    print('Running command:', ' '.join(command))
    
    result = subprocess.run(command, capture_output=False, text=True)
    
    return result
