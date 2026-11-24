import os
import time

_output_dir_cache = None

def _get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_output_dir(base_dir=None):
    global _output_dir_cache
    
    if base_dir is None:
        base_dir = _get_project_root()
    
    if _output_dir_cache is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        _output_dir_cache = os.path.join(base_dir, "output", timestamp)
        os.makedirs(_output_dir_cache, exist_ok=True)
    
    return _output_dir_cache

def get_excel_path(file_name, base_dir=None):
    output_dir = get_output_dir(base_dir)
    return os.path.join(output_dir, file_name)

def get_image_path(file_name, base_dir=None):
    output_dir = get_output_dir(base_dir)
    return os.path.join(output_dir, file_name)

def get_data_path(file_name, base_dir=None):
    if base_dir is None:
        base_dir = _get_project_root()
    
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    return os.path.join(data_dir, file_name)
