from llm4ad.method import meoh

__all__ = ['meoh']

import importlib
import inspect
import os


def import_all_method_classes_from_subfolders(root_directory: str):
    for subdir in os.listdir(root_directory):
        subdir_path = os.path.join(root_directory, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for module_basename in (subdir, 'profiler'):
            module_path = os.path.join(subdir_path, f'{module_basename}.py')
            if not os.path.exists(module_path):
                continue
            module_name = f'{__name__}.{subdir}.{module_basename}'
            module = importlib.import_module(module_name)
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if isinstance(attribute, type) and inspect.getmodule(attribute).__file__ == module.__file__:
                    globals()[attribute_name] = attribute
