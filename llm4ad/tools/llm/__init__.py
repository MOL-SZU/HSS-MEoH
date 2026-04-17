import importlib
import inspect
import os

from .llm_api_https import HttpsApi

__all__ = ['HttpsApi', 'import_all_llm_classes_from_subfolders']


def import_all_llm_classes_from_subfolders(root_directory):
    for subdir in os.listdir(root_directory):
        module_path = os.path.join(root_directory, subdir)
        if os.path.basename(module_path) == '__init__.py':
            continue
        if os.path.isdir(module_path):
            module_name = f'{__name__}.{subdir}'
        else:
            if not subdir.endswith('.py') or subdir == 'llm_api_https.py':
                continue
            module_name = f'{__name__}.{subdir[:-3]}'
        module = importlib.import_module(module_name)
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if isinstance(attribute, type) and inspect.getmodule(attribute).__file__ == module.__file__:
                globals()[attribute_name] = attribute
