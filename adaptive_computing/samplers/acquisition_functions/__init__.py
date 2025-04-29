import importlib
import pkgutil

__all__ = []

# Discover and import all functions
acq_func_map = {}

for loader, module_name, _ in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f"{__name__}.{module_name}")
    if hasattr(module, module_name):
        acq_func_map[module_name] = getattr(module, module_name)
        __all__.append(module_name)
