def get_module_class(name):
    return getattr(getattr(__import__('bsmllib'), 'model'), name)