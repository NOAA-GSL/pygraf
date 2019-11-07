import importlib as il
import sys


def get_func(val):

    '''
    Given an input string, val, returns the corresponding callable function.
    This functino is borrowed from stackoverflow.com response to "Python: YAML
    dictionary of functions: how to load without converting to strings."
    '''

    if '.' in val:
        module_name, fun_name = val.rsplit('.', 1)
    else:
        module_name = '__main__'
        fun_name = val

    mod_spec = il.util.find_spec(module_name, package='adb_graphics')
    if mod_spec is None:
        mod_spec = il.util.find_spec('.' + module_name, package='adb_graphics')

    try:
        __import__(mod_spec.name)
    except ImportError as exc:
        print(f'Could not load {module_name} while trying to locate function in get_func')
        raise exc
    module = sys.modules[mod_spec.name]
    fun = getattr(module, fun_name)
    return fun
