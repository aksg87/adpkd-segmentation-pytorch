from importlib import import_module

"""Config parsing utilities"""

CLASS_INFO = "_CLASS_INFO"
CLASS_NAME = "_CLASS_NAME"
MODULE_NAME = "_MODULE_NAME"


def get_simple_instance(module_name, class_name, processed_params):
    """
    Returns a class instance.

    Args:
        module_name: str, module name
        class_name: str, class_name
        processed_params: dict, class params
    """
    module = import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_(**processed_params)

    return instance


def is_object_config(object_config):
    criterion = (
        isinstance(object_config, dict)
        and CLASS_INFO in object_config
        and MODULE_NAME in object_config[CLASS_INFO]
        and CLASS_NAME in object_config[CLASS_INFO]
    )
    return criterion


def is_simple_value(value):
    return (not isinstance(value, dict)) and (not isinstance(value, list))


def process_nested(config):
    """
    Process a json serializable config, replacing
    all nested object configurations with instances
    of those objects.

    Args:
        config: dict, list, or `object_config` as defined by `is_object_config`
    Returns:
        dict, list, or object (replacing all `object_config` with instances)
    """

    if isinstance(config, list):
        processed = []
        for v in config:
            if is_simple_value(v):
                processed.append(v)
            else:
                processed.append(process_nested(v))

    elif is_object_config(config):
        object_params = {}
        class_info = config[CLASS_INFO]
        module_name = class_info[MODULE_NAME]
        class_name = class_info[CLASS_NAME]

        for k, v in config.items():
            # special info key is not to be used inside `get_simple_instance`
            if k == CLASS_INFO:
                continue
            # value is a leaf
            elif is_simple_value(v):
                object_params[k] = v
            # nested object
            else:
                object_params[k] = process_nested(v)
        processed = get_simple_instance(module_name, class_name, object_params)

    elif isinstance(config, dict):
        processed = {}
        for k, v in config.items():
            if is_simple_value(v):
                processed[k] = v
            else:
                processed[k] = process_nested(v)

    return processed


def get_object_instance(object_config):
    assert is_object_config(object_config)
    return process_nested(object_config)


class ObjectGetter:
    def __init__(self, module_name, object_name):
        self.module_name = module_name
        self.object_name = object_name

    def __call__(self):
        module_ = import_module(self.module_name)
        return getattr(module_, self.object_name)
