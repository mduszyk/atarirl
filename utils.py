import json
import os
import tomllib


class FrozenAttrDict(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, '__dict__', self)

    def __setattr__(self, key, value):
        raise AttributeError('FrozenAttrDict is immutable')

    def __delattr__(self, key):
        raise AttributeError('FrozenAttrDict is immutable')

    def __setitem__(self, key, value):
        raise TypeError('FrozenAttrDict is immutable')

    def __delitem__(self, key):
        raise TypeError('FrozenAttrDict is immutable')


def load_params(path, profile=None, env_var_prefix=None):
    with open(path, 'rb') as f:
        all_params = tomllib.load(f)
        params = all_params.get('default', {})

    if profile is not None:
        params.update(all_params[profile])

    if env_var_prefix is not None:
        for env_key, env_value in os.environ.items():
            if env_key.startswith(env_var_prefix):
                key = env_key.replace(env_var_prefix, '').lower()
                value = params.get(key)
                if value is not None and not isinstance(value, str):
                    if isinstance(value, dict) or isinstance(value, list):
                        env_value = json.loads(env_value)
                    elif isinstance(value, bool):
                        env_value = env_value.lower() == 'true'
                    else:
                        env_value = type(value)(env_value)
                params[key] = env_value

    return FrozenAttrDict(params)
