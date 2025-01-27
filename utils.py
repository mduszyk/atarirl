import os
import tomllib


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def load_params(path, profile=None, env_var_prefix=None):
    with open(path, 'rb') as f:
        all_params = tomllib.load(f)
        params = all_params.get('default', {})

    if profile is not None:
        params.update(all_params[profile])

    if env_var_prefix is not None:
        for key, value in os.environ.items():
            if key.startswith(env_var_prefix):
                params[key.replace(env_var_prefix, '').lower()] = value

    return AttrDict(params)
