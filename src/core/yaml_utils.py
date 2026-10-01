"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

import os
import copy
import yaml 
from typing import Any, Dict, Optional, List

from .workspace import GLOBAL_CONFIG

__all__ = [
    'load_config', 
    'merge_config', 
    'merge_dict', 
    'parse_cli',
]


INCLUDE_KEY = '__include__'


def load_config(file_path, cfg=dict()):
    """
    Load a YAML config file with optional base includes.

    Args:
        file_path: Path to the YAML file.
        cfg: Mutable config dict for recursive merges.

    Returns:
        Merged configuration dict.
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files"

    with open(file_path) as f:
        file_cfg = yaml.load(f, Loader=yaml.Loader)
        if file_cfg is None:
            return {}

    if INCLUDE_KEY in file_cfg:
        base_yamls = list(file_cfg[INCLUDE_KEY])
        for base_yaml in base_yamls:
            if base_yaml.startswith('~'):
                base_yaml = os.path.expanduser(base_yaml)

            if not base_yaml.startswith('/'):
                base_yaml = os.path.join(os.path.dirname(file_path), base_yaml)

            with open(base_yaml) as f:
                base_cfg = load_config(base_yaml, cfg)
                merge_dict(cfg, base_cfg)

    return merge_dict(cfg, file_cfg)


def merge_dict(dct, another_dct, inplace=True) -> Dict:
    """
    Merge another dict into a dict, recursively.

    Args:
        dct: Destination dict.
        another_dct: Source dict.
        inplace: Whether to mutate the destination dict.

    Returns:
        Merged dict.
    """
    def _merge(dct, another) -> Dict:
        for k in another:
            if (k in dct and isinstance(dct[k], dict) and isinstance(another[k], dict)):
                _merge(dct[k], another[k])
            else:
                dct[k] = another[k]

        return dct
    
    if not inplace:
        dct = copy.deepcopy(dct)
    
    return _merge(dct, another_dct)


def dictify(s: str, v: Any) -> Dict:
    if '.' not in s:
        return {s: v}
    key, rest = s.split('.', 1)
    return {key: dictify(rest, v)}


def parse_cli(nargs: List[str]) -> Dict:
    """
    Parse CLI key-value pairs into a nested dict.

    Example:
        a.c=3 b=10 -> {'a': {'c': 3}, 'b': 10}

    Args:
        nargs: List of "key=value" strings.

    Returns:
        Nested dict of parsed values.
    """
    cfg = {}
    if nargs is None or len(nargs) == 0:
        return cfg

    for s in nargs:
        s = s.strip()
        k, v = s.split('=', 1)
        d = dictify(k, yaml.load(v, Loader=yaml.Loader))
        cfg = merge_dict(cfg, d)

    return cfg



def merge_config(cfg, another_cfg=GLOBAL_CONFIG, inplace: bool = False, overwrite: bool = False):
    """
    Merge another config dict into cfg.

    Args:
        cfg: Base config.
        another_cfg: Config to merge in.
        inplace: Whether to mutate cfg.
        overwrite: Whether to overwrite conflicting keys.

    Returns:
        Merged config dict.
    """
    def _merge(dct, another):
        for k in another:
            if k not in dct:
                dct[k] = another[k]
            
            elif isinstance(dct[k], dict) and isinstance(another[k], dict):
                _merge(dct[k], another[k])   
            
            elif overwrite:
                dct[k] = another[k]

        return cfg
    
    if not inplace:
        cfg = copy.deepcopy(cfg)

    return _merge(cfg, another_cfg)
