# lidar2d/registry.py
REGISTRY = {"proj": {}, "rast": {}, "enc": {}}

def register(kind, name):
    def deco(cls):
        REGISTRY[kind][name] = cls
        return cls
    return deco

def build(kind, name, **kwargs):
    return REGISTRY[kind][name](**kwargs)
