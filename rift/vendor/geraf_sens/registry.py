"""Small dependency shim; does not alter any source model or transform math."""
from torch import nn
class ModelRegistry:
    @staticmethod
    def build(config):
        if not isinstance(config, nn.Module):
            raise TypeError("The port supplies already constructed source networks")
        return config
MODELS=ModelRegistry()
class BaseTransform:
    def __call__(self, results):
        return self.transform(results)
