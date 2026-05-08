"""
File with Optimizers
"""

# =============================================================================
# Optimizer (base class)
# =============================================================================

class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        # Aggregate parameters except None into the list
        params = [p for p in self.target.params() if p.grad is not None]

        # preprocessing (optional)
        for f in self.hooks:
            f(params)

        # updating parameters
        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)