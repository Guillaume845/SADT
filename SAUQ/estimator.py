class Estimator:
    """Minimal stub satisfying SpectralRAUQ's inheritance requirement."""

    def __init__(self, dependencies, level):
        self.dependencies = dependencies
        self.level = level

    def __call__(self, stats):
        raise NotImplementedError
