"""smth2smth: video classification project (tracks A and B).

The package is organized as:
    shared/    Reusable building blocks (data, models, engine, io, utils).
    track_a/   Track-A specific overrides.
    track_b/   Track-B specific overrides.
    pipelines/ Hydra-driven entrypoints (train, evaluate, submit).
    cli/       Thin command-line wrappers around the pipelines.
"""

__version__ = "0.1.0"
