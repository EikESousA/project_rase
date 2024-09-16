from os.path import abspath, join


class DatasetConfig:
    CABRITA_PATH = abspath(join("app", "datasets", "cabrita_dataset.json"))
    RASE_PATH = abspath(join("app", "datasets", "rase_dataset.json"))
