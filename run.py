import json
import shutil
import sys

from allennlp.commands import main

config_file = "train_configs/defaults.jsonnet"

# Use overrides to train on CPU.
overrides = json.dumps({"trainer": {"cuda_device": -1}})

exp = "experiments/multi_all"
TRAINING = True

if TRAINING:
    # Training will fail if the serialization directory already
    # has stuff in it. If you are running the same training loop
    # over and over again for debugging purposes, it will.
    # Hence we wipe it out in advance.
    # BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
    shutil.rmtree(exp, ignore_errors=True)

    sys.argv = [
        "allennlp",  # command name, not used by main
        "train",
        config_file,
        "-s", exp,
        "--include-package", "dataset_readers.spider",
        "--include-package", "models.semantic_parsing.spider_parser",
        # "-o", overrides,
        # "--recover"
    ]
else:
    sys.argv = [
        "allennlp",
        "predict", exp,
        "sparc/dev.json",
        "--predictor", "spider",
        "--use-dataset-reader",
        "--cuda-device=1",
        "--silent",
        "--output-file", exp + "/prediction-t.sql",
        "--include-package", "models.semantic_parsing.spider_parser", 
        "--include-package", "dataset_readers.spider",
        "--include-package", "predictors.spider_predictor",
        "--weights-file", exp + "/best.th",
        "--batch-size", "1",
        "-o", "{\"dataset_reader\":{\"keep_if_unparsable\":true}, \"validation_iterator\":{\"batch_size\": 1}}"
    ]

main()