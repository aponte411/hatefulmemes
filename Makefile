.EXPORT_ALL_VARIABLES:

TRAIN_PATH=../inputs/data/train.jsonl
DEV_PATH=../inputs/data/dev.jsonl
DATA_DIR=../inputs/data
TEST_PATH=../inputs/data/test.jsonl
OUTPUT_PATH=model-outputs
CHECKPOINT=model-outputs/epoch=0.ckpt
PYTHONPATH=.

train:
	python bin/train.py

submit:
	python bin/submit.py
