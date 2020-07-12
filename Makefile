.EXPORT_ALL_VARIABLES:

TRAIN_PATH=/storage/hm_example_mmf/data/train.jsonl
DEV_PATH=/storage/hm_example_mmf/data/dev.jsonl
DATA_DIR=/storage/hm_example_mmf/data
TEST_PATH=/storage/hm_example_mmf/data/test.jsonl
OUTPUT_PATH=model-outputs
CHECKPOINT=model-outputs/epoch=0_v0.ckpt
PYTHONPATH=.

train:
	python bin/train.py

submit:
	python bin/submit.py
