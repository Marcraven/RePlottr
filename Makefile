clean_dataset:
	rm -rf dataset/*

train_donut:
	python train.py --config config/train_cord.yaml --exp_version "test_experiment"
