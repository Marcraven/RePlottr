default: install_requirements

streamlit:
	@streamlit run frontend/app.py

install_requirements:
	@pip install -r requirements.txt

clean_all_datasets:
	rm -rf dataset/*
	rm -rf TextRecognition/DonutApproach/dataset/*
	rm -rf ObjectRecognition/yolo/dataset/*

all_datasets:
	python CreateData_IO.py
	python CreateData_Donut.py
	python CreateData_Yolo.py

clean_donut_dataset:
	rm -rf TextRecognition/DonutApproach/dataset/*

donut_dataset:
	python CreateData_Donut.py

train_donut:
	python TextRecognition/DonutApproach/train.py --config TextRecognition/DonutApproach/config/train_cord.yaml --exp_version "test_experiment"

clean_yolo_dataset:
	rm -rf ObjectRecognition/yolo/dataset/*

yolo_dataset:
	python CreateData_Yolo.py

clean_io_dataset:
	rm -rf dataset/*

io_dataset:
	python CreateData_IO.py

box:
	python utils/utils.py
