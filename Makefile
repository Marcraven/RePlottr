default: install_requirements

streamlit:
	@streamlit run frontend/app.py

install_requirements:
	@pip install -r requirements.txt

clean_donut_dataset:
	rm -rf TextRecognition/DonutApproach/dataset/*

donut_dataset:
	python CreateData_Donut.py

train_donut:
	python donut-master/train.py --config donut-master/config/train_plot.yaml --exp_version "test_experiment"

clean_yolo_dataset:
	rm -rf ObjectRecognition/yolo/dataset/*

yolo_dataset:
	python CreateData_Yolo.py
