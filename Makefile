default: install_requirements

streamlit:
	@streamlit run frontend/app.py

install_requirements:
	@pip install -r requirements.txt

clean_dataset:
	rm -rf TextRecognition/DonutApproach/dataset/*

dataset_donut:
	python CreateData_Donut.py

train_donut:
	python TextRecognition/DonutApproach/train.py --config TextRecognition/DonutApproach/config/train_cord.yaml --exp_version "test_experiment"
