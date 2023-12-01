default: install_requirements

streamlit:
	@streamlit run frontend/app.py

install_requirements:
	@pip install -r requirements.txt

data:
	python interface/CreateData.py

clean_data:
	rm -rf /data/*

train_yolo:
	python ml_logic/yolo/model_yolo.py
