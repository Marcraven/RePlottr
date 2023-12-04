default: install_requirements

streamlit:
	@streamlit run interface/app.py

streamlit_bw:
	@streamlit run interface/app_bw.py

install_requirements:
	@pip install -r requirements.txt

data_please:
	python interface/CreateData.py

clean_data:
	rm -rf data

train_yolo:
	python ml_logic/yolo/model_yolo.py
