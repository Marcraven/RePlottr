default: install_requirements

streamlit:
	@streamlit run frontend/interface/app.py

streamlit_bw:
	@streamlit run frontend/interface/app_bw.py

install_requirements:
	@pip install -r requirements.txt

data_please:
	python donutplot/interface/CreateData.py

clean_data:
	rm -rf data

train_yolo:
	python donutplot/ml_logic/yolo/yolo.py
