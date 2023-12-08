default: install_requirements

streamlit:
	@streamlit run frontend/app.py

uvicorn:
	@uvicorn donutplot.api.fast:app --reload

install_requirements:
	@pip install -r requirements.txt

data_please:
	python donutplot/interface/CreateData.py

clean_data:
	rm -rf ~/.donutplot/data

train_yolo:
	python donutplot/ml_logic/yolo/yolo.py

benchmark:
	python donutplot/interface/benchmark.py

clean_create_benchmark:
	rm -rf ~/.donutplot/data
	python donutplot/interface/CreateData.py
	python donutplot/interface/benchmark.py
