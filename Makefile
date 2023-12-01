default: install_requirements

streamlit:
	@streamlit run frontend/app.py

install_requirements:
	@pip install -r requirements.txt

clean_io_dataset:
	rm -rf data/*

io_dataset:
	python interface/CreateData_IO.py

box:
	python ml_logic/yolo/utils/draw_box.py

boxes:
	python ml_logic/yolo/utils/draw_boxes.py
