default: install_requirements

streamlit:
	@streamlit run frontend/app.py

install_requirements:
	@pip install -r requirements.txt

train_yolo:
	python ObjectRecognition/train_yolo.py

clean_yolo_dataset:
	rm -rf ObjectRecognition/yolo/dataset/*

yolo_dataset:
	python CreateData_Yolo.py

clean_io_dataset:
	rm -rf datas/*

io_dataset:
	python interface/CreateData_IO.py

box:
	python ml_logic/yolo/utils/draw_box.py

boxes:
	python ml_logic/yolo/utils/draw_boxes.py
