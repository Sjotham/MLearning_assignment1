run:
	python3 classifier.py

install:
	pip install -r requirements.txt

build:
	python3 setup.py build bdist_wheel

clean:
	rd /s /q build
	rd /s /q dist
	rd /s /q classifier.py.egg-info