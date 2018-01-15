.PHONY: tests

tests:
	python3 find_things.py test_images/test1.jpg output_images/test1.jpg
	python3 find_things.py test_images/test2.jpg output_images/test2.jpg
	python3 find_things.py test_images/test3.jpg output_images/test3.jpg
	python3 find_things.py test_images/test4.jpg output_images/test4.jpg
	python3 find_things.py test_images/test5.jpg output_images/test5.jpg
	python3 find_things.py test_images/test6.jpg output_images/test6.jpg
