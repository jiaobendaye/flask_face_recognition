usage:
first: run the preprocess_dataset.py to create embedded.npy

second:
  	set FLASK_ENV=development && flask run --host=0.0.0.0
	
third:
	curl -X POST -F image=@test.jpg 'http://localhost:5000/recognition'
