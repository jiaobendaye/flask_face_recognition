# usage:

zero:
	download the model  [nn4_small2.h5](https://www.jianguoyun.com/p/DaTZKmwQno3qBxinvIMC) and [shape_predictor_68_face_landmarks.dat](https://www.jianguoyun.com/p/DbNGwpoQno3qBxj8u4MC), put them on ./model

first: 
	run the preprocess_dataset.py to create embedded.npy

second:
  	set FLASK_ENV=development && flask run --host=0.0.0.0
	
third:
	curl -X POST -F image=@test.jpg 'http://localhost:5000/recognition'
