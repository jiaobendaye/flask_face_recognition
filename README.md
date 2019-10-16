usage:
  	set FLASK_ENV=development && flask run --host=0.0.0.0
	curl -X POST -F image=@test.jpg 'http://localhost:5000/recognition'
