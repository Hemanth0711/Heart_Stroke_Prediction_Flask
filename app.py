
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, static_url_path='/static')

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store'
    return response

# prediction function
def ValuePredictor(to_predict_list):
	to_predict = np.array(to_predict_list).reshape(1, -1)
	loaded_model = pickle.load(open("models/RandomForestClassifier_bagging_best_model.h5", "rb"))
	result = loaded_model.predict(to_predict)
	return result[0]


@app.route('/', methods = ['GET'])
def hello_world():
    return render_template('index.html', prediction = None)


@app.route('/', methods = ['POST'])
def result():
	if request.method == 'POST':
		to_predict_list = request.form.to_dict()
		to_predict_list = list(to_predict_list.values())
		to_predict_list = list(map(float, to_predict_list))
		result = ValuePredictor(to_predict_list)
		if int(result)== 1:
			prediction ='Heart Stroke Detected '+str(result)
		else:
			prediction ='No heart stroke '+str(result)
		return render_template("index.html", prediction = prediction)

if __name__ == '__main__':
    app.run(port = 3000, debug = True)
