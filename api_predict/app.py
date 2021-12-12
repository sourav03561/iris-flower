from flask import Flask,request
import joblib
import numpy as np
app = Flask(__name__)
model_jl = joblib.load(open("flower-v1.jl","rb"))
@app.route("/api_predict",methods=["GET","POST"])
def predict():
    if request.method == "GET":
        return "Please send Post Request"
    elif request.method == "POST":
        data = request.get_json()

        sepal_length = data["sepal_length"]
        sepal_width = data["sepal_width"]
        petal_length = data["petal_length"]
        petal_width = data["petal_width"]
        arr = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
        return str(model_jl.predict(arr))



if __name__ == '__main__':
    app.run()