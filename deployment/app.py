from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model/rf_modelsmote.pkl', 'rb'))

@app.route("/predict", methods=["POST"])
def predict ():
    age = float(request.form['age'])
    creatinine_phosphokinase = float(request.form['creatinine_phosphokinase'])
    ejection_fraction = float(request.form['ejection_fraction'])
    platelets = float(request.form['platelets'])
    serum_creatinine = float(request.form['serum_creatinine'])
    serum_sodium = float(request.form['serum_sodium'])
    time = float(request.form['time'])
    
    columns = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets",
               "serum_creatinine", "serum_sodium", "time"]
    columns = np.array(columns)
    
    x = np.zeros(len(columns))
    x[0] = age
    x[1] = creatinine_phosphokinase
    x[2] = ejection_fraction
    x[3] = platelets
    x[4] = serum_creatinine
    x[5] = serum_sodium
    x[6] = time
    
    pred = model.predict([x])[0]
    
    if pred == 1:
        output = "The patient is expected to die"
        return render_template("index.html", prediction = output, age = age, creatinine_phosphokinase = creatinine_phosphokinase, 
                               ejection_fraction = ejection_fraction, platelets = platelets, serum_creatinine = serum_creatinine, 
                               serum_sodium = serum_sodium, time = time)
    elif pred == 0:
        output = "The patient is not expected to die"
        return render_template("index.html", prediction = output, age = age, creatinine_phosphokinase = creatinine_phosphokinase, 
                               ejection_fraction = ejection_fraction, platelets = platelets, serum_creatinine = serum_creatinine, 
                               serum_sodium = serum_sodium, time = time)
         
@app.route("/")
def index():
    return  render_template('index.html')


    

if __name__ == "__main__":
    app.run(debug=True) 
    
    
