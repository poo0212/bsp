from flask import Flask, escape, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
model=pickle.load(open("models\model.sav","rb"))
app = Flask(__name__)
@app.route('/analysis')
def analysis():
    return render_template("stroke.html")

@app.route('/', methods=['GET' , 'POST'])
def home():
    if request.method == "POST":
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        married = request.form['married']
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        gender = request.form['gender']
        work = request.form['work']
        residence = request.form['residence']
        smoking = request.form['smoking']

        # married
        if (married == "Yes"):
            ever_married = 1
        else:
            ever_married = 0

        # gender
        if (gender == "Male"):
            sex_Female = 0
            sex_Male  = 1
            sex_Other = 0
        elif (gender == "Female"):
            sex_Female = 1
            sex_Male =  0
            sex_Other = 0
        elif (gender == "Other"):
            sex_Female = 0
            sex_Male = 0
            sex_Other = 1
        else:
            sex_Female = 0
            sex_Male  = 0
            sex_Other = 0


        # work  type
        if (work == 'Self-employed'):
            work_Govt_job=0
            work_Never_worked = 0
            work_Private = 0
            work_Self_employed = 1
            work_children = 0
        elif (work == 'Govt_job'):
            work_Govt_job = 1
            work_Never_worked = 0
            work_Private = 0
            work_Self_employed = 0
            work_children = 0
        elif (work == 'Private'):
            work_Govt_job = 0
            work_Never_worked = 0
            work_Private = 1
            work_Self_employed = 0
            work_children = 0
        elif (work == "children"):
            work_Govt_job = 0
            work_Never_worked = 0
            work_Private = 0
            work_Self_employed = 0
            work_children = 1
        elif (work == "Never_worked"):
            work_Govt_job = 0
            work_Never_worked = 1
            work_Private = 0
            work_Self_employed = 0
            work_children = 0
        else:
            work_Govt_job = 0
            work_Never_worked = 0
            work_Private = 0
            work_Self_employed = 0
            work_children = 0

        # residence type
        if (residence == "Urban"):
            residence_Urban = 1
            residence_Rural = 0
        elif (residence == 'Rural'):
            residence_Urban = 0
            residence_Rural=1
        else:
            residence_Rural = 0
            residence_Urban = 0

        # smoking sttaus
        if (smoking == 'formerly smoke'):
            smoke_Unknown=0
            smoke_formerly = 1
            smoke_never = 0
            smoke_smokes = 0
        elif (smoking == 'Unknown'):
            smoke_Unknown = 1
            smoke_formerly = 0
            smoke_never = 0
            smoke_smokes = 0
        elif (smoking == 'smokes'):
            smoke_Unknown = 0
            smoke_formerly = 0
            smoke_never = 0
            smoke_smokes = 1

        elif (smoking == "never smoked"):
            smoke_Unknown = 0
            smoke_formerly = 0
            smoke_never = 1
            smoke_smokes = 0
        else:
            smoke_Unknown = 0
            smoke_formerly = 0
            smoke_never = 0
            smoke_smokes = 0

        feature = ([[age,hypertension,heart_disease,ever_married,avg_glucose_level,bmi,sex_Female,sex_Male,work_Govt_job,work_Never_worked,work_Private,work_Self_employed,work_children,residence_Rural,residence_Urban,smoke_formerly,smoke_Unknown,smoke_never,smoke_smokes]])
        input_data_as_numpy_array = np.asarray(feature)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        stroke = model.predict(input_data_reshaped)
        # print(prediction)
        #
        if stroke == 0:

            #if (hypertension==1 & heart_disease==1):
                #return render_template( "remedies.html")
            #else:
            stroke="NO"
            
        else:
            stroke = "YES"

        return render_template("index.html", prediction_text="Chance of Stroke Prediction is --> {}".format(stroke))


    else:
        return render_template("index.html")

if __name__ == "__main__":
        app.run(debug=True)