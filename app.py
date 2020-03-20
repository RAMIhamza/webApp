# Importing librairies
from flask import Flask,render_template,url_for,request, json
from utils import prediction2 , preprocessing2

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        IMMAT = request.form['IMMAT']
        ValeurPuissance = request.form['ValeurPuissance']
        Type_Apporteur = request.form['Type_Apporteur']
        Activite = request.form['Activite']
        Formule = request.form['Formule']
        Classe_Age_Situ_Cont = request.form['Classe_Age_Situ_Cont']
        prediction_ = prediction2(preprocessing2(float(IMMAT) , float(ValeurPuissance), float(Type_Apporteur), float(Activite) , float(Formule), Classe_Age_Situ_Cont))
    return render_template('home.html',prediction = ["The predicted class using pretrained GMM is : ",prediction_],input1=IMMAT,input2=ValeurPuissance,input3=Type_Apporteur,input4=Activite,input5=Formule,input6=Classe_Age_Situ_Cont)

if __name__ == '__main__':
    app.run(debug=True)