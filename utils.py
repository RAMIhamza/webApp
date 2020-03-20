import pickle
import pandas as pd
from preprocessing import preprocessing_
pickle_in = open("model_gmm.pkl","rb")
estimator = pickle.load(pickle_in)
pickle_in.close()


# preprocessing
def preprocessing():
    data_contrats = pd.read_csv('train_contrats_approx.csv',sep=";")
    train_data,test_data = preprocessing_(data_contrats, balance=False, train_size=1.)
    return train_data
# preprocessing
def preprocessing2(IMMAT , ValeurPuissance, Type_Apporteur, Activite , Formule, Classe_Age_Situ_Cont):
    data_contrats = pd.read_csv('train_contrats_approx.csv',sep=";")
    data_contrats = data_contrats.append({"IMMAT":IMMAT , "ValeurPuissance":ValeurPuissance, "Type_Apporteur":Type_Apporteur, "Activite":Activite , "Formule":Formule, "Classe_Age_Situ_Cont":Classe_Age_Situ_Cont},ignore_index=True).fillna(method="pad")
    
    train_data,test_data = preprocessing_(data_contrats, balance=False, train_size=1.)
#     print(train_data.columns)
#     train_data.drop(["Freq_sinitre"],axis=1)
    return train_data.iloc[len(train_data)-1:len(train_data)].drop(['Freq_sinistre'],axis=1)
def prediction(v):
    train_data=preprocessing()
    index=estimator.predict(train_data.sample().drop(["Freq_sinistre"],axis=1))
    return estimator.predict(train_data.sample().drop(["Freq_sinistre"],axis=1))[0]

def prediction2(v):
    index=estimator.predict(v)
    return index[0]