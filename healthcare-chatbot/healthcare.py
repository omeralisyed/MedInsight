import time
import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
import streamlit as st
from streamlit_lottie import st_lottie
import json
def load_lottie_file(filepath:str):
    with open(filepath,"r") as f:
        return json.load(f)
    
lottie_file1 =load_lottie_file('./DoctorAnimation.json')
lottie_file2 =load_lottie_file('./Application.json')
warnings.filterwarnings("ignore", category=DeprecationWarning)


training = pd.read_csv('./Data/Training.csv')
testing= pd.read_csv('./Data/Testing.csv')
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y 


reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)


clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
# st.write(clf.score(x_train,y_train))
# print ("cross result========")
scores = cross_val_score(clf, x_test, y_test, cv=3)
# st.write(scores)
with st.sidebar:
    st_lottie(lottie_file1,speed=0.5,reverse=False,height=150,width=150)
    # st.write(f"Accuracy : {round(scores.mean()*100)}%")
    st.title('MedInsight')
    model=SVC()
    model.fit(x_train,y_train)
    # st.write("### for svm: ")
    # st.write(f"Accuracy : {round(model.score(x_test,y_test)*100)}%")

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

def readn(nstr):
    engine = pyttsx3.init()

    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()


severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index
def calc_condition(exp,days):
    sum=0
    # time.sleep(7)
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        st.error("You should take the consultation from doctor. ")
    else:
        st.success("It might not be that bad but you should take precautions.")


def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)




def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)


def getInfo():
    c1,c2 = st.columns([0.4,0.6])
    with c1:
        st_lottie(lottie_file2,speed=0.5,reverse=False,height=150,width=150)
    with c2:
        # st_lottie(lottie_file2,speed=0.5,reverse=False,height=200,width=250)
        st.markdown('\n\n')
        st.markdown('\n\n')
        st.title("MedInsight",anchor=False)
    # st.write("\nYour Name? \t\t\t\t->",)
    name=st.text_input("Enter Your Name:")
    # st.warning(name)
    if name:
       return name

def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]
def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    while True:

        # st.write("Select the symptom you are experiencing")
        symp = cols.to_list()
        disease_input = st.selectbox("Select the symptom you are experiencing",symp)
        conf,cnf_dis=check_pattern(chk_dis,disease_input)

        if conf==1:
            st.write("searches related to input: ")
            for num,it in enumerate(cnf_dis):
                st.error(f"{num}){it}")
            if num!=0:
                conf_inp = int(st.radio(f"Select the one you meant (0 - {num}): ",[0,1],horizontal=True))
            else:
                conf_inp=0

            disease_input=cnf_dis[conf_inp]
            break
            # st.write("Did you mean: ",cnf_dis,"?(yes/no) :",end="")
            # conf_inp = input("")
            # if(conf_inp=="yes"):
            #     break
        else:
            st.write("Enter valid symptom.")

    while True:
        if disease_input:
            try:
                st.write("Okay. From how many days ? : ")
                num_days=int(st.number_input('Enter Number of Days',min_value=0,value=1))
                if num_days <=0:
                    st.success("No Need to worry")
                    return
                break
            except:
                st.write("Enter valid input.")
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            # st.write( "You may have " +  present_disease )
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            # dis_list=list(symptoms_present)
            # if len(dis_list)!=0:
            #     st.write("symptoms present  " + str(list(symptoms_present)))
            # st.write("symptoms given "  +  str(list(symptoms_given)) )
            # time.sleep(5)
            st.write("Are you experiencing any ")
            symptoms_exp=[]
            i = 1
            c1,c2=st.columns([0.5,0.5])
            for syms in list(symptoms_given):
                inp=""
                while True:
                    i = i+1
                    if (i%2)==0:
                        with c1:
                            st.write(f"{syms},? : ")
                            inp=st.selectbox("select",['yes','no'],key=i,index=1)
                            if(inp=="yes" or inp=="no"):
                                break
                            else:
                                st.warning("provide proper answers i.e. (yes/no) : ")
                    elif (i%2)!=0:
                        with c2:
                            st.write(f"{syms},? : ")
                            inp=st.selectbox("select",['yes','no'],key=i,index=1)
                            if(inp=="yes" or inp=="no"):
                                break
                            else:
                                st.warning("provide proper answers i.e. (yes/no) : ")

                if(inp=="yes"):
                    symptoms_exp.append(syms)


            second_prediction=sec_predict(symptoms_exp)
            # st.write(second_prediction)
            calc_condition(symptoms_exp,num_days)
            if(present_disease[0]==second_prediction[0]):
                st.warning(f"You may have {present_disease[0]}")
                st.info(description_list[present_disease[0]])

                # readn(f"You may have {present_disease[0]}")
                # readn(f"{description_list[present_disease[0]]}")

            else:
                st.warning(f"You may have  {present_disease[0]} or  {second_prediction[0]}")
                st.info(description_list[present_disease[0]])
                st.info(description_list[second_prediction[0]])

            # st.write(description_list[present_disease[0]])
            precution_list=precautionDictionary[present_disease[0]]
            st.success("Take following measures : ")
            for  i,j in enumerate(precution_list):
                st.write(f"{i+1}){j}")

            # confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
            # st.write("confidence level is " + str(confidence_level))

    recurse(0, 1)
getSeverityDict()
getDescription()
getprecautionDict()
name = getInfo()
if name is not None:
    st.success(f"Hello {name}")
    tree_to_code(clf,cols)
st.write("----------------------------------------------------------------------------------------")

