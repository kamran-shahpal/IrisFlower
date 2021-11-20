from flask import Flask,request,jsonify,session , render_template , url_for,redirect


import numpy as np
import joblib
from tensorflow.keras.models import load_model


from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField




app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'




class FlowerForm(FlaskForm):

    sep_len = StringField('Sepal Length')
    sep_wid = StringField('Sepal Width')
    pet_len = StringField('Petal Length')
    pet_wid = StringField('Petal Width')

    Submit = SubmitField('Predict')

@app.route('/',methods = ['GET','POST'])
def index():
    form = FlowerForm()
    if form.validate_on_submit():
        session['sep_len'] = request.form['sep_len']
        session['sep_wid'] = request.form['sep_wid']
        session['pet_len'] = request.form['pet_len']
        session['pet_wid'] = request.form['pet_wid']
        session['Submit']  = request.form['Submit']
        return redirect(url_for('prediction'))
    return render_template('index.html',form = form)



def return_prediction(model,scaler,json_example):
    
    s_len = json_example['sepal_length']
    s_wid = json_example['sepal_width']
    p_len = json_example['petal_length']
    p_wid = json_example['petal_width']
    
    flower = [[s_len,s_wid,p_len,p_wid]]
    flower_scaled = scaler.transform(flower)
    
    classes = np.array(['iris-setosa', 'iris-versicolor', 'iris-virginica'])
    pred_iris = np.argmax(model.predict(flower_scaled),axis = -1)
    
    return classes[pred_iris]

iris_model = load_model('iris_saved_model.h5')
iris_scaler = joblib.load('iris_scaled.pkl')

@app.route('/prediction',methods = ['GET','POST'])
def prediction():
    content = {}
    content['sepal_length'] = session['sep_len']
    content['sepal_width'] = session['sep_wid']
    content['petal_length'] = session['pet_len']
    content['petal_width'] = session['pet_wid']
    
    results = return_prediction(iris_model,iris_scaler,content)
    return render_template('prediction.html' , results = results)
if __name__ == '__main__':
    app.run()