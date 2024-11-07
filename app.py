from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import io
import base64

app = Flask(__name__)

# Load data
df = pd.read_csv('diabetes.csv')

# Prepare data for model
x = df.drop(['Outcome'], axis=1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve user inputs and convert them to appropriate data types
    try:
        pregnancies = int(request.form.get('pregnancies'))
        glucose = float(request.form.get('glucose'))
        bp = float(request.form.get('bp'))
        skinthickness = float(request.form.get('skinthickness'))
        insulin = float(request.form.get('insulin'))
        bmi = float(request.form.get('bmi'))
        dpf = float(request.form.get('dpf'))
        age = int(request.form.get('age'))
    except ValueError:
        return render_template('index.html', output="Invalid input values. Please enter valid data.")

    # Prepare data for model prediction
    user_data = pd.DataFrame({
        'pregnancies': [pregnancies],
        'glucose': [glucose],
        'bp': [bp],
        'skinthickness': [skinthickness],
        'insulin': [insulin],
        'bmi': [bmi],
        'dpf': [dpf],
        'age': [age]
    })

    # Model prediction
    user_result = rf.predict(user_data)

    # Visualizations
    color = 'blue' if user_result[0] == 0 else 'red'

    # Generate plots
    fig_preg = create_plot('age', 'pregnancies', df, user_data, 'Outcome', 'pregnancy count Graph', color)
    fig_glucose = create_plot('age', 'glucose', df, user_data, 'Outcome', 'glucose Value Graph', color)
    fig_bp = create_plot('age', 'bp', df, user_data, 'Outcome', 'bp Value Graph', color)

    # Output message
    output = 'You are not Diabetic' if user_result[0] == 0 else 'You are Diabetic'
    accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100

    return render_template('index.html', output=output, accuracy=accuracy, 
                           fig_preg=fig_preg, fig_glucose=fig_glucose, fig_bp=fig_bp)

def create_plot(x_col, y_col, df, user_data, hue_col, title, color):
    fig = plt.figure()
    sns.scatterplot(x=x_col, y=y_col, data=df, hue=hue_col)
    sns.scatterplot(x=user_data[x_col], y=user_data[y_col], s=150, color=color)
    plt.title(title)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 220, 10))

    # Convert plot to PNG image and encode to base64 for embedding in HTML
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

if __name__ == '__main__':
    app.run(debug=True)