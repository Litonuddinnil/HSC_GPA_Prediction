import gradio as gr 
import pandas as pd 
import numpy as np 
import pickle 

#loaded model pickle file 
with open("student_rf_pipeline.pkl", "rb") as file:
    model = pickle.load(file)

#main logic 
def predict_gpa(
    st_gender, st_age, st_address, fam_size, Pstatus,
    mother_edu, father_edu, mother_job, father_job,
    st_relationship, smoker, tuition_fee,
    time_with_friends, ssc_result
):
    input_data = pd.DataFrame({
        'gender': [st_gender],
        'age': [st_age],
        'address': [st_address],
        'famsize': [fam_size],
        'Pstatus': [Pstatus],
        'M_Edu': [mother_edu],
        'F_Edu': [father_edu],
        'M_Job': [mother_job],
        'F_Job': [father_job],
        'relationship': [st_relationship],
        'smoker': [smoker],
        'tuition_fee': [tuition_fee],
        'time_friends': [time_with_friends],
        'ssc_result': [ssc_result]
    })

    prediction = model.predict(input_data)
    prediction = np.clip(prediction, 0, 5)

    return f"Predicted HSC GPA: {prediction[0]:.2f}"

inputs=[ 
    gr.Dropdown(choices=['M', 'F'], label="Student  Gender", value='M'),
    gr.Slider(15, 30, step=1, label="Student Age", value=20),
    gr.Dropdown(choices=['Urban', 'Rural'], label="Student Address", value='Urban'),
    gr.Dropdown(choices=['LE3', 'GT3'], label="Family Size", value='GT3'),
    gr.Dropdown(choices=['Together', 'Apart'], label="Parent's Cohabitation Status", value='Together'),   
    gr.Dropdown(choices=[0, 1, 2, 3, 4], label="Mother's Education Level", value=2),
    gr.Dropdown(choices=[0, 1, 2, 3, 4], label="Father's Education Level", value=2),
    gr.Dropdown(choices=['teacher', 'health', 'services', 'at_home', 'other'], label="Mother's Job", value='services'),
    gr.Dropdown(choices=['teacher', 'health', 'services', 'at_home', 'other'], label="Father's Job", value='services'),
    gr.Dropdown(choices=['family', 'friends', 'course', 'other'], label="Student's Relationship Status", value='friends'),
    gr.Radio(choices=['yes', 'no'], label="Smoker", value='no'),
    gr.Slider(0, 20000, step=100, label="Tuition Fee", value=5000),
    gr.Slider(0, 10, step=1, label="Time Spent with Friends (hours/week)", value=5),
    gr.Number(label="SSC Result (GPA out of 5)")
]

# interface 
app = gr.Interface(
    fn= predict_gpa,
    inputs= inputs,
    outputs=gr.Textbox(label="Output"),
    title="Student GPA Prediction",
    description="Predict the GPA of a student based on various features."
)

#lanuch 
app.launch(share=True)
