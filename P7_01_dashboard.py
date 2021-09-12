import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import shap

### Functions ###

COLOR_BR_r = ['dodgerblue', 'indianred']
COLOR_BR =['#EF553B', '#00CC96']

@st.cache
def histogram(df, x='str', legend=True, client=None):
    '''client = [df_test, input_client] '''
    if x == "TARGET":
        fig = px.histogram(df,
                        x=x,
                        color="TARGET",
                        width=300,
                        height=200,
                        category_orders={"TARGET": [1, 0]},
                        color_discrete_sequence=COLOR_BR)
        fig.update_xaxes(showticklabels=False)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=50))
    else:
        fig = px.histogram(df,
                x=x,
                color="TARGET",
                width=300,
                height=200,
                category_orders={"TARGET": [1, 0]},
                color_discrete_sequence=COLOR_BR,
                barmode="group",
                histnorm='percent')
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    if legend == True:
        fig.update_layout(legend=dict(yanchor="top",xanchor="right"))
    else:
        fig.update_layout(showlegend=False)
    if client:
        client_data = client[0][client[0].SK_ID_CURR ==  client[1]]
        vline = client_data[x].to_numpy()[0]
        fig.add_vline(x=vline, line_width=3, line_dash="dash", line_color="black")
    return fig

### Data ###

train = pd.read_csv('../Dataset/train_small.csv', nrows = 100000)
test = pd.read_csv('../Dataset/test_small.csv', nrows = 10000)
test_ID = test['SK_ID_CURR']
test_features = test.drop(columns=['SK_ID_CURR'])

with open('Final_Model.pkl', 'rb') as file:
    Final_Model = pickle.load(file)

preds_proba = Final_Model.predict_proba(test_features)[:, 1]
preds = Final_Model.predict(test_features)

### Title principal + input client ###

st.write('''
# Client's Scoring

Machine learning model to predict how capable each applicant is
of repaying a loan from [**Home Credit Default Risk**]
(https://www.kaggle.com/c/home-credit-default-risk/data)

***
''')

col1, col2 = st.columns(2)

input_client = col1.selectbox('Select random client ID', test_ID)

### Prediction ###

data_for_prediction = test_features[test['SK_ID_CURR']==input_client]
y_prob = Final_Model.predict_proba(data_for_prediction)
y_prob = [y_prob.flatten()[0], y_prob.flatten()[1]]

if y_prob[1] < y_prob[0]:
    col2.subheader(f"**Successful payment probability.**")
else:
    col2.subheader(f"**Failure payment probability.**")

fig = px.pie(values=y_prob, names=[0,1], color=[0,1], color_discrete_sequence=COLOR_BR_r,
width=230, height=230)
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
# fig.update_layout(legend=dict(yanchor="top",xanchor="right"))
col2.plotly_chart(fig, use_container_width=True)

### Summary plot SHAP Values ###

data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

shap.initjs()
explainer = shap.TreeExplainer(Final_Model)

st.set_option('deprecation.showPyplotGlobalUse', False)
shap_values = explainer.shap_values(test_features)

if st.button('SHAP Summary'):
    st.pyplot(shap.summary_plot(shap_values[1], test_features, max_display=10))

shap_sum = np.abs(shap_values[0]).mean(axis=0)

importance_df = pd.DataFrame([test_features.columns.tolist(), shap_sum.tolist()]).T
importance_df.columns = ['column_name', 'shap_importance']
importance_df = importance_df.sort_values('shap_importance', ascending=False)

most_important_var = importance_df['column_name'][0:5].tolist()

if st.button('Grahps'):
    for x in most_important_var:
        st.plotly_chart(histogram(train, x=x, client=[test, input_client]), use_container_width=True)

symbol = Image.open('Symbol.png')
st.sidebar.image(symbol, use_column_width=True)

st.sidebar.header('User Input parameters')
st.sidebar.slider('DAYS_EMPLOYED', 0, 10, 5)

if st.button('SHAP Values'):
    shap_values = explainer.shap_values(data_for_prediction_array)
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction))
