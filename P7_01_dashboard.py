import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import shap

st.set_page_config(layout="wide")

### Functions ###

COLOR_BR_r = ['#00CC96', '#e03838']
COLOR_BR =['#e03838', '#00CC96']

@st.cache
def histogram(df, x='str', legend=True, client=None):
    '''client = [df_test, input_client] '''
    if x == "TARGET":
        fig = px.histogram(df,
                        x=x,
                        color="TARGET",
                        width=300,
                        height=300,
                        category_orders={"TARGET": [1, 0]},
                        color_discrete_sequence=COLOR_BR,
                        marginal='box')
        fig.update_xaxes(showticklabels=False)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    else:
        fig = px.histogram(df,
                x=x,
                color="TARGET",
                width=300,
                height=250,
                category_orders={"TARGET": [1, 0]},
                color_discrete_sequence=COLOR_BR,
                barmode="group",
                histnorm='percent',
                marginal='box')
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

train = pd.read_csv('Datasets/reduced_train.csv')
test = pd.read_csv('Datasets/reduced_test.csv')
test_ID = test['SK_ID_CURR']
test_features = test.drop(columns=['SK_ID_CURR'])

with open('Model/Final_Model.pkl', 'rb') as file:
    Final_Model = pickle.load(file)

### Title principal + input client ###

col1, col2 = st.columns((252,1024))

symbol = Image.open('Images/Symbol.png')
hcg = Image.open('Images/Home Credit Group.png')
col1.image(symbol, use_column_width=True)
col2.image(hcg, use_column_width=True)

st.write('''
# Client's Scoring

Machine learning model to predict how capable each applicant is
of repaying a loan from [**Home Credit Default Risk**]
(https://www.kaggle.com/c/home-credit-default-risk/data).
''')

st.markdown(':arrow_upper_left: Click on the left sidebar to adjust most important variables and improve the prediction score.')

st.write(' *** ')
col1, col2 = st.columns(2)

input_client = col1.selectbox('Select Client ID', test_ID)

### Prediction ###

data_for_prediction = test_features[test['SK_ID_CURR']==input_client]
y_prob = Final_Model.predict_proba(data_for_prediction)
y_prob = [y_prob.flatten()[0], y_prob.flatten()[1]]

fig = px.pie(values=y_prob, names=['Success', 'Failure '], color=[0,1], color_discrete_sequence=COLOR_BR_r,
width=230, height=230)
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
# fig.update_layout(legend=dict(yanchor="top",xanchor="right"))
col2.plotly_chart(fig, use_container_width=True)

if y_prob[1] < y_prob[0]:
    col2.subheader('**Successful payment probability.**')
else:
    col2.subheader('**Failure payment probability.**')

### Summary plot SHAP Values ###

data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

shap.initjs()
explainer = shap.TreeExplainer(Final_Model)

st.set_option('deprecation.showPyplotGlobalUse', False)
shap_values = explainer.shap_values(test_features)

shap_sum = np.abs(shap_values[0]).mean(axis=0)

importance_df = pd.DataFrame([test_features.columns.tolist(), shap_sum.tolist()]).T
importance_df.columns = ['column_name', 'shap_importance']
importance_df = importance_df.sort_values('shap_importance', ascending=False)

most_important_var = importance_df['column_name'][0:3].tolist()

st.write(''' *** ''')

st.subheader('''**Summary plot with SHAP Values:**''')

st.pyplot(shap.summary_plot(shap_values[1], test_features, max_display=10))

st.write(''' *** ''')

st.subheader('''**Most important variables:**''')

for x in most_important_var:
    st.plotly_chart(histogram(train, x=x, client=[test, input_client]), use_container_width=True)

st.write(''' *** ''')

st.subheader('''**Force plot with SHAP Values:**''')

shap_values = explainer.shap_values(data_for_prediction_array)
st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction, plot_cmap=COLOR_BR_r))

st.write(''' *** ''')

### Sidebar ###

st.sidebar.write('# Adjustable parameters:')

def adjusted_variables():

    var_1 = st.sidebar.slider(most_important_var[0], train[most_important_var[0]].min(), train[most_important_var[0]].max(), float(data_for_prediction[most_important_var[0]]))
    var_2 = st.sidebar.slider(most_important_var[1], train[most_important_var[1]].min(), train[most_important_var[1]].max(), float(data_for_prediction[most_important_var[1]]))
    var_3 = st.sidebar.slider(most_important_var[2], train[most_important_var[2]].min(), train[most_important_var[2]].max(), float(data_for_prediction[most_important_var[2]]))
    #var_4 = st.sidebar.slider(most_important_var[3], train[most_important_var[3]].min(), train[most_important_var[3]].max(), float(data_for_prediction[most_important_var[3]]))
    #var_5 = st.sidebar.slider(most_important_var[4], float(train[most_important_var[4]].min()), float(train[most_important_var[4]].max()), float(data_for_prediction[most_important_var[4]]))

    dict = {most_important_var[0] : [var_1],
            most_important_var[1] : [var_2],
            most_important_var[2] : [var_3]}
            #most_important_var[3] : [var_4],
            #most_important_var[4] : [var_5]}

    data_adjusted = data_for_prediction.copy()

    for key,value in dict.items():
        data_adjusted[key] = value

    return data_adjusted

### Adjusted prediction ###

adj = adjusted_variables()

y_prob_adj = Final_Model.predict_proba(adj)
y_prob_adj = [y_prob_adj.flatten()[0], y_prob_adj.flatten()[1]]

st.sidebar.write(''' *** ''')

st.sidebar.write('# Result on predictions:')

fig = px.pie(values=y_prob_adj, names=['Success', 'Failure '], color=[0,1], color_discrete_sequence=COLOR_BR_r,
width=230, height=230)
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
# fig.update_layout(legend=dict(yanchor="top",xanchor="right"))
st.sidebar.plotly_chart(fig, use_container_width=True)

if y_prob_adj[1] < y_prob_adj[0]:
    st.sidebar.subheader('**Successful payment probability after adjusting variables.**')
else:
    st.sidebar.subheader('**Failure payment probability after adjusting variables.**')
