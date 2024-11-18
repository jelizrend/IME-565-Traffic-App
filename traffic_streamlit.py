# Import libraries
import streamlit as st
import pandas as pd
import pickle
import sklearn
import warnings
import xgboost as xgb
warnings.filterwarnings('ignore')


# Set up the app title and image
st.markdown('# :rainbow[Traffic Volume Predictor]', ) 
st.write("Utilize our advanced Machine Learning application to predict traffic volume.") 
st.image('traffic_image.gif', use_column_width = True)

# Reading the pickle file that we created before 
reg_pickle = open('reg_traffic.pickle', 'rb') 
reg_model = pickle.load(reg_pickle) 
reg_pickle.close()


st.info('Please choose a data input method to proceed', icon= "ℹ️")


#to check if button clicked
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

#alpha value slider
alpha_val = st.slider('Select alpha value for prediction intervals', min_value=0.01, max_value=0.50, value= 0.10)


# Load the default dataset
default_df = pd.read_csv('Traffic_Volume.csv')

# side bar stuffs
st.sidebar.image('traffic_sidebar.jpg', use_column_width=True, caption='Traffic Volume Predictor')
st.sidebar.subheader('Input Features')
st.sidebar.write('You can either upload your data file or manually enter input features')


with st.sidebar.expander('Option 1: Upload CSV File'):
    st.write('Upload a CSV file containing traffic details.')
    user_traffic = st.file_uploader('Choose a CSV File')
    st.write('# Sample Data Format for Upload')
    st.write(default_df.head())
    st.warning('Ensure your uploaded file has the same column and data types as shown above.', icon="⚠️")

#make function to get max and min for following 

with st.sidebar.expander('Option 2: Fill Out Form'):
    with st.form('user_inputs_form'):
        
        st.header("Enter the traffic details manually using the form below.")
        
        holiday = st.selectbox('Choose whether today is a designated holiday or not', options=['None','Christmas Day','Columbus Day','Independence Day','Labor Day','Martin Luther King Jr Day','Memorial Day','New Years Day'])
        temp = st.number_input('Average temperature in Kelvin', min_value=0.0, max_value=251.4, value=100.0, step=1.5)
        rain_1hr = st.number_input('Amount in mm of rain that occurred in the hour', min_value=0.0, max_value=4.5, value=2.0, step=0.1)   
        snow_1hr = st.number_input('Amount in mm of snow that occurred in the hour', min_value=0.0, max_value=0.6, value=0.3, step=0.01)
        clouds_all = st.number_input('Percentage of cloud cover', min_value=0.0, max_value=100.0, value=25.0, step=1.0)
        weather_main = st.selectbox('Choose the current weather', options=['Clear','Clouds','Drizzle','Fog','Haze','Mist','Rain','Smoke','Snow','Squall','Thunderstorm'])
        month = st.selectbox('Choose month', options=['January', 'February','March','April','May','June','July','August','September','October','November','December'])
        weekday = st.selectbox('Choose day of the week', options=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        hour = st.selectbox('Choose hour', options=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
        submit_button = st.form_submit_button("Submit Form Data", on_click=click_button)

#if option 1 chosen
if user_traffic is not None:

    st.success('CSV file uploaded successfully.')

    user_df = pd.read_csv(user_traffic)

    user_len = len(user_df)
    
    #Encode the inputs for model prediction
    encode_df = default_df.copy()
    encode_df = encode_df.drop(columns=['traffic_volume'])


    # Combine the list of user data as a row to default_df
    encode_df = pd.concat([encode_df, user_df], ignore_index=True)

    # Create dummies for encode_df
    # encode_dummy_df = pd.get_dummies(encode_df)

    # Extract encoded user data
    user_encoded_df = encode_df.tail(user_len)

    # Get the prediction with its intervals
    alpha = alpha_val # For 90% confidence level
    prediction = reg_model.predict(user_encoded_df)
    intervals = reg_model.predict(user_encoded_df)
    pred_value = prediction[0]
    # lower_limit = intervals[:, 0]
    # upper_limit = intervals[:, 1]


    # Show the prediction on the app
    st.write(f"### **Prediction Results with {(1-alpha)*100:.2f}% Confidence Interval**")

    user_pred_df = user_df
    user_pred_df['Predicted Price'] = prediction.round(2)
    # user_pred_df['Lower Price Limit'] = lower_limit
    # user_pred_df['Upper Price Limit'] = upper_limit
    
    # #Set negative values to zero
    # user_pred_df['Lower Price Limit'] = user_pred_df['Lower Price Limit'].apply(lambda x: 0 if x < 0 else x)


    st.write(user_pred_df)

#If option 2 chosen:
if st.session_state.clicked:
    
    st.success('Form data submitted successfully.')

    #Encode the inputs for model prediction
    encode_df = default_df.copy()
    encode_df = encode_df.drop(columns=['traffic_volume'])

    st.write(encode_df)

    # Combine the list of user data as a row to default_df
    encode_df.loc[len(encode_df)] = [holiday, temp, rain_1hr, snow_1hr, clouds_all, weather_main, month, weekday, hour]

    # # Create dummies for encode_df
    # encode_dummy_df = pd.get_dummies(encode_df)

    # Extract encoded user data
    user_encoded_df = encode_df.tail(1)

    # Get the prediction with its intervals
    alpha = alpha_val 
    prediction = reg_model.predict(user_encoded_df, alpha = alpha)
    pred_value = prediction[0]
    # lower_limit = intervals[:, 0]
    # upper_limit = intervals[:, 1]

    ci = (1-alpha)*100

    # low_limit = (lower_limit).round(2).astype(float)
    # up_limit = (upper_limit).round(2).astype(float)

    #Ensure limits are within [0, 1] to prevent negative values as lower limit
    # lower_limit = max(0, lower_limit[0][0])
    #upper_limit = min(1, upper_limit[0][0]) #change limit

    # Show the prediction on the app
    st.write("## Predicting Traffic Volume...")
    pred_value = pred_value.as_type(int)
    # Display results using metric card
    st.metric(label = "Predicted Traffic Volume", value = f"${pred_value:.2f}")
    # st.write(f"**Prediction Interval** ({ci:.0f}%): [${low_limit}, ${up_limit}]")


# Additional tabs for DT model performance
st.subheader("Model Performance and Inference")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals.")
