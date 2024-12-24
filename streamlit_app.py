import streamlit as st
import pandas as pd
import pickle
import Feature_extraction_new as feature
from urllib.parse import urlparse

with open("model_svm_rbf.pkl", "rb") as f:
        model_svm_rbf = pickle.load(f)


st.title("Phishing Website Detection")
st.subheader("Enter the URL to check if it is a phishing website or not")
url = st.text_input("Enter URL : ")
if st.button("Predict"):
    if urlparse(url).scheme not in ['http', 'https']:
         st.warning("Please enter a valid URL with scheme (http:// or https://)")
    else:
        st.write("Extracting Features...")
        st.write("This takes few seconds...")
        y_for_test = feature.get_data_set(url)
        val = y_for_test.fillna(0)
        pred = model_svm_rbf.predict(val)
        st.subheader("Prediction:")
        if pred[0] == 1:
            res = "Legitimate website"
        elif pred[0] == -1:
            res = "Phished website"
        st.success(res)

st.header("About")
st.subheader("What is Phishing?")
st.write("""Phishing is a type of social engineering where the attacker steals confidential data. 
         This includes login credentials, card details, or passwords. Phishing happens over electronic 
         means like emails or text messages that originate from unreliable sources. Phishing has 
         a list of negative effects on a business, including loss of money, loss of intellectual property, 
         damage to reputation, and disruption of operational activities.""")

st.subheader("Impact of Phishing")
st.write("""The consequences of falling victim to a phishing attack can be severe, leading to identity theft, 
         financial loss, or unauthorized access to personal accounts. Moreover, businesses can suffer 
         reputational damage and financial repercussions due to data breaches and compromised customer information.""")

st.subheader("How Machine Learning Helps in Fighting Phishing Attacks?")
st.write("""Fake websites and landing pages bear a close resemblance to their genuine counterparts. 
         They make authentic-looking promotions and have a cleverly engineered social media presence. 
         New tools of cyber scamming that possess overwhelming threats to the users and their security emerge every day. 
         Managing such advanced attack scenarios by human capacities alone becomes quite tedious and erroneous. 
         This is where Machine Learning providing adequate anti-phishing solutions becomes relevant. Its regular updating 
         and learning patterns make its countermeasures more effective and quicker.""")

st.write("""Machine learning is used to fight phishing websites by analyzing vast datasets to detect patterns and characteristics 
         of phishing sites. Algorithms can identify suspicious URLs distinguishing legitimate content from 
         potential threats. By using real-time threat intelligence and historical data, machine learning helps organizations stay 
         ahead of evolving phishing techniques. It also aids in user education and awareness by identifying trends in successful 
         attacks. Leveraging the power of machine learning enables quicker response times, better protection, and a more proactive 
         approach to combating phishing, reducing the risk of falling victim to these malicious schemes. However, human vigilance 
         remains essential in any comprehensive anti-phishing strategy.""")

st.markdown("<h5>For more information on this model, view the sidebar</h5>", unsafe_allow_html=True)

st.sidebar.header("About the Model")
st.sidebar.subheader("Dataset")
st.sidebar.write("The dataset used for training the model is the Phishing Websites Dataset from the Kaggle repository.")
st.sidebar.write("""The following features are extracted from the URLs provided\n
1.IP Address\n
2.Length of URL\n
3.Short URL service\n
4.‘@’ symbol\n
5.Double slash Redirection\n
6.Prefix Suffix\n
7.Subdomains\n
8.Response\n
9.Domain Registration Length\n
10.Favicon Icon\n
11.Port\n
12.HTTPS token\n
13.Request URL\n
14.Google Index\n
15.Fraud IP\n
16.Links in tags\n
17.Number of link\n
And many more...""")

st.sidebar.subheader("Model")
st.sidebar.write("""The model used for prediction is the Support Vector Machine (SVM) with Radial Basis Function (RBF) kernel.""")
st.sidebar.write("""Support Vector Machines (SVMs) are a type of machine learning algorithm that can be used for classification tasks.
                  They work by finding the best hyperplane that separates different classes of data points. \nSVMs have several 
                 advantages over other machine learning algorithms, including their ability to handle high-dimensional data and 
                 their ability to find non-linear decision boundaries.""")
st.sidebar.write("""Radial Basis Function (RBF) kernel is a popular kernel function used in various kernelized learning algorithms.""")
st.sidebar.write("This model gave an accuracy of 94.02% on the test dataset.")