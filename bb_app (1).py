import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

def main():

	st.title("Binary Classification Web Application")
	st.sidebar.title("Binary Classification Web App")
	st.markdown("Ar your mushrooms poisonous?")
	st.sidebar.markdown("Are your mushrooms poisonous?")
	
	@st.cache(persist=True)
	#function decorator. Tells st to cache outputs to disc if arguments don't change
	#stops app from loading data every time the script is run
	def load_data():
		temp = pd.read_csv('/home/rhyme/Desktop/Projects/mushrooms.csv')
		lab_enc = LabelEncoder()
		for col in temp.columns:
			temp[col] = lab_enc.fit_transform(temp[col])
		return temp
		
	@st.cache(persist=True)
	def split(df):
		y = df.type
		X = df.drop(columns=['type'])
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
		return X_train, X_test, y_train, y_test
		
	def plot_metrics(metrics_list):
		if 'Confusion Matrix' in metrics_list:
			st.subheader("Confusion Matrix")
			plot_confusion_matrix(model, X_test, y_test, display_labels=class_names)
			st.pyplot()
			
		if 'ROC Curve' in metrics_list:
			st.subheader("ROC Curve")
			plot_roc_curve(model, X_test, y_test)
			st.pyplot()
			
		if 'Precision-Recall Curve' in metrics_list:
			st.subheader("Precision-Recall Curve")
			plot_precision_recall_curve(model, X_test, y_test)
			
	
	df = load_data()
	X_train, X_test, y_train, y_test = split(df)
	class_names = ['edible', 'poisonous']
	st.sidebar.subheader("Choose Classifier")
	classifer = st.sidebar.selectbox("Classifier, ("SVM", "LogisticRegression", "Random Forest"))
	
	
	##SVM CLASSIFIER##
    if classifier == "SVM":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel coefficient)", ("Scale", "Auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("SVM Results")
            model = SVM(C=C, kernel=kernel, gamma=gamma))
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            yhat = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, yhat, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, yhat, labels=class_names).round(2))
            plot_metrics(metrics)
			
			
	##LOGISTIC REGRESSION CLASSIFIER##
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 1000, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            yhat = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, yhat, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, yhat, labels=class_names).round(2))
            plot_metrics(metrics)


    ##RANDOM FOREST CLASSIFIER##
    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("Maximum depth of a tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Boostrap samples when building trees?", ("True", "False"), key='bootstrap')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            yhat = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, yhat, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, yhat, labels=class_names).round(2))
            plot_metrics(metrics)


    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Dataset for Classification")
        st.write(df)








if __name__ == '__main__':
    main()


