t, y_test)
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


