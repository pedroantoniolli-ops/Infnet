from sklearn.metrics import classification_report

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return classification_report(y_test, preds)