import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import load, dump
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

#factory design pattern for models used
class RegressionModels:
    @staticmethod #so we don't have to create more classes with their method
    def get_model(choose):
        if choose == "Linear":
            return LinearRegression()
        elif choose == "Ridge":
            return Ridge()
        elif choose == "Decision Tree":
            return DecisionTreeRegressor()
        elif choose == "Random Forest":
            return RandomForestRegressor()
        elif choose == "Gradient Boosting":
            return GradientBoostingRegressor()
        else:
            raise ValueError(f"Model type '{choose}' is not found!")

def load_data():
    print("Loading data...")
    df = pd.read_excel("train.xlsx")
    return df

def preprocess_data(df):
    print("Preprocessing data...")
        
    print("Original input data shape: ", df.shape)   
    # drop the null rows
    df = df.dropna(axis = 0, how = 'any')
    print("Processed input data shape: ", df.shape)
    
    #check for missing values
    if df.isnull().any().any():
        raise ValueError("Data cannot have missing values. Try change or use another data.")
    
    x_feature = df.drop(["User_ID", "Product_ID", "Stay_In_Current_City_Years", "Product_Category_1","Product_Category_2","Product_Category_3","Purchase"], axis=1)
    y_label = df["Purchase"]
    
    #Changing categorical values into numerical values
    convert_data = LabelEncoder()
    x_feature['Gender'] = convert_data.fit_transform(x_feature['Gender'])
    x_feature['City_Category'] = convert_data.fit_transform(x_feature['City_Category'])
    x_feature['Age'] = convert_data.fit_transform(x_feature['Age'])
    
    # finding out the unique values in a column
    # print(x_feature['Age'].unique())

    x_sc = MinMaxScaler()
    y_sc = MinMaxScaler()
    
    x = x_sc.fit_transform(x_feature)
    y_reshape= y_label.values.reshape(-1,1)
    y = y_sc.fit_transform(y_reshape)
    
    # printing out Numpy array to check if the transformed worked correctly
    # print(x[:5])
    return x, y, y_sc

def split_data(x, y):
    print("Splitting data...")
    x_train, x_test, y_train, y_test = train_test_split(x, y.ravel(), test_size=0.2, random_state=42)
    print("SUCCESS!")
    return x_train, x_test, y_train, y_test

def train_models(x_train, y_train):
    class_models = RegressionModels()
    model_names = ["Linear", "Ridge", "Decision Tree", "Random Forest", "Gradient Boosting"]
    
    models = {}
    for name in model_names:
        print(f"Training {name} Regression model...")
        model = class_models.get_model(name)
        model.fit(x_train, y_train)
        models[name] = model
        print(f"Successfully trained {name} Regression model!")
    
    return models

def evaluate_models(x_test, y_test, models):
    rmse_values = {}
    
    for name, model in models.items():
        preds = model.predict(x_test)
        rmse_values[name] = mean_squared_error(y_test, preds, squared=False)
        
    return rmse_values

def plot_model_performance(rmse_values):
    models = list(rmse_values.keys())
    rmse = list(rmse_values.values())
    plt.figure(figsize=(10,7))
    bars = plt.bar(models, rmse, color=['blue', 'green', 'red', 'purple', 'orange'])

    # Add RMSE values on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.00001, round(yval, 5), ha='center', va='bottom', fontsize=10)

    plt.xlabel('Models')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Model RMSE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def save_best_model(models, rmse_values):
    best_model_name = min(rmse_values, key=rmse_values.get)
    print(f"Best model is {best_model_name} Regression.\nProceeding to save trained model...")
    best_model = models[best_model_name]
    filename = "regressmodel_newtest.joblib"
    dump(best_model, filename)
    
    loaded_model = load(filename)
    print(f"Model successfully saved as {filename}.")
    return loaded_model

def predict_new_data(loaded_model, get_data, y_sc):
    input_data = np.array(get_data).reshape(1, -1)
    predict_value = loaded_model.predict(input_data)
    predict_reshape = predict_value.reshape(-1,1)
    real_value_np = y_sc.inverse_transform(predict_reshape)
    
    # Extract a single element from the NumPy array
    real_value = real_value_np.item()
    print(f"Predicted Purchase Amount: ${real_value:.3f}")
    

def numeric_validation(prompt):
    while True:
        try:
            value = float(input(prompt))            
            return value
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

if __name__ == "__main__":
    try:
        print("===========Welcome==========")
        data = load_data()
        x, y, y_sc = preprocess_data(data)
        x_train, x_test, y_train, y_test = split_data(x, y)
        print("=============LOAD MODELS============")
        models = train_models(x_train, y_train)
        rmse_values = evaluate_models(x_test, y_test, models)
        
        print("=============================")
        print("1. Models RMSE Comparison\n2. Try testing the best model")
        user = input("Please choose a number: ")
        
        if user in ("1"):
            plot_model_performance(rmse_values)
        elif user in ("2"):
            loaded_model = save_best_model(models, rmse_values)
            
            #User inputs
            while True:
                gender = input("Enter gender (0 for female, 1 for male): ")
                if gender in ("0", "1"):
                    gender = int(gender)
                    break
                else:
                    print("Invalid input. Please enter 0 or 1")
                
            job = numeric_validation("Enter years of occupation: ")
            city = numeric_validation("Enter the city category (A = 0, B = 1, C = 2): ")
            marital_status = numeric_validation("Enter the marital status (0 for not married, 1 for married): ")
            average_age = numeric_validation("(0 = 0-17, 1 = 18-25, 2 = 26-35, 3 = 36-45, 4 = 46-50, 5 = 51-55, 6 = 55+)\nEnter your age based on the range above: ")
            if average_age < 0 or average_age > 6:  
                raise ValueError("Please select available age range number.")

            get_data = [gender, job, city, marital_status, average_age]
            # get_data = np.array(get_datalist)
            predict_new_data(loaded_model, get_data, y_sc)
        else:
            quit       
    except ValueError as ve:
        print(f"Error: {ve}")
