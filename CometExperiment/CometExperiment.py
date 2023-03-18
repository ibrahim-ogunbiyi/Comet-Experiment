import pandas as pd
import numpy as np
from comet_ml import Experiment
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class CometExperiment:
    
    def __init__(self, path, artifact_name, api_key, project_name, workspace):
        self.path = path
        self.artifact_name = artifact_name
        self.api_key = api_key
        self.project_name = project_name
        self.workspace = workspace
    
    def read_data(self):
        df = pd.read_csv(self.path)

        df.drop(columns=["cityCode"], inplace=True)

        X = df.drop(columns = ["price"])
        y = df["price"]

        result = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return result
    
    def compute_metrics(self, y_test, y_preds):
    
        # Create dictionary called metrics to store all metrics
        metrics = {}
        metrics["r2_score"] = r2_score(y_test, y_preds)
        metrics["mean_absolute_error"] = mean_absolute_error(y_test, y_preds)
        metrics["root_mean_squared_error"] = np.sqrt(mean_squared_error(y_test, y_preds))

        return metrics
    
    def run_experiment(self, model, model_name):
    
        #Instantiate the Experiment Object from comet
        experiment = Experiment(
            api_key=self.api_key,
            project_name= self.project_name,
            workspace= self.workspace)

        # Set the name of the experiment to be that of the model name passed
        experiment.set_name(model_name)
        experiment.add_tag(model_name)
        
        
        #Get the train and test data from read_data function
        X_train, X_test, y_train, y_test = self.read_data()

        with experiment.train(): 
            model.fit(X_train, y_train)

            # Logs the text message to comet
            experiment.log_text("This is the Evaluation Metrics for the Training Set")
            y_preds = model.predict(X_train)

            # Compute the mertrics
            metrics = self.compute_metrics(y_train, y_preds)

            #Log metrics to comet
            experiment.log_metrics(metrics)


        with experiment.validate():
            #Logs the text message to comet
            experiment.log_text("This is the Evaluation Metrics for the Validation Set")
            y_preds = model.predict(X_test)

            #Compute metrics
            metrics = self.compute_metrics(y_test, y_preds)

            #Log metrics to comet
            experiment.log_metrics(metrics)


if __name__ == "__main__":
    API_KEY = "9Sui2hzr0DDk4yWzikeUdssAB"
    project_name = "Comet Experiment"
    workspace = "ibrahim-ogunbiyi"
    
    path = "data/ParisHousing.csv"
    artifact_name = "ParisHousing"

   #Initialize the Object
    project = CometExperiment(path, artifact_name, API_KEY, project_name, workspace)
    
    #Pass in the model and it's name
    project.run_experiment(RandomForestRegressor(), "Random Forest Regressor")
    project.run_experiment(DecisionTreeRegressor(), "Decision Tree Regressor")