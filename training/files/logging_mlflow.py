# This file defines the classes used for logging metrics, params, artifacts and models to MLFlow.

import mlflow
import logging
import importlib
import os

class LogMLflow:
    def __init__(self, proj_tags: dict, mlflow_uri: str, mlflow_experiment: str):
        """
        This class is used for high level abstraction of MLFlow.

        proj_tags: dict containing the project flags
        mlflow_uri: URL for the mlflow server instance
        mlflow_experiment: experiment name to send the logs
        """

        self.tags = proj_tags
        self.mlflow_uri = mlflow_uri
        self.mlflow_experiment = mlflow_experiment
        
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.mlflow_experiment)


    def send_logs(self, model_params: dict=None, training_metrics: dict=None, training_artifacts: dict=None, step: int=None):
        """
        Method for sending logs to MLFlow

        model_params: dict containing the model parameters
        training_metrics: dict containing the training metrics
        training_artifacts: dict containing the directories and the folder where to log the artifacts.
        step: step for metric logging
        """

        self.artifact_uri = mlflow.get_artifact_uri()
        self.mlflow_run_id = mlflow.get_run(mlflow.active_run().info.run_id)
        self.mlflow_run_data = mlflow.get_run(mlflow.active_run().info.run_id).data

        mlflow.log_params(params=self.tags)

        if model_params:
            mlflow.log_params(params=model_params)

        if training_metrics:
            mlflow.log_metrics(metrics=training_metrics, step=step)

        if training_artifacts:
            for artifact_server_folder_name in training_artifacts.keys():
                if type(training_artifacts[artifact_server_folder_name]) == list:
                    for artifact_local_path in training_artifacts[artifact_server_folder_name]:
                        if artifact_server_folder_name == 'root':
                            mlflow.log_artifact(artifact_local_path)
                        else:
                            mlflow.log_artifact(artifact_local_path, artifact_path=artifact_server_folder_name)
                else:
                    if artifact_server_folder_name == 'root':
                        mlflow.log_artifacts(training_artifacts[artifact_server_folder_name])
                    else:
                        mlflow.log_artifacts(training_artifacts[artifact_server_folder_name], artifact_path=artifact_server_folder_name)

        logging.info(f'Sucesso ao enviar métricas para {self.mlflow_uri} no experimento "{self.mlflow_experiment}"')

        return self


    def save_model(self, model, flavor: str, registered_model_name: str=None, change_to_production: bool=True):
        """
        Method for saving a model as an artifact.

        model: model object
        flavor: mlflow model flavor (keras, sklearn etc.)
        registered_model_name: if provided, register the model with the respective string
        """

        assert self.mlflow_run_id, 'You need to send at least 1 log first. Use the send_logs() method for this'

        m = importlib.import_module(f'mlflow.{flavor}')

        m.log_model(model, 'model', registered_model_name=registered_model_name)

        if change_to_production:
            self._change_current_version_to_production(registered_model_name)

        self.end_current_run()

    
    def _change_current_version_to_production(self, registered_model_name):
        """
        Method for change the last version for production.

        registered_model_name: name of the model to change
        """

        r = Repositorio(self.mlflow_uri)

        model_version = r.get_production_version(registered_model_name)

        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(self.mlflow_uri)

        client = MlflowClient()

        if model_version:
            current_version = int(model_version['version'])
            nb_versions = len(client.search_model_versions(f"name='{registered_model_name}'"))

            client.transition_model_version_stage(
                name=registered_model_name,
                version=current_version,
                stage='Archived'
            )

            client.transition_model_version_stage(
                name=registered_model_name,
                version=nb_versions,
                stage='Production'
            )
        else:
            client.transition_model_version_stage(
                name=registered_model_name,
                version=1,
                stage='Production'
            )

    def end_current_run(self):
        """
        Ends the current run
        """

        mlflow.end_run()

    
    def __repr__(self):
        return f'Logger MLFlow. Experiment: {self.mlflow_experiment}'



class Repositorio:
    def __init__(self, mlflow_uri: str):
        """
        This class is used for access the model repository
        """

        self.mlflow_uri = mlflow_uri

    
    def get_model(self, model_name: str, flavor: str, stage: str='Production'):
        """
        Select the most recent version for model and stage.

        model_name: model's name
        flavor: mlflow model flavor (keras, sklearn etc.)
        stage: model's stage to select
        """

        m = importlib.import_module(f'mlflow.{flavor}')

        mlflow.set_tracking_uri(self.mlflow_uri)

        model = m.load_model(
            model_uri=f'models:/{model_name}/{stage}'
        )

        return model

    
    def get_production_version(self, model_name: str):
        """
        Returns a dict with information about the model.

        return: {
            'name': model's name,
            'description': model's description,
            'run_id': run id that saved the model,
            'status': model's status,
            'version': model's version
        }
        """

        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(self.mlflow_uri)

        client = MlflowClient()

        versions = client.search_model_versions(f"name='{model_name}'")

        production_version = [prod_ver for prod_ver in versions if prod_ver.current_stage == 'Production']

        if production_version:
            p = production_version[0]
            production_version_info = dict(
                name=p.name,
                description=p.description,
                run_id=p.run_id,
                status=p.status,
                version=p.version
            )

            return production_version_info
        else:
            logging.warning(f'There is no production version for the model {model_name}')
            return None


