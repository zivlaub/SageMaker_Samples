# This is a sample Python program for deploying a trained TensorFow model to a SageMaker MME endpoint.
# Inference is done with a file in S3 instead of http payload for the SageMaker Endpoint.
###############################################################################################

from sagemaker.local import LocalSession
from sagemaker.tensorflow import TensorFlowModel
from sagemaker.multidatamodel import MultiDataModel
from sagemaker import image_uris
import sagemaker
from datetime import datetime
import time
from time import gmtime
from time import strftime
import json
from sagemaker import get_execution_role
import logging
import boto3

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'


def main():
    role = get_execution_role()
    sagemaker_session = sagemaker.Session()
    model_url = f's3://d3zsvmfm322edj93/MME/'# Put models tar file on S3 and change url here
    container_image = image_uris.retrieve(framework='tensorflow',region='us-east-1',version='2.3.0',image_scope='inference',instance_type='ml.m5.xlarge')

    current_time = f'-{strftime("%Y-%m-%d-%H-%M-%S", gmtime())}'
    sm_client = boto3.client(service_name="sagemaker")
    model_name = "DEMO-MultiModelModel" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    
    print("Model name: " + model_name)
    print("Model data Url: " + model_url)
    print("Container image: " + container_image)
    
    container = {"Image": container_image, "ModelDataUrl": model_url, "Mode": "MultiModel", 
        "Environment": {
        "SAGEMAKER_CONTAINER_LOG_LEVEL": "10",
        "SAGEMAKER_REGION": "us-east-1"
     }
    }
    
    create_model_response = sm_client.create_model(
        ModelName=model_name, ExecutionRoleArn=role, Containers=[container]
    )
    print("Model Arn: " + create_model_response["ModelArn"])
    
    endpoint_config_name = "DEMO-MultiModelEndpointConfig-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    print("Endpoint config name: " + endpoint_config_name)
    
    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "InstanceType": "ml.m5.xlarge",
                "InitialInstanceCount": 1,
                "InitialVariantWeight": 1,
                "ModelName": model_name,
                "VariantName": "AllTraffic",
            }
        ],
    )
    
    print("Endpoint config Arn: " + create_endpoint_config_response["EndpointConfigArn"])
    
    endpoint_name = "DEMO-MultiModelEndpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    print("Endpoint name: " + endpoint_name)
    
    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )
    print("Endpoint Arn: " + create_endpoint_response["EndpointArn"])
    
    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = resp["EndpointStatus"]
    print("Endpoint Status: " + status)
    
    print("Waiting for {} endpoint to be in service...".format(endpoint_name))
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)

    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = resp["EndpointStatus"]
    print("Endpoint Status: " + status)
    
    
    runtime_sm_client = boto3.client(service_name="sagemaker-runtime")
    with open("instances.json", 'r') as f:
        payload = f.read().strip()
    
    print("Invokding model_A...")
    response = runtime_sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        TargetModel="model_A.tar.gz",  # this is the rest of the S3 path where the model artifacts are located
        Body=payload,
    )
    
    response_body = response['Body']
    print(response_body.read())
    
    print("Invokding model_B...")
    response = runtime_sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        TargetModel="model_B.tar.gz",  # this is the rest of the S3 path where the model artifacts are located
        Body=payload,
    )
    response_body = response['Body']
    print(response_body.read())

    print("cleaning up...")
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    sm_client.delete_model(ModelName=model_name)
    print("Done")
    

if __name__ == "__main__":
    main()