# This scripts dynamically loads two models into memory using tf.keras.models.load_model() and use them for inference 
# instead of sending the requests to TFS REST API
# 
###############################################################################################

import json
import os
import requests
import tensorflow as tf
import time
import numpy as np



def handler(data, context):
    """Handle request.
    Args:
        data (obj): the request data
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, (optional) response content type
    """
    print("handler method start")
    
    print('model_name {}'.format(context.model_name))
    ensure_load_model(context.model_name)

    input_json = _process_input(data, context)
    input_array = np.array(input_json['instances'])
    
    #DO NOT CALL TFS
    #response = requests.post(context.rest_uri, data=processed_input)
    ###
    global wrapper_model_0
    global wrapper_model_1
    #Predict against first mode
    response_1 = wrapper_model_0(input_array)
    
     #Predict against second mode
    response_2 = wrapper_model_1(input_array)
    
    response = {}
    
    response['response_1'] = response_1.numpy().tolist()
    response['response_2'] = response_2.numpy().tolist()
  
    res = json.dumps(response)
 
    
    return _process_output(res, context)
    

def _process_input(data, context):
    if context.request_content_type == 'application/json':
        # pass through json (assumes it's correctly formed)
        d = data.read().decode('utf-8')
        print('input data: {}'.format(d))
        input_json = json.loads(d)
        
        return input_json

    if context.request_content_type == 'text/csv':
        # very simple csv handler
        return json.dumps({
            'instances': [float(x) for x in data.read().decode('utf-8').split(',')]
        })

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        context.request_content_type or "unknown"))


def _process_output(data, context):
    response_content_type = context.accept_header
    prediction = data
    return prediction, response_content_type

def ensure_load_model(model_name):
    global models_loaded
    global wrapper_model_0
    global wrapper_model_1
    if models_loaded is None or models_loaded == False:
        print('Loading models...')
        model_path_0 = '/opt/ml/models/{}/model/0/'.format(model_name)
        wrapper_model_0 = tf.keras.models.load_model(model_path_0)
        print('model {} was successfully loaded'.format(model_path_0))
        model_path_1 = '/opt/ml/models/{}/model/1/'.format(model_name)
        wrapper_model_1 = tf.keras.models.load_model(model_path_1)
        models_loaded = True
        print('model {} was successfully loaded'.format(model_path_1))
        print('Models were successfuly loaded')


def init_inference():
    print('init_inference started')
    global wrapper_model_0
    global wrapper_model_1
    global models_loaded
    models_loaded = False
    print('init_inference done')

init_inference()
