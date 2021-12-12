import os
import json
import pytest
import requests
from azureml.core import Workspace, Webservice

deployment_name = os.getenv('DEPLOYMENT_NAME')

test_sample = json.dumps({
    'data': [{'A1Cresult': 'None',
              'admission_source_id': 'Emergency Room',
              'admission_type_id': 'Emergency',
              'age': '0-20',
              'change': 'Ch',
              'diabetesMed': 'Yes',
              'diag_1': 0,
              'discharge_disposition_id': 'Home',
              'gender': 'Female',
              'insulin': 'Up',
              'max_glu_serum': 'None',
              'num_lab_procedures': 59,
              'num_medications': 18,
              'num_procedures': 0,
              'number_diagnoses': 9,
              'number_emergency': 0,
              'number_inpatient': 0,
              'number_outpatient': 0,
              'race': 'Caucasian',
              'time_in_hospital': 3}]
})

ws = Workspace.from_config()


def test_deployed_model_service():
    service = Webservice(ws, deployment_name)
    assert service is not None

    key1, key2 = service.get_keys()
    uri = service.scoring_uri

    assert key1 is not None
    assert uri.startswith('http')

    headers = {'Content-Type': 'application/json',
               'Authorization': f'Bearer {key1}'}
    response = requests.post(uri, test_sample, headers=headers)
    assert response.status_code is 200
    assert abs(1 - sum(response.json()['predict_proba'][0])) < 0.01
