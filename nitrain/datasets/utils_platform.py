import requests
import os
import tempfile
import json
from tqdm import tqdm
import ants

api_url = 'http://127.0.0.1:8000' #'https://api.ants.dev'

def upload_dataset_to_platform(dataset, name):
    """
    Upload a nitrain dataset to the platform.
    
    Arguments
    ---------
    dataset : a nitrain dataset
        The dataset to be uploaded
        
    name : string
        What to call the dataset on the platform
    """
    # create dataset record
    _create_dataset_record(name, parameters=_dataset_parameters(dataset))

    # upload images (x)
    for i in tqdm(range(len(dataset))):
        # get transformed data
        x_filename = dataset.x[i]
        x, y = dataset[i]
        
        # save to tempfile
        tmpfile = tempfile.NamedTemporaryFile(suffix='.nii.gz')
        ants.image_write(x, tmpfile.name)
        
        # upload to server
        response = _upload_dataset_file(name,
                                        file=tmpfile,
                                        filename=x_filename.replace(dataset.base_dir, ''))
        tmpfile.close()

    # upload participants file (y)
    response = _upload_dataset_file(name,
                                    file=open(os.path.join(dataset.base_dir, 
                                                        dataset.y_config['file']),
                                                        'rb'),
                                    filename=dataset.y_config['file'])
    
    # if BIDS dataset -> write json file because BIDS layout doesnt work on platform ?
    return response

def _create_dataset_record(name, parameters, token=None):
    if token is None:
        token = os.environ['NITRAIN_API_TOKEN']
    ## create the dataset record
    response = requests.post(f'{api_url}/datasets/', 
                json={
                    'name': name,
                    'source': 'LocalStorage',
                    'parameters': parameters,
                    'status': 'Connected',
                    'cached': False
                },
                headers = {'Authorization': f'Bearer {token}'})
    return response

def _list_dataset_records(token=None):
    if token is None:
        token = os.environ['NITRAIN_API_TOKEN']
    
    response = requests.get(f'{api_url}/datasets/', 
                headers = {'Authorization': f'Bearer {token}'})
    return json.loads(response.content)

def _get_dataset_record(name, token=None):
    if token is None:
        token = os.environ['NITRAIN_API_TOKEN']
    response = requests.get(f'{api_url}/datasets/{name}/', 
                headers = {'Authorization': f'Bearer {token}'})
    return response

def _delete_dataset_record(name, token=None):
    if token is None:
        token = os.environ['NITRAIN_API_TOKEN']
    response = requests.delete(f'{api_url}/datasets/{name}/', 
                headers = {'Authorization': f'Bearer {token}'})
    return response

def _upload_dataset_file(name, file, filename, token=None):
    if token is None:
        token = os.environ['NITRAIN_API_TOKEN']
    response = requests.post(f'{api_url}/datasets/{name}/files/', 
                files={'file': file},
                data={'relative_path': filename},
                headers = {'Authorization': f'Bearer {token}'})
    return response

def _dataset_parameters(dataset):
    if type(dataset).__name__ == 'BIDSDataset':
        parameters = {'x_config': dataset.x_config,
                      'y_config': dataset.y_config}
    return parameters
    
