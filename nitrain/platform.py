import requests
import os
import tempfile
import json
from tqdm import tqdm
import ants

from .datasets.platform_dataset import PlatformDataset

api_url = 'https://api.ants.dev'

def list_platform_datasets():
    return _list_dataset_records()

def _launch_training_job_on_platform(job_name):
    """
    Send post request to api launching job
    """
    pass

def _upload_dataset_to_platform(dataset, name):
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
    _create_dataset_record(name, parameters=_config_for_platform_dataset(dataset))

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
        
        if response.status_code != 201:
            print(f'Could not upload file {x_filename}')
            
        tmpfile.close()

    # upload participants file (y)
    response = _upload_dataset_file(name,
                                    file=open(os.path.join(dataset.base_dir, 
                                                        dataset.y_config['file']),
                                                        'rb'),
                                    filename=dataset.y_config['file'])
    
    # if BIDS dataset -> write json file because BIDS layout doesnt work on platform ?
    return response

def _get_user_from_token(token=None):
    if token is None:
        token = os.environ['NITRAIN_API_TOKEN']
    ## create the dataset record
    response = requests.get(f'{api_url}/username/',
                headers = {'Authorization': f'Bearer {token}'})
    if response.status_code != 200:
        raise Exception('Could not infer username from api token. Is it valid?')
    return json.loads(response.content)
    
def _convert_to_platform_dataset(dataset, name, fuse=True):
    """
    Convert any nitrain dataset to a platform dataset that
    can be used in training a model on the platform.
    """
    params = _config_for_platform_dataset(dataset)
    return PlatformDataset(
        name = name,
        x = params['x_config'],
        y = params['y_config'],
        x_transforms = dataset.x_transforms,
        y_transforms = dataset.y_transforms,
        fuse = fuse,
        credentials = None
    )

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
    
    if response.status_code != 200:
        raise Exception('Could not access datasets.')
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

def _upload_file_to_platform(file, relative_path, token=None):
    if token is None:
        token = os.environ['NITRAIN_API_TOKEN']
    response = requests.post(f'{api_url}/files/', 
                files={'file': file},
                data={'relative_path': relative_path},
                headers = {'Authorization': f'Bearer {token}'})
    if response.status_code != 201:
        print(f'Error: {response.status_code}')
    return response 

def _config_for_platform_dataset(dataset):
    """Get the x + y config that is appropriate for a PlatformDataset"""
    if type(dataset).__name__ == 'BIDSDataset':
        # BIDS entities not
        parameters = {'x_config': {'filenames': dataset.x},
                      'y_config': dataset.y_config}
    elif type(dataset).__name__ == 'FolderDataset':
        parameters = {'x_config': dataset.x_config,
                      'y_config': dataset.y_config}
    return parameters
    
