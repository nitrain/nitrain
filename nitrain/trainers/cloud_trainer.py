


class CloudTrainer:
    """
    Launch a nitrain training job in the cloud using your own
    Google Cloud or AWS account. It is recommended to use this
    trainer in combination with the GoogleCloudDataset class so
    that you can access your data directly during training without
    any data upload needed.
    """
    def __init__(self, model, task, name, credentials, resource='gpu-small'):
        pass