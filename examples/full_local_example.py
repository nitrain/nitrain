# Full example fit locally

import nitrain as nit


# create dataset from folder of images + participants file
dataset = nit.FolderDataset(base_dir='ds004711',
                            x={'pattern': 'sub-*/anat/*_T1w.nii.gz'},
                            y={'file': 'participants.tsv',
                                    'column': 'age'},
                            x_transforms=[nit.ResizeImage((64,64,64)),
                                          nit.NormalizeIntensity(0,1)])

# create loader with random transforms
loader = nit.DatasetLoader(dataset,
                           batch_size=32,
                           shuffle=True,
                           x_transforms=[nit.RandomSlice(axis=2),
                                         nit.RandomNoise(sd=0.2)])

# create model from architecture
architecture_fn = nit.fetch_architecture('alexnet', task='continuous_prediction')
model = architecture_fn(layers=[128, 64, 32, 10], n_outcomes=1)

# create trainer and fit model
trainer = nit.ModelTrainer(model,
                           loss='mse',
                           optimizer='adam',
                           lr=1e-3,
                           callbacks=[nit.EarlyStopping(),
                                      nit.ModelCheckpoints(freq=25)])
trainer.fit(loader, epochs=100)

# upload trained model to platform
nit.register_model(trainer.model, 'nick/t1-brainage-model')