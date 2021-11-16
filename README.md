# DeepPCO

Contains the development work needed to establish visual odometry. 

We've included 2D encoded panoramic depth images, which are generated from KITTI odometry datasets.
Note that there is a difference between point cloud data in KITTI raw data and KITTI odometry datasets, and the model trained on odometry datasets will perform poorly on raw data.

## Directory structure

`src`: All the python files

`datasets`: The data (but only a little bit for testing).

`models`: Save the trained pytorch models here so you can load them later e.g. to demonstrate your results.

`runs`: Put your tensorboard(x) logs here.

### Source directory structure

`datasets`: Put your torch.datasets.Datasets here. One file per dataset!

`models`: Put your torch.nn.Modules here. You can but related modules in a single file.

`utils`: Utility methods here.

`visualization`: All code regarding visualization here. 


## Installing and exporting dependencies

If you add a pip or conda dependency update the `requirements.yml` using `conda env export > environment.yml --no-builds`. (see https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment)

To create an environment from this file use `conda env create -f environment.yml (see https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
