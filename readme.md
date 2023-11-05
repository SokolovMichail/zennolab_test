### Before building(if you haven't downloaded the whole container):

Download and unzip tasks archive into the data folder in repository root, so the directory structure must be as follows:

```
data
  tasks
    squirrels_head
    squirrels_tail
    ...
src
run.py
...
```
### Launch requirements:
Docker, optional - NVIDIA Container Toolkit in order to use GPU.

If using GPU - at least 8GB of video card memory.


### Building Container

```
docker build -f Dockerfile . --tag zennolab_test
```

### Running container
To run with GPU(Nvidia Container Toolkit required):
```
docker run --gpus=all  -v ./reports:/app/reports zennolab_test
```

To run with CPU only:
```
docker run -v ./reports:/app/reports zennolab_test
```
The container runs for about 2.4 hours on NVidia RTX3050.
After the run, the report can be found in "reports" folder

### Notes

A report from my run can be found in report.json file. 
It contains data about each task in particular and a total report on all tasks.
The total report(average distance, precision and run time on all tasks)
is listed under "total" key.