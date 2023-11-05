### Before building(if you haven't downloaded the whole container):

Download and unzip tasks archive into the data folder in repository root, so the directory structure must be as follows:

```
data
  tasks
    task1
    task2
    ...
src
run.py
...
```
### Building Container

```
docker build -f Dockerfile . --tag zennolab_test
```

### Running container
```
docker run zennolab_test -v ./reports:/app/reports --gpus=all
```
The container runs for about 2.4 hours on NVidia RTX3050.

### Notes

A report from my run can be found in report.json file. 
It contains data about each task in particular and a total report on all tasks.
The total report(average distance, precision and run time on all tasks)
is listed under "total" key.