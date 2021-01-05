## To run:
1. Clone the repository
```
git clone https://gitlab.com/dellacortelab/hydrolase-design.git
```
2. Build the docker container
```
cd hydrolase-design && docker build -t hydrolase-design -f docker/Dockerfile .
```
3. Run an interactive docker container (replacing `/path/to/data` and `/path/to/logs` with the correct paths)
```
docker run -it --rm --name myname-hydrolase -v $(pwd):/code -v /path/to/data:/data -v /path/to/logs:/logs hydrolase-design
```
4. Inside the docker container, run 
```
cd code/src 
python3 experiment_driver.py
```
There is no command line interface set up yet, so to change options, you can tweak them in the definition of the `hotspot_experiment` function.