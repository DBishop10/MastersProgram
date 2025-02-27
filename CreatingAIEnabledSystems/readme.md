# Welcome to a Readme File

## How to run the Jupyter notebook
1. Download jhebeler/classtank
2. $: docker run --restart=unless-stopped --gpus all -it -p 8888:8888 -p 8787:8787 -p 8786:8786 -p 5000:5000 -v /workspace:/workspace jhebeler/classtank:705.603.jupyterlab
3. $: docker ps 
4. $: docker exec -it <Container ID> bash
5. $: jupyter lab --no-browser --ip=0.0.0.0 --allow-root
6. http://localhost:8888 