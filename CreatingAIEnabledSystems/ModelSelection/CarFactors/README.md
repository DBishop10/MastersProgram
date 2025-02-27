# How To Run carfactors_service.py

## Steps to run flask locally
1. Ensure all packages are installed in requirements.txt
2. $: python carfactors_service.py
3. Open browser to http://127.0.0.1/stats

## Steps to run flask through docker
1. Pull docker image
2. $: docker run -it -p 8080:8080 dbishop7/705.603:Assignment3_1
3. Open browser to http://127.0.0.1/stats

## How to get information from flask
1. http://127.0.0.1/stats (Returns ML Algorithms Stats)
2. http://localhost:8786/infer?transmission=<CarTransmissionType>&color=<CarColor>&odometer=<number>&year=<number>&bodytype=<CarBodyType>&price=<number>
3. http://127.0.0.1/post (Outputs information to a file)