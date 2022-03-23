# Data-driven bicycle network planning for demand and safety

This is the repository for my Lagrange Project on urban data science. 

This is the source code for the scientific paper Data-driven bicycle network planning for demand and safety by P. Folco, L. Gauvin, M. Tizzoni and M. Szell.

The notebooks are divided in 3 sections.

## Section 1: collect e-scooter position data from the Bird API.
* Run the notebook "1a_locations-for-requests.ipynb" to define the location of your queries.
* Run the notebook "1b_collect-escooter-data.ipynb" to collect the real-time Bird e-scooter positions. To access to the API we followed the istructions available here: https://github.com/ubahnverleih/WoBike/blob/master/Bird.md

## Section 2: pre-process the data.
This section is specific for Turin: if you want to test our model in another city, you should find the following data available for your city:
* Geolocated crashes involving bikes;
* Population density divided (at least) by neighborhood.

Before running these notebooks, please download these data sets and put it in the "data" folder:
* Crashes in Turin (2019), available here: http://aperto.comune.torino.it/dataset/elenco-incidenti-nell-anno-2019-nella-citta-di-torino
* Shapefiles of statistical areas of Turin, available here: http://aperto.comune.torino.it/en/dataset/popolazione-per-sesso-e-zona-statistica-2020
* Number of residents in Turin (2020) by statistical areas, available here: http://geoportale.comune.torino.it/geodati/zip/zonestat_popolazione_residente_geo.zip

Then:
* Run the notebook "2a_crashes-data.ipynb" to process the crashes data and obtain a data set with the geolocation of crashes involving at least 1 bike.
* Run the notebook "2b_generate-ODdata.ipynb" to generate Origin-Destination data set from the e-scooter positions collected in "1b_collect-escooter-data.ipynb".
* Run the notebook "2c_population-density.ipynb" to generate a data set with the population density of Turin (2020) divided by statistical area.

## Section 3: Run the model and analyze the results.
* Run the notebook "3a_prepare-networks.ipynb" to download from OpenStreetMap the car and bicycle networks. In the folder "data/turin" we provided the data of Turin, downloaded on 2021-07-12. This notebook is developed by M. Szell, you can find it also here: https://github.com/mszell/bikenwgrowth.
* Run the notebook "3b_pois.ipynb" to set the POIs. This notebook is based on the notebook of M. Szell "02_prepare_pois.ipynb" available here: https://github.com/mszell/bikenwgrowth.
* Run the notebook "3c_crash&trips-per-link.ipynb" to compute the number of crashes and trips passing through each link. 
* Run the notebook "3d_network-growth.ipynb" to compute the network growth and analyze the results. This notebook is based on the notebooks of M. Szell "03_poi_based_generation.ipynb" and "04_analyze_results.ipynb" available here: https://github.com/mszell/bikenwgrowth.
* Run the notebook "3e_plots.ipynb" to visualize the results, creating figures equivalent to the figures we showed in our paper.

The .py files "functions.py", "path.py", "setup.py" and "parameters.py" are based on the code of M. Szell available here: https://github.com/mszell/bikenwgrowth.

