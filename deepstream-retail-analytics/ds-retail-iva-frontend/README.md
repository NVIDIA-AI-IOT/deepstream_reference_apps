# DeepStream Retail IVA Frontend

This repo is a django-based frontend application for [DeepStream Retail IVA](https://gitlab-master.nvidia.com/admantri/ds-retail-iva). This application will be referred to as the parent application henceforth.

# Prerequisites

* Working installation of python and pip. Preferred to use conda

    * Installing anaconda/miniconda
    
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh
    chmod +x Miniconda3-py37_4.12.0-Linux-x86_64.sh
    ./Miniconda3-py37_4.12.0-Linux-x86_64.sh
    ```

    * You can create a conda environment using the [environment.yml](./environment.yml) file provided in this repo

    ```bash
    conda env create -f environment.yml
    ```

    * If you already have python installed and prefer to not use conda, you can setup the required packages by using the [requirements.txt](./requirements.txt) file

    ```bash
    pip install -r requirements.txt
    ```

* This project assumes that you have already setup confluent platform with kSQL database running as instructed in the [prerequisites](https://gitlab-master.nvidia.com/admantri/ds-retail-iva#prerequisites) section of the parent application.


# Running the dashboard

```bash
python manage.py runserver 0.0.0.0:8000
```

# URLs supported by this API

All the endpoints supported by this application can be found in the [urls.py](./retail_iva/urls.py) file of the project.

* http://localhost:8000/ - Homepage where a dashboard with all the plots is shown
* http://localhost:8000/num-visitors-region/ - API endpoint to get the number of visitors in a given region. 

    Inputs for this endpoint

    * topleftx - x coordinate of the top left corner
    * toplefty - y coordinate of the top left corner
    * bottomrightx - x coordinate of the bottom right corner
    * bottomrighty - y coordinate of the bottom right corner

    Example URL: http://localhost:8000/num-visitors-region?topleftx=50&bottomrightx=1400&toplefty=0&bottomrighty=1440

* http://localhost:8000/num-visitors-time/ - API endpoint to get the number of visitors in a given time window

    Inputs for this endpoint

    * start_time: Starting time of the time window
    * end_time: Ending time of the time window

    The API returns 0 if start_time > end_time

    This API supports three forms of inputs

    * If both start_time and end_time are provided, the API calculates the number
    of visitors that arrived in that time window
    * If only start_time is provided, the API calculates the number of visitors
    that arrived after that time
    * If only end_time is provided, the API calculates the number of visitors
    that arrived before that time
    * If none are provided, the API returns "-1"

    Example URL: http://localhost:8000/num-visitors-time?start_time=2022-07-29T03:44:41&end_time=2022-07-29T03:44:45

* http://localhost:8000/visitor-path - API endpoint to get the path of a visitor in the store. This API renders a path of the visitor on the background of the store.

    Inputs for this endpoint

    * person_id: This is the ID given by the NvDCF tracker

    Example URL: http://localhost:8000/visitor-path?person_id=7


# Config Files

This project comes with two config files

## [store_config.ini](./store_config.ini) 

This is a config file that carries information about coordinates of each aisle. This config file is used to generate a bar chart shown on the dashboard with information about how many visitors are present in each aisle. Adding new sections to this file will automatically result in a refreshed bar graph.

Each section in the bar graph requires 4 coordinates:

* topleftx - x coordinate of the top left corner
* toplefty - y coordinate of the top left corner
* bottomrightx - x coordinate of the bottom right corner
* bottomrighty - y coordinate of the bottom right corner


## [config.py](./config.py)

This is the configuration file for the Django project.

`TEST_MODE` - Set this to True to show a dashboard with fake data generated by running [random_message_generator.py](./random_message_generator.py)

`ksql_server` - URL for the kSQL server.

`kafka_server` - URL for the kafka server

`ksql_stream_name` and `kafka_topic` are set depending on `TEST_MODE`.


# Generating random data for testing

* [`random_message_generator.py`](./random_message_generator.py) can be used to generate fake kafka message data and deliver it to the kafka server.

* Although, kafka server creates the topic as messages are delivered, we have to explicitly create a kSQL stream. It is important to create the stream since the dashboard is entirely dependant on the kSQL stream. Follow the instructions given in the parent application's README to create a stream.

```SQL
CREATE STREAM TEST_STREAM (
messageid varchar,
mdsversion varchar,
timestamp varchar,
object struct<
    id varchar,
    speed int,
    direction int,
    orientation int,
    detection varchar,
    obj_prop struct<
        hasBasket varchar,
        confidence double>,
    bbox struct<
        topleftx int,
        toplefty int,
        bottomrightx int,
        bottomrighty int>>,
event_des struct<
    id varchar,
    type varchar>,
videopath varchar) WITH (
    KAFKA_TOPIC='detections', 
    VALUE_FORMAT='JSON', 
    TIMESTAMP='timestamp',
    TIMESTAMP_FORMAT='yyyy-MM-dd''T''HH:mm:ss.SSS''Z''');

```

* To check if the stream was created successfully run `list streams` from the ksql-cli

* Modify [this line](./random_message_generator.py#L69) to change the number of messages generated.


* To generate fake data and send messages execute

```bash
python random_message_generator.py
```

* To verify if the messages were stored in the stream, run the below command from ksql-cli

```SQL
select * from test_stream
```
