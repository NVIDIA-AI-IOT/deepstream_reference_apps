import sys
sys.path.append("../retail_iva")

import ast
import datetime
from configparser import ConfigParser
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ksql import KSQLAPI
import matplotlib.pyplot as plt
# import imageio
from tqdm import tqdm
import config


def query_parser(query):
    """
    Utility function to parse the query generator object returned by KSQLAPI
    This function returns a list of all rows of the table as retuened by the query
    """
    # The first element returned by the generator object is a list containing the column headers
    # Skip the first row
    next(query)

    # Iterate through the rest of the output
    output = [item[:-2] for item in query]

    # The last element returned by the generator object is always "]"
    # Remove the last element
    output = output[:-1]
    return output


def get_num_visitors_time_window(client, start_time=None, end_time=None):
    """
    This API can perform the following functions based on the inputs provided

    * If both start_time and end_time are provided, the API calculates the number
    of visitors that arrived in that time window
    * If only start_time is provided, the API calculates the number of visitors
    that arrived after that time
    * If only end_time is provided, the API calculates the number of visitors
    that arrived before that time
    """

    # Both start_time and end_time are not None
    if start_time is not None and end_time is not None:
        query = client.query(f'select object->id from {config.ksql_stream_name} where \
            rowtime > \'{start_time}\' and rowtime < \'{end_time}\'')
    # Only start_time is provided
    elif start_time is not None and end_time is None:
        query = client.query(f'select object->id from {config.ksql_stream_name} where \
            rowtime > \'{start_time}\'')    
    # Only end_time is provided
    elif start_time is None and end_time is not None:
        query = client.query(f'select object->id from {config.ksql_stream_name} where \
            rowtime < \'{end_time}\'')
    else:
        return -1

    # Parse the query
    output = query_parser(query)

    # Count the number of unique objects and return it
    # It is important to convert it to a set to avoid duplicates in counting
    items = [item for item in output]
    items = set(items)

    return len(items), items


def get_num_visitors_in_region(topleftx, bottomrightx, toplefty, bottomrighty, client):
    """
    This API is used to fetch the number of visitors in a rectangle defined by its
    top left and bottom right corners
    """
    topleftx, toplefty = int(topleftx), int(toplefty)
    bottomrightx, bottomrighty = int(bottomrightx), int(bottomrighty)
    img = cv2.imread("background.png")
    cv2.rectangle(img, (topleftx, toplefty), (bottomrightx, bottomrighty), (0, 0, 255), 3)
    cv2.imwrite("roi.png", img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    query = client.query(f'select object->id from {config.ksql_stream_name} where \
        object->bbox->topleftx > {topleftx} and \
        object->bbox->bottomrightx < {bottomrightx} and \
        object->bbox->toplefty > {toplefty} and \
        object->bbox->bottomrighty < {bottomrighty};')
    
    output = query_parser(query)

    items = [item for item in output]
    items = set(items)

    return len(items), items, Image.fromarray(img)


def get_visitor_path(person_id, client):
    """
    This API gets a visitor's path given the tracker ID.
    In order to estimate the visitor's location, we take the midpoint
    of the box detected.
    """
    query = client.query(f'select \
        object->bbox->topleftx, \
        object->bbox->bottomrightx, \
        object->bbox->toplefty, \
        object->bbox->bottomrighty \
        from {config.ksql_stream_name} \
        where object->id=\'{person_id}\';')

    output = query_parser(query)

    path = []

    frames = []

    # Iterate through all the rows and store the coordinates of the bbox
    for item in output:
        img = cv2.imread("background.png")
        item = ast.literal_eval(item)
        topleftx = item["row"]["columns"][0]
        bottomrightx = item["row"]["columns"][1]
        toplefty = item["row"]["columns"][2]
        bottomrighty = item["row"]["columns"][3]
        center_x, center_y = (bottomrightx + topleftx)//2, bottomrighty
        img[center_y-10:center_y+10, center_x-10:center_x+10, :] = 255
        img = cv2.resize(img, dsize=(1280, 720))
        frames.append(img)
        path.append([center_x, center_y])

    # with imageio.get_writer("test.gif", mode="I") as writer:
    #     for frame in tqdm(frames):
    #         writer.append_data(frame)

    img = cv2.imread("background.png")
    path = np.array(path)
    cv2.drawContours(img, [path], 0, (0,0,255), 5)
    img = cv2.resize(img, (1280, 720))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("testImg.png", img)

    return path, Image.fromarray(img)


def get_multiple_visitor_path(person_ids, client):
    """
    API to fetch the paths of multiple visitors.
    Example use case:
    To track a parent and child in the store and identify where the parent and child split up
    """
    img = cv2.imread("background.png")
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for id in person_ids:
        path, _ = get_visitor_path(id, client)
        cv2.drawContours(img, [path], 0, (0,0,255), 5)
    img = cv2.resize(img, (1280, 720))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("multiple.png", img)
    return


def get_store_heatmap(client, start_time=None, end_time=None):
    """
    Function to get a heatmap of the store.
    """
    img = plt.imread("background.png")
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(img, extent=(0, 2560, 0, 1440))
    query = client.query(f'select \
        object->bbox->topleftx, \
        object->bbox->bottomrightx, \
        object->bbox->toplefty, \
        object->bbox->bottomrighty \
        from {config.ksql_stream_name}')

    output = query_parser(query)

    coordinates = []

    for item in output:
        item = ast.literal_eval(item)
        topleftx = item["row"]["columns"][0]
        bottomrightx = item["row"]["columns"][1]
        toplefty = item["row"]["columns"][2]
        bottomrighty = item["row"]["columns"][3]
        coordinates.append([(bottomrightx + topleftx)//2, (bottomrighty + toplefty)//2])

    coordinates = np.array(coordinates)
    ax[1].hist2d(coordinates[:, 0], coordinates[:, 1], range=[[0, 2560], [0, 1440]])
    # plt.colorbar()
    fig.savefig("heatmap.png", dpi=100)
    return


def get_basket_pie(client):
    """
    Function to get a pie chart of the number of people holding baskets vs no basket
    """
    query = client.query(f'select object->obj_prop->hasBasket, \
                        object->id \
                        from {config.ksql_stream_name};')
    output = query_parser(query)

    counts = {}

    for item in output:
        item = ast.literal_eval(item)
        id = item["row"]["columns"][1]
        basket_class = item["row"]["columns"][0]
        counts.__setitem__(id, basket_class)

    results = {'hasBasket':0, 'noBasket':0}

    for value in counts.values():
        results[value] += 1

    return results


def get_time_plot(client):
    timeframe = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    query = client.query(f'select timestamp, object->id from {config.ksql_stream_name} \
                            where rowtime > \'{timeframe}\'')
    results = []

    output = query_parser(query)

    for item in output:
        item = ast.literal_eval(item)
        timestamp = pd.to_datetime(item["row"]["columns"][0]).round('1h').strftime('%Y-%m-%d %H:%M:%S')
        id = item["row"]["columns"][1]
        results.append({"TIMESTAMP":timestamp, "ID":id})

    df = pd.DataFrame(results, columns=["TIMESTAMP", "ID"])

    target_df = (
    df.groupby('TIMESTAMP')
    .agg(COUNT_PERSONID=('ID', 'nunique'))
    .reset_index()
    )
    return target_df.to_dict('records')


def get_aisle_counts(client):
    cfg = ConfigParser()
    cfg.read("store_config.ini")

    # Dictionary to store the number of visitors in each aisle
    results = []

    # Read all the sections from the config file
    for section in cfg.sections():
        topleftx = cfg.getint(section, "topleftx")
        bottomrightx = cfg.getint(section, "bottomrightx")
        toplefty = cfg.getint(section, "toplefty")
        bottomrighty = cfg.getint(section, "bottomrighty")
        num_people, _, _ = get_num_visitors_in_region(topleftx, bottomrightx,
                                                    toplefty, bottomrighty, client)
        results.append({"aisle": section, "count": num_people})

    return results



if __name__ == "__main__":
    client = KSQLAPI(config.ksql_server)
    # print(client.ksql("list streams;"))
    # print(get_num_visitors_in_region(50, 1250, 0, 1440, client))
    # get_visitor_path(7, client)
    # print(get_num_visitors_time_window(client, "2022-07-13", "2022-07-16"))
    # print(get_num_visitors_time_window(client, start_time="2022-07-13"))
    # print(get_num_visitors_time_window(client, end_time="2022-07-20"))
    # get_store_heatmap(client)
    # get_multiple_visitor_path([3, 17], client)
    # print(get_basket_pie(client))
    # print(get_time_plot(client))
    # print(get_aisle_counts(client))