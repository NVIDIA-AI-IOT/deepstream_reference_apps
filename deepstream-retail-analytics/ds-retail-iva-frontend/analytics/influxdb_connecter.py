import sys
import cv2
import numpy as np
from influxdb import InfluxDBClient as idb
from PIL import Image

def get_num_visitors_time_window(start_time, end_time, client):
    pass

def get_num_visitors_in_region(topleftx, bottomrightx, toplefty, bottomrighty, client):
    client.switch_database("detections")
    result = client.query(f'SELECT * FROM detections where \
        topleftx > {topleftx} and bottomrightx < {bottomrightx} and \
        toplefty > {toplefty} and bottomrighty < {bottomrighty}')

    num_results = len(result.raw["series"][0]["values"])

    return num_results, result

def get_visitor_path(person_id, client):
    client.switch_database("detections")
    result = client.query(f'SELECT * FROM detections where id=\'{person_id}\'')

    path = []

    for val in result.raw["series"][0]["values"]:
        bottomrightx = val[1]
        bottomrighty = val[2]
        topleftx = val[5]
        toplefty = val[6]
        path.append([(bottomrightx + topleftx)//2, (bottomrighty + toplefty)//2])

    # img = cv2.imread("frame0.jpg")
    img = np.zeros([1080,1920,3],dtype=np.uint8)
    img.fill(0)
    path = np.array(path)
    cv2.drawContours(img, [path], 0, (255,255,255), 2)
    cv2.imwrite("testImg.png", img)

    return path, Image.fromarray(img)

