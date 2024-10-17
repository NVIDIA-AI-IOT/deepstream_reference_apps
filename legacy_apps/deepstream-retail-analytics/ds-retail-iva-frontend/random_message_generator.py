# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from random import randrange, choice
import datetime
import json
from uuid import uuid4
from confluent_kafka import Producer
from config import kafka_topic, kafka_server
from tqdm import tqdm

def generate_random_timestamp(start):
    ts = (start - datetime.timedelta(hours=randrange(12),
                                     minutes=randrange(60), 
                                     seconds=randrange(60))).strftime("%Y-%m-%dT%H:%M:%S.%f")
    return ts[:-3] + 'Z'


def generate_random_bbox(max_width=2560, max_height=1440):
    bottomrightx = randrange(max_width)
    topleftx = randrange(bottomrightx)
    bottomrighty = randrange(max_height)
    toplefty = randrange(bottomrighty)
    return {"topleftx": topleftx,
            "toplefty": toplefty,
            "bottomrightx": bottomrightx,
            "bottomrighty": bottomrighty}


def generate_random_obj_id(max_id=10):
    return str(randrange(max_id))


def generate_random_basket_des(classes=["hasBasket", "noBasket"]):
    return choice(classes)


def generate_kafka_message():
    message = {}
    # Set messageid and mdsversion
    message.__setitem__("messageid", str(uuid4()))
    message.__setitem__("mdsversion", "1.0")

    # Set timestamp
    message.__setitem__("timestamp", generate_random_timestamp(datetime.datetime.now()))

    # Create and set object
    object = {"id":generate_random_obj_id(),
              "speed": 0,
              "direction": 0,
              "orientation": 0,
              "detection": "person",
              "obj_prop": {"hasBasket": generate_random_basket_des(),
                           "confidence": 0.99},
              "bbox": generate_random_bbox()}
    message.__setitem__("object", object)

    # Set event_des and videopath
    message.__setitem__("event_des", {"id": str(uuid4()), "type": "entry"})
    message.__setitem__("videopath", "")
    return message


if __name__ == "__main__":
    if kafka_topic == "detections":
        print("WARNING: Writing messages to main topic. Set TEST_MODE to True in config.py to write messages to dummy topic")

    # Define a producer for the topic
    prd = Producer({"bootstrap.servers":kafka_server})

    # Write messages to the topic
    for _ in tqdm(range(1)):
        msg = generate_kafka_message()
        print(msg)
        prd.produce(topic=kafka_topic, value=json.dumps(msg, indent=4))

    # Block until messages are sent
    prd.poll(10000)
    prd.flush()
