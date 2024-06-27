# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import base64
import json
import datetime
from io import BytesIO
from django.http import HttpResponse
from django.shortcuts import render
from .ksqldb_connecter import get_num_visitors_in_region, get_visitor_path
from .ksqldb_connecter import get_num_visitors_time_window
from .ksqldb_connecter import get_basket_pie
from .ksqldb_connecter import get_time_plot, get_aisle_counts
from ksql import KSQLAPI

import config
client = KSQLAPI(config.ksql_server)


def img_to_base64_str(img):
    """
    Helper function to convert PIL image to base64 for rendering
    on the webpage
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    return img_str


def home(request):
    """
    Function to return the dashboard view.
    This function calls multiple APIs that return data
    * to generate a pie showing people w & w/o baskets
    * to generate a plot showing how many people were in the store vs time of the day
    * to generate a bar showing how many people were in each aisle
    """
    now = datetime.datetime.now()
    delta = (now - datetime.timedelta(hours=10)).strftime("%Y-%m-%dT%H:%M:%S")
    now = now.strftime("%Y-%m-%dT%H:%M:%S")
    basket_counts = get_basket_pie(client)
    time_bar = get_time_plot(client)
    aisle_counts = get_aisle_counts(client)
    customer_count, _ = get_num_visitors_time_window(client, 
                                                  start_time=delta,
                                                  end_time=now)
    return render(request, "dashboard.html", context={"basket_counts":json.dumps(basket_counts),
                                                      "time_bar": json.dumps(time_bar),
                                                      "aisle_counts": aisle_counts,
                                                      "customer_count": customer_count})


def num_visitors_in_region_view(request):
    if request.method == "GET":
        topleftx = request.GET.get("topleftx")
        toplefty = request.GET.get("toplefty")
        bottomrightx = request.GET.get("bottomrightx")
        bottomrighty = request.GET.get("bottomrighty")
        coordinates = [topleftx, toplefty, bottomrightx, bottomrighty]
        if None in coordinates:
            return HttpResponse("Unsupported request format. Use GET request with \
                topleftx, toplefty, bottomrightx, bottomrighty params")
            
        num_visitors, visitor_ids, roi = get_num_visitors_in_region(topleftx, 
                                                    bottomrightx, 
                                                    toplefty, 
                                                    bottomrighty, client)
        
        print(visitor_ids)

        roi = img_to_base64_str(roi)

        # return HttpResponse(num_visitors)
        return render(request, "region_view.html", context={"num_visitors":num_visitors,
                                                            "visitors":visitor_ids,
                                                            "img":roi})
    
    return HttpResponse("Unsupported request format. Use GET request with \
        topleftx, toplefty, bottomrightx, bottomrighty params")


def visitor_path_view(request):
    """
    Function to generate a visitor's path given the tracker ID
    """
    if request.method == "GET":
        person_id = request.GET.get("person_id")
        path, img = get_visitor_path(person_id, client)
        img = img_to_base64_str(img)
        return render(request, "path_view.html", context={"img":img})

    return HttpResponse("Invalid request format")


def num_visitors_in_time_window_view(request):
    if request.method == "GET":
        start_time = request.GET.get("start_time")
        end_time = request.GET.get("end_time")
        num_visitors, visitor_ids = get_num_visitors_time_window(client, start_time, end_time)
        # return HttpResponse(num_visitors)
        return render(request, "time_view.html", context={"num_visitors":num_visitors,
                                                        "visitors":visitor_ids})

    return HttpResponse("Unsupported request format. Use GET request with at least \
        start_time and end_time")
