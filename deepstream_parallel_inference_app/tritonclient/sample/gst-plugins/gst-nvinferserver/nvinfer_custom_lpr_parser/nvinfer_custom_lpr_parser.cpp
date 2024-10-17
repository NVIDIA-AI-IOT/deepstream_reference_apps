/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <string>
#include <string.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <assert.h>
#include <locale>
#include <codecvt>
#include "nvdsinfer.h"
#include <fstream>

using namespace std;
using std::string;
using std::vector;

static bool dict_ready=false;
std::vector<string> dict_table;

extern "C"
{

bool NvDsInferParseCustomNVPlate(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                 NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                 std::vector<NvDsInferAttribute> &attrList, std::string &attrString)
{   
    int *outputStrBuffer = NULL;
    float *outputConfBuffer = NULL;
    NvDsInferAttribute LPR_attr;
   
    int seq_len = 0; 

    // Get list
    vector<int> str_idxes;
    int prev = 100;

    // For confidence
    double bank_softmax_max[16] = {0.0};
    unsigned int valid_bank_count = 0;
    bool do_softmax = false;
    ifstream fdict;

    setlocale(LC_CTYPE, "");

    if(!dict_ready) {
        fdict.open("dict.txt");
        if(!fdict.is_open())
        {
            cout << "open dictionary file failed." << endl;
	        return false;
        }
	    while(!fdict.eof()) {
	        string strLineAnsi;
	        if ( getline(fdict, strLineAnsi) ) {
	            dict_table.push_back(strLineAnsi);
	        }
        }
        dict_ready=true;
        fdict.close();
    }

    int layer_size = outputLayersInfo.size();

    LPR_attr.attributeConfidence = 1.0;

    seq_len = networkInfo.width/4;

    for( int li=0; li<layer_size; li++) {
        if(!outputLayersInfo[li].isInput) {
            if (outputLayersInfo[li].dataType == 0) {
                if (!outputConfBuffer)
                    outputConfBuffer = static_cast<float *>(outputLayersInfo[li].buffer);
            }
            else if (outputLayersInfo[li].dataType == 3) {
                if(!outputStrBuffer)
                    outputStrBuffer = static_cast<int *>(outputLayersInfo[li].buffer);
            }
        }
    }
 
    for(int seq_id = 0; seq_id < seq_len; seq_id++) {
       do_softmax = false;

       int curr_data = outputStrBuffer[seq_id];
       if (seq_id == 0) {
           prev = curr_data;
           str_idxes.push_back(curr_data);
           if ( curr_data != static_cast<int>(dict_table.size()) ) do_softmax = true;
       } else {
           if (curr_data != prev) {
               str_idxes.push_back(curr_data);
               if (static_cast<unsigned long>(curr_data) != dict_table.size()) do_softmax = true;
           }
           prev = curr_data;
       }

       // Do softmax
       if (do_softmax) {
           do_softmax = false;
           bank_softmax_max[valid_bank_count] = outputConfBuffer[curr_data];
           valid_bank_count++;
       }
    }

    attrString = "";
    for(unsigned int id = 0; id < str_idxes.size(); id++) {
        if (static_cast<unsigned int>(str_idxes[id]) != dict_table.size()) {
            attrString += dict_table[str_idxes[id]];
        }
    }

    //Ignore the short string, it may be wrong plate string
    if (valid_bank_count >=  3) {

        LPR_attr.attributeIndex = 0;
        LPR_attr.attributeValue = 1;
        LPR_attr.attributeLabel = strdup(attrString.c_str());
        for (unsigned int count = 0; count < valid_bank_count; count++) {
            LPR_attr.attributeConfidence *= bank_softmax_max[count];
        }
        attrList.push_back(LPR_attr);
    }

    return true;
}

}//end of extern "C"
