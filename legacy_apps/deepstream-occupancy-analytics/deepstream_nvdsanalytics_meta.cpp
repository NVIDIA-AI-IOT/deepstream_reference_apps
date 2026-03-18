/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gst/gst.h>
#include <glib.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include "gstnvdsmeta.h"
#include "nvds_analytics_meta.h"
#include "analytics.h"

/* custom_parse_nvdsanalytics_meta_data 
 * and extract nvanalytics metadata */
	extern "C" void
analytics_custom_parse_nvdsanalytics_meta_data (NvDsMetaList *l_user, AnalyticsUserMeta *data)
{
	std::stringstream out_string;
	NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
	/* convert to  metadata */
	NvDsAnalyticsFrameMeta *meta =
		(NvDsAnalyticsFrameMeta *) user_meta->user_meta_data;
	/* Fill the data for entry, exit,occupancy */
	data->lcc_cnt_entry = 0;
	data->lcc_cnt_exit = 0;
	data->lccum_cnt = 0;
	data->lcc_cnt_entry = meta->objLCCumCnt["Entry"];
	data->lcc_cnt_exit = meta->objLCCumCnt["Exit"];

	if (meta->objLCCumCnt["Entry"]> meta->objLCCumCnt["Exit"])
		data->lccum_cnt = meta->objLCCumCnt["Entry"] - meta->objLCCumCnt["Exit"];
	// g_print("Enter: %d, Exit: %d\n", data->lcc_cnt_entry,data->lcc_cnt_exit);
}


