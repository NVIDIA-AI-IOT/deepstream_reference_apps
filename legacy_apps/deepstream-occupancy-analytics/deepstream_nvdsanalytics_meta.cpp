/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
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


