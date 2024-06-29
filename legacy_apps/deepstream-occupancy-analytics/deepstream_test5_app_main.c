/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <sys/stat.h>
#include <sys/time.h>
#include <sys/timeb.h>
#include <sys/types.h>
#include <sys/inotify.h>

#include <time.h>
#include <unistd.h>
#include <errno.h>

#include "deepstream_app.h"
#include "deepstream_config_file_parser.h"
#include "nvds_version.h"

#include <termios.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"

#include "deepstream_test5_app.h"

/*Analytics header*/
#include "analytics.h"

#define MAX_DISPLAY_LEN (64)
#define MAX_TIME_STAMP_LEN (64)
#define STREAMMUX_BUFFER_POOL_SIZE (16)

/** @{
 * Macro's below and corresponding code-blocks are used to demonstrate
 * nvmsgconv + Broker Metadata manipulation possibility
 */

/**
 * IMPORTANT Note 1:
 * The code within the check for model_used == APP_CONFIG_ANALYTICS_RESNET_PGIE_3SGIE_TYPE_COLOR_MAKE
 * is applicable as sample demo code for
 * configs that use resnet PGIE model
 * with class ID's: {0, 1, 2, 3} for {CAR, BICYCLE, PERSON, ROADSIGN}
 * followed by optional Tracker + 3 X SGIEs (Vehicle-Type,Color,Make)
 * only!
 * Please comment out the code if using any other
 * custom PGIE + SGIE combinations
 * and use the code as reference to write your own
 * NvDsEventMsgMeta generation code in generate_event_msg_meta()
 * function
 */
typedef enum
{
    APP_CONFIG_ANALYTICS_MODELS_UNKNOWN = 0,
    APP_CONFIG_ANALYTICS_RESNET_PGIE_3SGIE_TYPE_COLOR_MAKE = 1,
} AppConfigAnalyticsModel;

#define RESNET10_PGIE_3SGIE_TYPE_COLOR_MAKECLASS_ID_CAR    (0)
#ifdef GENERATE_DUMMY_META_EXT
#define RESNET10_PGIE_3SGIE_TYPE_COLOR_MAKECLASS_ID_PERSON (2)
#endif
/** @} */

/* PERSON ID definition. */
#define PERSON_ID 0

#ifdef EN_DEBUG
#define LOGD(...) printf(__VA_ARGS__)
#else
#define LOGD(...)
#endif

static TestAppCtx *testAppCtx;
GST_DEBUG_CATEGORY (NVDS_APP);

/** @{ imported from deepstream-app as is */


#define MAX_INSTANCES 128
#define APP_TITLE "DeepStreamTest5App"

#define DEFAULT_X_WINDOW_WIDTH 1920
#define DEFAULT_X_WINDOW_HEIGHT 1080

AppCtx *appCtx[MAX_INSTANCES];
static guint cintr = FALSE;
static GMainLoop *main_loop = NULL;
static gchar **cfg_files = NULL;
static gchar **input_files = NULL;
static gchar **override_cfg_file = NULL;
static gboolean playback_utc = TRUE;
static gboolean print_version = FALSE;
static gboolean show_bbox_text = TRUE;
static gboolean force_tcp = TRUE;
static gboolean print_dependencies_version = FALSE;
static gboolean quit = FALSE;
static gint return_value = 0;
static guint num_instances;
static guint num_input_files;
static GMutex fps_lock;
static gdouble fps[MAX_SOURCE_BINS];
static gdouble fps_avg[MAX_SOURCE_BINS];

static Display *display = NULL;
static Window windows[MAX_INSTANCES] = { 0 };

static GThread *x_event_thread = NULL;
static GMutex disp_lock;

static guint rrow, rcol, rcfg;
static gboolean rrowsel = FALSE, selecting = FALSE;
static AppConfigAnalyticsModel model_used = APP_CONFIG_ANALYTICS_MODELS_UNKNOWN;

/** @} imported from deepstream-app as is */
GOptionEntry entries[] = {
    {"version", 'v', 0, G_OPTION_ARG_NONE, &print_version,
	"Print DeepStreamSDK version", NULL}
    ,
	{"tiledtext", 't', 0, G_OPTION_ARG_NONE, &show_bbox_text,
	    "Display Bounding box labels in tiled mode", NULL}
    ,
	{"version-all", 0, 0, G_OPTION_ARG_NONE, &print_dependencies_version,
	    "Print DeepStreamSDK and dependencies version", NULL}
    ,
	{"cfg-file", 'c', 0, G_OPTION_ARG_FILENAME_ARRAY, &cfg_files,
	    "Set the config file", NULL}
    ,
	{"override-cfg-file", 'o', 0, G_OPTION_ARG_FILENAME_ARRAY, &override_cfg_file,
	    "Set the override config file, used for on-the-fly model update feature",
	    NULL}
    ,
	{"input-file", 'i', 0, G_OPTION_ARG_FILENAME_ARRAY, &input_files,
	    "Set the input file", NULL}
    ,
	{"playback-utc", 'p', 0, G_OPTION_ARG_INT, &playback_utc,
	    "Playback utc; default=true (base UTC from file/rtsp URL); =false (base UTC from file-URL or RTCP Sender Report)",
	    NULL}
    ,
	{"pgie-model-used", 'm', 0, G_OPTION_ARG_INT, &model_used,
	    "PGIE Model used; {0 - Unknown [DEFAULT]}, {1: Resnet 4-class [Car, Bicycle, Person, Roadsign]}",
	    NULL}
    ,
	{"no-force-tcp", 0, G_OPTION_FLAG_REVERSE, G_OPTION_ARG_NONE, &force_tcp,
	    "Do not force TCP for RTP transport", NULL}
    ,
	{NULL}
    ,
};


    static void
generate_ts_rfc3339 (char *buf, int buf_size)
{
    time_t tloc;
    struct tm tm_log;
    struct timespec ts;
    char strmsec[6];              //.nnnZ\0

    clock_gettime (CLOCK_REALTIME, &ts);
    memcpy (&tloc, (void *) (&ts.tv_sec), sizeof (time_t));
    gmtime_r (&tloc, &tm_log);
    strftime (buf, buf_size, "%Y-%m-%dT%H:%M:%S", &tm_log);
    int ms = ts.tv_nsec / 1000000;
    g_snprintf (strmsec, sizeof (strmsec), ".%.3dZ", ms);
    strncat (buf, strmsec, buf_size);
}



    static GstClockTime
generate_ts_rfc3339_from_ts (char *buf, int buf_size, GstClockTime ts,
	gchar * src_uri, gint stream_id)
{
    time_t tloc;
    struct tm tm_log;
    char strmsec[6];              //.nnnZ\0
    int ms;

    GstClockTime ts_generated;

    if (playback_utc
	    || (appCtx[0]->config.multi_source_config[stream_id].type !=
		NV_DS_SOURCE_RTSP)) {
	if (testAppCtx->streams[stream_id].meta_number == 0) {
	    testAppCtx->streams[stream_id].timespec_first_frame =
		extract_utc_from_uri (src_uri);
	    memcpy (&tloc,
		    (void *) (&testAppCtx->streams[stream_id].timespec_first_frame.
			tv_sec), sizeof (time_t));
	    ms = testAppCtx->streams[stream_id].timespec_first_frame.tv_nsec /
		1000000;
	    testAppCtx->streams[stream_id].gst_ts_first_frame = ts;
	    ts_generated =
		GST_TIMESPEC_TO_TIME (testAppCtx->streams[stream_id].
			timespec_first_frame);
	    if (ts_generated == 0) {
		g_print
		    ("WARNING; playback mode used with URI [%s] not conforming to timestamp format;"
		     " check README; using system-time\n", src_uri);
		clock_gettime (CLOCK_REALTIME,
			&testAppCtx->streams[stream_id].timespec_first_frame);
		ts_generated =
		    GST_TIMESPEC_TO_TIME (testAppCtx->streams[stream_id].
			    timespec_first_frame);
	    }
	} else {
	    GstClockTime ts_current =
		GST_TIMESPEC_TO_TIME (testAppCtx->
			streams[stream_id].timespec_first_frame) + (ts -
			    testAppCtx->streams[stream_id].gst_ts_first_frame);
	    struct timespec timespec_current;
	    GST_TIME_TO_TIMESPEC (ts_current, timespec_current);
	    memcpy (&tloc, (void *) (&timespec_current.tv_sec), sizeof (time_t));
	    ms = timespec_current.tv_nsec / 1000000;
	    ts_generated = ts_current;
	}
    } else {
	/** ts itself is UTC Time in ns */
	struct timespec timespec_current;
	GST_TIME_TO_TIMESPEC (ts, timespec_current);
	memcpy (&tloc, (void *) (&timespec_current.tv_sec), sizeof (time_t));
	ms = timespec_current.tv_nsec / 1000000;
	ts_generated = ts;
    }
    gmtime_r (&tloc, &tm_log);
    strftime (buf, buf_size, "%Y-%m-%dT%H:%M:%S", &tm_log);
    g_snprintf (strmsec, sizeof (strmsec), ".%.3dZ", ms);
    strncat (buf, strmsec, buf_size);
    LOGD ("ts=%s\n", buf);

    return ts_generated;
}


    static gpointer
meta_copy_func (gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;
    NvDsEventMsgMeta *dstMeta = NULL;

    dstMeta = g_memdup (srcMeta, sizeof (NvDsEventMsgMeta));

    if (srcMeta->ts)
	dstMeta->ts = g_strdup (srcMeta->ts);

    if (srcMeta->objSignature.size > 0) {
	dstMeta->objSignature.signature = g_memdup (srcMeta->objSignature.signature,
		srcMeta->objSignature.size);
	dstMeta->objSignature.size = srcMeta->objSignature.size;
    }

    if (srcMeta->objectId) {
	dstMeta->objectId = g_strdup (srcMeta->objectId);
    }

    /*
       if (srcMeta->sensorStr) {
       dstMeta->sensorStr = g_strdup (srcMeta->sensorStr);
       }
       */

    if (srcMeta->extMsgSize > 0) {
	if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON) {
	    NvDsPersonObject *srcObj = (NvDsPersonObject *) srcMeta->extMsg;
	    NvDsPersonObject *obj =
		(NvDsPersonObject *) g_malloc0 (sizeof (NvDsPersonObject));

	    obj->age = srcObj->age;

	    if (srcObj->gender)
		obj->gender = g_strdup (srcObj->gender);
	    if (srcObj->cap)
		obj->cap = g_strdup (srcObj->cap);
	    if (srcObj->hair)
		obj->hair = g_strdup (srcObj->hair);
	    if (srcObj->apparel)
		obj->apparel = g_strdup (srcObj->apparel);

	    dstMeta->extMsg = obj;
	    dstMeta->extMsgSize = sizeof (NvDsPersonObject);
	}
    }

    return dstMeta;
}

    static void
meta_free_func (gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;
    user_meta->user_meta_data = NULL;

    if (srcMeta->ts) {
	g_free (srcMeta->ts);
    }

    if (srcMeta->objSignature.size > 0) {
	g_free (srcMeta->objSignature.signature);
	srcMeta->objSignature.size = 0;
    }

    if (srcMeta->objectId) {
	g_free (srcMeta->objectId);
    }

    if (srcMeta->sensorStr) {
	g_free (srcMeta->sensorStr);
    }

    if (srcMeta->extMsgSize > 0) {
	if (srcMeta->objType == NVDS_OBJECT_TYPE_VEHICLE) {
	    NvDsVehicleObject *obj = (NvDsVehicleObject *) srcMeta->extMsg;
	    if (obj->type)
		g_free (obj->type);
	    if (obj->color)
		g_free (obj->color);
	    if (obj->make)
		g_free (obj->make);
	    if (obj->model)
		g_free (obj->model);
	    if (obj->license)
		g_free (obj->license);
	    if (obj->region)
		g_free (obj->region);
	} else if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON) {
	    NvDsPersonObject *obj = (NvDsPersonObject *) srcMeta->extMsg;

	    if (obj->gender)
		g_free (obj->gender);
	    if (obj->cap)
		g_free (obj->cap);
	    if (obj->hair)
		g_free (obj->hair);
	    if (obj->apparel)
		g_free (obj->apparel);
	}
	g_free (srcMeta->extMsg);
	srcMeta->extMsg = NULL;
	srcMeta->extMsgSize = 0;
    }
    g_free (srcMeta);
}

#ifdef GENERATE_DUMMY_META_EXT

#endif /**< GENERATE_DUMMY_META_EXT */

    static void
generate_event_msg_meta (gpointer data, gint class_id, gboolean useTs,
	GstClockTime ts, gchar * src_uri, gint stream_id, guint sensor_id,
	AnalyticsUserMeta * obj_params, float scaleW, float scaleH,
	NvDsFrameMeta * frame_meta)
{
    NvDsEventMsgMeta *meta = (NvDsEventMsgMeta *) data;
    GstClockTime ts_generated = 0;
    meta->objType = NVDS_OBJECT_TYPE_UNKNOWN; /**< object unknown */
    meta->frameId = frame_meta->frame_num;
    meta->ts = (gchar *) g_malloc0 (MAX_TIME_STAMP_LEN + 1);
    meta->objectId = (gchar *) g_malloc0 (MAX_LABEL_SIZE);

    //strncpy (meta->objectId, obj_params->obj_label, MAX_LABEL_SIZE);

    /** INFO: This API is called once for every 30 frames (now) */
    if (useTs && src_uri) {
	ts_generated =
	    generate_ts_rfc3339_from_ts (meta->ts, MAX_TIME_STAMP_LEN, ts, src_uri,
		    stream_id);
    } else {
	generate_ts_rfc3339 (meta->ts, MAX_TIME_STAMP_LEN);
    }

    /** tracking ID */
    meta->trackingId = class_id;

    (void) ts_generated;
    meta->type = NVDS_EVENT_ENTRY;
    meta->objType = NVDS_OBJECT_TYPE_PERSON;
    meta->objClassId = PERSON_ID;
    meta->occupancy = obj_params->lccum_cnt;
    meta->lccum_cnt_entry = obj_params->lcc_cnt_entry;
    meta->lccum_cnt_exit = obj_params->lcc_cnt_exit ;
    meta->source_id = obj_params->source_id;
//    g_print("source id: %d, Enter: %d, Exit: %d\n", meta->source_id, meta->lccum_cnt_entry, meta->lccum_cnt_exit);

}

/*
 * Access analytics data.
 */
void analytics_custom_parse_nvdsanalytics_meta_data (NvDsMetaList *l_user, AnalyticsUserMeta *data);

/**
 * Callback function to be called once all inferences (Primary + Secondary)
 * are done. This is opportunity to modify content of the metadata.
 * e.g. Here Person is being replaced with Man/Woman and corresponding counts
 * are being maintained. It should be modified according to network classes
 * or can be removed altogether if not required.
 */
    static void
bbox_generated_probe_after_analytics (AppCtx * appCtx, GstBuffer * buf,
	NvDsBatchMeta * batch_meta, guint index)
{
    NvDsObjectMeta *obj_meta = NULL;
    GstClockTime buffer_pts = 0;
    guint32 stream_id = 0;


    if (!appCtx->config.dsanalytics_config.enable){
	g_print ("Unable to get nvdsanalytics src pad\n");
	return;	
    }

    for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
	    l_frame = l_frame->next) {
	NvDsFrameMeta *frame_meta = l_frame->data;
	stream_id = frame_meta->source_id;
	GstClockTime buf_ntp_time = 0;
	if (playback_utc == FALSE) {
	    /** Calculate the buffer-NTP-time
	     * derived from this stream's RTCP Sender Report here:
	     */
	    StreamSourceInfo *src_stream = &testAppCtx->streams[stream_id];
	    buf_ntp_time = frame_meta->ntp_timestamp;

	    if (buf_ntp_time < src_stream->last_ntp_time) {
		NVGSTDS_WARN_MSG_V ("Source %d: NTP timestamps are backward in time."
			" Current: %lu previous: %lu", stream_id, buf_ntp_time,
			src_stream->last_ntp_time);
	    }
	    src_stream->last_ntp_time = buf_ntp_time;
	}

	GList *l;
	NvDsMetaList *l_analyticsuser;
	l_analyticsuser = frame_meta->frame_user_meta_list;

	AnalyticsUserMeta *user_data = 
	    (AnalyticsUserMeta *) g_malloc0(sizeof(AnalyticsUserMeta));
	if (l_analyticsuser != NULL) {
	    analytics_custom_parse_nvdsanalytics_meta_data(l_analyticsuser, user_data);
	}
	user_data->source_id = stream_id;

	//		l_analyticsuser = l_analyticsuser->next;

	/* Code from test5 application */
	for (l = frame_meta->obj_meta_list; l != NULL; l = l->next) {
	    /* Now using above information we need to form a text that should
	     * be displayed on top of the bounding box, so lets form it here. */
	    obj_meta = (NvDsObjectMeta *) (l->data);

	    {
		/**
		 * Enable only if this callback is after tiler
		 * NOTE: Scaling back code-commented
		 * now that bbox_generated_probe_after_analytics() is post analytics
		 * (say pgie, tracker or sgie)
		 * and before tiler, no plugin shall scale metadata and will be
		 * corresponding to the nvstreammux resolution
		 */
		float scaleW = 0;
		float scaleH = 0;
		/* Frequency of messages to be send will be based on use case.
		 * Here message is being sent for first object every 30 frames.
		 */
		buffer_pts = frame_meta->buf_pts;
		if (!appCtx->config.streammux_config.pipeline_width
			|| !appCtx->config.streammux_config.pipeline_height) {
		    g_print ("invalid pipeline params\n");
		    return;
		}
		LOGD ("stream %d==%d [%d X %d]\n", frame_meta->source_id,
			frame_meta->pad_index, frame_meta->source_frame_width,
			frame_meta->source_frame_height);
		scaleW =
		    (float) frame_meta->source_frame_width /
		    appCtx->config.streammux_config.pipeline_width;
		scaleH =
		    (float) frame_meta->source_frame_height /
		    appCtx->config.streammux_config.pipeline_height;

		if (playback_utc == FALSE) {
		    /** Use the buffer-NTP-time derived from this stream's RTCP Sender
		     * Report here:
		     */
		    buffer_pts = buf_ntp_time;
		}
		/** Generate NvDsEventMsgMeta for every object */
		NvDsEventMsgMeta *msg_meta =
		    (NvDsEventMsgMeta *) g_malloc0 (sizeof (NvDsEventMsgMeta));
		generate_event_msg_meta (msg_meta, PERSON_ID, TRUE,
			/**< useTs NOTE: Pass FALSE for files without base-timestamp in URI */
			buffer_pts,
			appCtx->config.multi_source_config[stream_id].uri, stream_id,
			appCtx->config.multi_source_config[stream_id].camera_id,
			user_data, scaleW, scaleH, frame_meta);
		testAppCtx->streams[stream_id].meta_number++;
		NvDsUserMeta *user_event_meta =
		    nvds_acquire_user_meta_from_pool (batch_meta);
		if (user_event_meta) {
		    /*
		     * Since generated event metadata has custom objects for
		     * Vehicle / Person which are allocated dynamically, we are
		     * setting copy and free function to handle those fields when
		     * metadata copy happens between two components.
		     */
		    user_event_meta->user_meta_data = (void *) msg_meta;
		    user_event_meta->base_meta.batch_meta = batch_meta;
		    user_event_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
		    user_event_meta->base_meta.copy_func =
			(NvDsMetaCopyFunc) meta_copy_func;
		    user_event_meta->base_meta.release_func =
			(NvDsMetaReleaseFunc) meta_free_func;
		    nvds_add_user_meta_to_frame (frame_meta, user_event_meta);
		} else {
		    g_print ("Error in attaching event meta to buffer\n");
		}
	    }
	}
	testAppCtx->streams[stream_id].frameCount++;

	g_free(user_data);
    }
}

/** @{ imported from deepstream-app as is */

/**
 * Function to handle program interrupt signal.
 * It installs default handler after handling the interrupt.
 */
    static void
_intr_handler (int signum)
{
    struct sigaction action;

    NVGSTDS_ERR_MSG_V ("User Interrupted.. \n");

    memset (&action, 0, sizeof (action));
    action.sa_handler = SIG_DFL;

    sigaction (SIGINT, &action, NULL);

    cintr = TRUE;
}

/**
 * callback function to print the performance numbers of each stream.
 */
    static void
perf_cb (gpointer context, NvDsAppPerfStruct * str)
{
    static guint header_print_cnt = 0;
    guint i;
    AppCtx *appCtx = (AppCtx *) context;
    guint numf = str->num_instances;

    g_mutex_lock (&fps_lock);
    for (i = 0; i < numf; i++) {
	fps[i] = str->fps[i];
	fps_avg[i] = str->fps_avg[i];
    }

    if (header_print_cnt % 20 == 0) {
	g_print ("\n**PERF:  ");
	for (i = 0; i < numf; i++) {
	    g_print ("FPS %d (Avg)\t", i);
	}
	g_print ("\n");
	header_print_cnt = 0;
    }
    header_print_cnt++;

    time_t t = time (NULL);
    struct tm *tm = localtime (&t);
    printf ("%s", asctime (tm));
    if (num_instances > 1)
	g_print ("PERF(%d): ", appCtx->index);
    else
	g_print ("**PERF:  ");

    for (i = 0; i < numf; i++) {
	g_print ("%.2f (%.2f)\t", fps[i], fps_avg[i]);
    }
    g_print ("\n");
    g_mutex_unlock (&fps_lock);
}

/**
 * Loop function to check the status of interrupts.
 * It comes out of loop if application got interrupted.
 */
    static gboolean
check_for_interrupt (gpointer data)
{
    if (quit) {
	return FALSE;
    }

    if (cintr) {
	cintr = FALSE;

	quit = TRUE;
	g_main_loop_quit (main_loop);

	return FALSE;
    }
    return TRUE;
}

/*
 * Function to install custom handler for program interrupt signal.
 */
    static void
_intr_setup (void)
{
    struct sigaction action;

    memset (&action, 0, sizeof (action));
    action.sa_handler = _intr_handler;

    sigaction (SIGINT, &action, NULL);
}

    static gboolean
kbhit (void)
{
    struct timeval tv;
    fd_set rdfs;

    tv.tv_sec = 0;
    tv.tv_usec = 0;

    FD_ZERO (&rdfs);
    FD_SET (STDIN_FILENO, &rdfs);

    select (STDIN_FILENO + 1, &rdfs, NULL, NULL, &tv);
    return FD_ISSET (STDIN_FILENO, &rdfs);
}

/*
 * Function to enable / disable the canonical mode of terminal.
 * In non canonical mode input is available immediately (without the user
 * having to type a line-delimiter character).
 */
    static void
changemode (int dir)
{
    static struct termios oldt, newt;

    if (dir == 1) {
	tcgetattr (STDIN_FILENO, &oldt);
	newt = oldt;
	newt.c_lflag &= ~(ICANON);
	tcsetattr (STDIN_FILENO, TCSANOW, &newt);
    } else
	tcsetattr (STDIN_FILENO, TCSANOW, &oldt);
}

    static void
print_runtime_commands (void)
{
    g_print ("\nRuntime commands:\n"
	    "\th: Print this help\n"
	    "\tq: Quit\n\n" "\tp: Pause\n" "\tr: Resume\n\n");

    if (appCtx[0]->config.tiled_display_config.enable) {
	g_print
	    ("NOTE: To expand a source in the 2D tiled display and view object details,"
	     " left-click on the source.\n"
	     "      To go back to the tiled display, right-click anywhere on the window.\n\n");
    }
}

/**
 * Loop function to check keyboard inputs and status of each pipeline.
 */
    static gboolean
event_thread_func (gpointer arg)
{
    guint i;
    gboolean ret = TRUE;

    // Check if all instances have quit
    for (i = 0; i < num_instances; i++) {
	if (!appCtx[i]->quit)
	    break;
    }

    if (i == num_instances) {
	quit = TRUE;
	g_main_loop_quit (main_loop);
	return FALSE;
    }
    // Check for keyboard input
    if (!kbhit ()) {
	//continue;
	return TRUE;
    }
    int c = fgetc (stdin);
    g_print ("\n");

    gint source_id;
    GstElement *tiler = appCtx[rcfg]->pipeline.tiled_display_bin.tiler;
    g_object_get (G_OBJECT (tiler), "show-source", &source_id, NULL);

    if (selecting) {
	if (rrowsel == FALSE) {
	    if (c >= '0' && c <= '9') {
		rrow = c - '0';
		g_print ("--selecting source  row %d--\n", rrow);
		rrowsel = TRUE;
	    }
	} else {
	    if (c >= '0' && c <= '9') {
		int tile_num_columns = appCtx[rcfg]->config.tiled_display_config.columns;
		rcol = c - '0';
		selecting = FALSE;
		rrowsel = FALSE;
		source_id = tile_num_columns * rrow + rcol;
		g_print ("--selecting source  col %d sou=%d--\n", rcol, source_id);
		if (source_id >= (gint) appCtx[rcfg]->config.num_source_sub_bins) {
		    source_id = -1;
		} else {
		    appCtx[rcfg]->show_bbox_text = TRUE;
		    appCtx[rcfg]->active_source_index = source_id;
		    g_object_set (G_OBJECT (tiler), "show-source", source_id, NULL);
		}
	    }
	}
    }
    switch (c) {
	case 'h':
	    print_runtime_commands ();
	    break;
	case 'p':
	    for (i = 0; i < num_instances; i++)
		pause_pipeline (appCtx[i]);
	    break;
	case 'r':
	    for (i = 0; i < num_instances; i++)
		resume_pipeline (appCtx[i]);
	    break;
	case 'q':
	    quit = TRUE;
	    g_main_loop_quit (main_loop);
	    ret = FALSE;
	    break;
	case 'c':
	    if (selecting == FALSE && source_id == -1) {
		g_print("--selecting config file --\n");
		c = fgetc(stdin);
		if (c >= '0' && c <= '9') {
		    rcfg = c - '0';
		    if (rcfg < num_instances) {
			g_print("--selecting config  %d--\n", rcfg);
		    } else {
			g_print("--selected config file %d out of bound, reenter\n", rcfg);
			rcfg = 0;
		    }
		}
	    }
	    break;
	case 'z':
	    if (source_id == -1 && selecting == FALSE) {
		g_print ("--selecting source --\n");
		selecting = TRUE;
	    } else {
		if (!show_bbox_text) {
		    GstElement *nvosd = appCtx[rcfg]->pipeline.instance_bins[0].osd_bin.nvosd;
		    g_object_set (G_OBJECT (nvosd), "display-text", FALSE, NULL);
		    g_object_set (G_OBJECT (tiler), "show-source", -1, NULL);
		}
		appCtx[rcfg]->active_source_index = -1;
		selecting = FALSE;
		rcfg = 0;
		g_print("--tiled mode --\n");
	    }
	    break;
	default:
	    break;
    }
    return ret;
}

    static int
get_source_id_from_coordinates (float x_rel, float y_rel, AppCtx *appCtx)
{
    int tile_num_rows = appCtx->config.tiled_display_config.rows;
    int tile_num_columns = appCtx->config.tiled_display_config.columns;

    int source_id = (int) (x_rel * tile_num_columns);
    source_id += ((int) (y_rel * tile_num_rows)) * tile_num_columns;

    /* Don't allow clicks on empty tiles. */
    if (source_id >= (gint) appCtx->config.num_source_sub_bins)
	source_id = -1;

    return source_id;
}

/**
 * Thread to monitor X window events.
 */
    static gpointer
nvds_x_event_thread (gpointer data)
{
    g_mutex_lock (&disp_lock);
    while (display) {
	XEvent e;
	guint index;
	while (XPending (display)) {
	    XNextEvent (display, &e);
	    switch (e.type) {
		case ButtonPress:
		    {
			XWindowAttributes win_attr;
			XButtonEvent ev = e.xbutton;
			gint source_id;
			GstElement *tiler;

			XGetWindowAttributes (display, ev.window, &win_attr);

			for (index = 0; index < MAX_INSTANCES; index++)
			    if (ev.window == windows[index])
				break;

			tiler = appCtx[index]->pipeline.tiled_display_bin.tiler;
			g_object_get (G_OBJECT (tiler), "show-source", &source_id, NULL);

			if (ev.button == Button1 && source_id == -1) {
			    source_id =
				get_source_id_from_coordinates (ev.x * 1.0 / win_attr.width,
					ev.y * 1.0 / win_attr.height, appCtx[index]);
			    if (source_id > -1) {
				g_object_set (G_OBJECT (tiler), "show-source", source_id, NULL);
				appCtx[index]->active_source_index = source_id;
				appCtx[index]->show_bbox_text = TRUE;
				GstElement *nvosd = appCtx[index]->pipeline.instance_bins[0].osd_bin.nvosd;
				g_object_set (G_OBJECT (nvosd), "display-text", TRUE, NULL);
			    }
			} else if (ev.button == Button3) {
			    g_object_set (G_OBJECT (tiler), "show-source", -1, NULL);
			    appCtx[index]->active_source_index = -1;
			    if (!show_bbox_text) {
				appCtx[index]->show_bbox_text = FALSE;
				GstElement *nvosd = appCtx[index]->pipeline.instance_bins[0].osd_bin.nvosd;
				g_object_set (G_OBJECT (nvosd), "display-text", FALSE, NULL);
			    }
			}
		    }
		    break;
		case KeyRelease:
		    {
			KeySym p, r, q;
			guint i;
			p = XKeysymToKeycode (display, XK_P);
			r = XKeysymToKeycode (display, XK_R);
			q = XKeysymToKeycode (display, XK_Q);
			if (e.xkey.keycode == p) {
			    for (i = 0; i < num_instances; i++)
				pause_pipeline (appCtx[i]);
			    break;
			}
			if (e.xkey.keycode == r) {
			    for (i = 0; i < num_instances; i++)
				resume_pipeline (appCtx[i]);
			    break;
			}
			if (e.xkey.keycode == q) {
			    quit = TRUE;
			    g_main_loop_quit (main_loop);
			}
		    }
		    break;
		case ClientMessage:
		    {
			Atom wm_delete;
			for (index = 0; index < MAX_INSTANCES; index++)
			    if (e.xclient.window == windows[index])
				break;

			wm_delete = XInternAtom (display, "WM_DELETE_WINDOW", 1);
			if (wm_delete != None && wm_delete == (Atom) e.xclient.data.l[0]) {
			    quit = TRUE;
			    g_main_loop_quit (main_loop);
			}
		    }
		    break;
	    }
	}
	g_mutex_unlock (&disp_lock);
	g_usleep (G_USEC_PER_SEC / 20);
	g_mutex_lock (&disp_lock);
    }
    g_mutex_unlock (&disp_lock);
    return NULL;
}

/**
 * callback function to add application specific metadata.
 * Here it demonstrates how to display the URI of source in addition to
 * the text generated after inference.
 */
    static gboolean
overlay_graphics (AppCtx * appCtx, GstBuffer * buf,
	NvDsBatchMeta * batch_meta, guint index)
{
    return TRUE;
}

/** @} imported from deepstream-app as is */

    int
main (int argc, char *argv[])
{
    testAppCtx = (TestAppCtx *) g_malloc0 (sizeof (TestAppCtx));
    GOptionContext *ctx = NULL;
    GOptionGroup *group = NULL;
    GError *error = NULL;
    guint i;

    ctx = g_option_context_new ("Nvidia DeepStream Test5");
    group = g_option_group_new ("abc", NULL, NULL, NULL, NULL);
    g_option_group_add_entries (group, entries);

    g_option_context_set_main_group (ctx, group);
    g_option_context_add_group (ctx, gst_init_get_option_group ());

    GST_DEBUG_CATEGORY_INIT (NVDS_APP, "NVDS_APP", 0, NULL);

    if (!g_option_context_parse (ctx, &argc, &argv, &error)) {
	NVGSTDS_ERR_MSG_V ("%s", error->message);
	g_print ("%s",g_option_context_get_help (ctx, TRUE, NULL));
	return -1;
    }

    if (print_version) {
	g_print ("deepstream-test5-app version %d.%d.%d\n",
		NVDS_APP_VERSION_MAJOR, NVDS_APP_VERSION_MINOR, NVDS_APP_VERSION_MICRO);
	return 0;
    }

    if (print_dependencies_version) {
	g_print ("deepstream-test5-app version %d.%d.%d\n",
		NVDS_APP_VERSION_MAJOR, NVDS_APP_VERSION_MINOR, NVDS_APP_VERSION_MICRO);
	return 0;
    }

    if (cfg_files) {
	num_instances = g_strv_length (cfg_files);
    }
    if (input_files) {
	num_input_files = g_strv_length (input_files);
    }

    if (!cfg_files || num_instances == 0) {
	NVGSTDS_ERR_MSG_V ("Specify config file with -c option");
	return_value = -1;
	goto done;
    }

    for (i = 0; i < num_instances; i++) {
	appCtx[i] = (AppCtx *) g_malloc0 (sizeof (AppCtx));
	appCtx[i]->person_class_id = -1;
	appCtx[i]->car_class_id = -1;
	appCtx[i]->index = i;
	appCtx[i]->active_source_index = -1;
	if (show_bbox_text) {
	    appCtx[i]->show_bbox_text = TRUE;
	}

	if (input_files && input_files[i]) {
	    appCtx[i]->config.multi_source_config[0].uri =
		g_strdup_printf ("file://%s", input_files[i]);
	    g_free (input_files[i]);
	}

	if (!parse_config_file (&appCtx[i]->config, cfg_files[i])) {
	    NVGSTDS_ERR_MSG_V ("Failed to parse config file '%s'", cfg_files[i]);
	    appCtx[i]->return_value = -1;
	    goto done;
	}

	if (override_cfg_file && override_cfg_file[i]) {
	    if (!g_file_test (override_cfg_file[i],
			G_FILE_TEST_IS_REGULAR | G_FILE_TEST_IS_SYMLINK))
	    {
		g_print ("Override file %s does not exist, quitting...\n",
			override_cfg_file[i]);
		appCtx[i]->return_value = -1;
		goto done;
	    }
	    	}
    }

    for (i = 0; i < num_instances; i++) {
	for (guint j = 0; j < appCtx[i]->config.num_source_sub_bins; j++) {
	    /** Force the source (applicable only if RTSP)
	     * to use TCP for RTP/RTCP channels.
	     * forcing TCP to avoid problems with UDP port usage from within docker-
	     * container.
	     * The UDP RTCP channel when run within docker had issues receiving
	     * RTCP Sender Reports from server
	     */
	    if (force_tcp)
		appCtx[i]->config.multi_source_config[j].select_rtp_protocol = 0x04;
	}
	if (!create_pipeline (appCtx[i], bbox_generated_probe_after_analytics,
		    NULL, perf_cb, overlay_graphics)) {
	    NVGSTDS_ERR_MSG_V ("Failed to create pipeline");
	    return_value = -1;
	    goto done;
	}

	/*  if (appCtx[i]->config.dsanalytics_config.enable){
	    GstPad *src_pad = NULL;
	    GstElement *nvdsanalytics = appCtx[i]->pipeline.common_elements.dsanalytics_bin.elem_dsanalytics;
	    src_pad = gst_element_get_static_pad (nvdsanalytics, "src");
	    if (!src_pad)
	    g_print ("Unable to get nvdsanalytics src pad\n");
	    else
	    {
	    gst_pad_add_probe (src_pad, GST_PAD_PROBE_TYPE_BUFFER,
	    nvdsanalytics_src_pad_buffer_probe, NULL, NULL);
	    gst_object_unref (src_pad);
	    }
	    }*/

	/** Now add probe to RTPSession plugin src pad */
	for (guint j = 0; j < appCtx[i]->pipeline.multi_src_bin.num_bins; j++) {
	    testAppCtx->streams[j].id = j;
	}
	/** In test5 app, as we could have several sources connected
	 * for a typical IoT use-case, raising the nvstreammux's
	 * buffer-pool-size to 16 */
	g_object_set (appCtx[i]->pipeline.multi_src_bin.streammux,
		"buffer-pool-size", STREAMMUX_BUFFER_POOL_SIZE, NULL);
    }

    main_loop = g_main_loop_new (NULL, FALSE);

    _intr_setup ();
    g_timeout_add (400, check_for_interrupt, NULL);

    g_mutex_init (&disp_lock);
    display = XOpenDisplay (NULL);
    for (i = 0; i < num_instances; i++) {
	guint j;

	if (!show_bbox_text) {
	    GstElement *nvosd = appCtx[i]->pipeline.instance_bins[0].osd_bin.nvosd;
	    g_object_set(G_OBJECT(nvosd), "display-text", FALSE, NULL);
	}

	if (gst_element_set_state (appCtx[i]->pipeline.pipeline,
		    GST_STATE_PAUSED) == GST_STATE_CHANGE_FAILURE) {
	    NVGSTDS_ERR_MSG_V ("Failed to set pipeline to PAUSED");
	    return_value = -1;
	    goto done;
	}

	if (!appCtx[i]->config.tiled_display_config.enable)
	    continue;

	for (j = 0; j < appCtx[i]->config.num_sink_sub_bins; j++) {
	    XTextProperty xproperty;
	    gchar *title;
	    guint width, height;
	    XSizeHints hints = {0};

	    if (!GST_IS_VIDEO_OVERLAY (appCtx[i]->pipeline.instance_bins[0].sink_bin.
			sub_bins[j].sink)) {
		continue;
	    }

	    if (!display) {
		NVGSTDS_ERR_MSG_V ("Could not open X Display");
		return_value = -1;
		goto done;
	    }

	    if (appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.width)
		width =
		    appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.width;
	    else
		width = appCtx[i]->config.tiled_display_config.width;

	    if (appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.height)
		height =
		    appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.height;
	    else
		height = appCtx[i]->config.tiled_display_config.height;

	    width = (width) ? width : DEFAULT_X_WINDOW_WIDTH;
	    height = (height) ? height : DEFAULT_X_WINDOW_HEIGHT;

	    hints.flags = PPosition | PSize;
	    hints.x = appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.offset_x;
	    hints.y = appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.offset_y;
	    hints.width = width;
	    hints.height = height;

	    windows[i] =
		XCreateSimpleWindow (display, RootWindow (display,
			    DefaultScreen (display)), hints.x, hints.y, width, height, 2,
			0x00000000, 0x00000000);

	    XSetNormalHints(display, windows[i], &hints);

	    if (num_instances > 1)
		title = g_strdup_printf (APP_TITLE "-%d", i);
	    else
		title = g_strdup (APP_TITLE);
	    if (XStringListToTextProperty ((char **) &title, 1, &xproperty) != 0) {
		XSetWMName (display, windows[i], &xproperty);
		XFree (xproperty.value);
	    }

	    XSetWindowAttributes attr = { 0 };
	    if ((appCtx[i]->config.tiled_display_config.enable &&
			appCtx[i]->config.tiled_display_config.rows *
			appCtx[i]->config.tiled_display_config.columns == 1) ||
		    (appCtx[i]->config.tiled_display_config.enable == 0 &&
		     appCtx[i]->config.num_source_sub_bins == 1)) {
	    } else {
		attr.event_mask = ButtonPress | KeyRelease;
	    }
	    XChangeWindowAttributes (display, windows[i], CWEventMask, &attr);

	    Atom wmDeleteMessage = XInternAtom (display, "WM_DELETE_WINDOW", False);
	    if (wmDeleteMessage != None) {
		XSetWMProtocols (display, windows[i], &wmDeleteMessage, 1);
	    }
	    XMapRaised (display, windows[i]);
	    XSync (display, 1);       //discard the events for now
	    gst_video_overlay_set_window_handle (GST_VIDEO_OVERLAY (appCtx
			[i]->pipeline.instance_bins[0].sink_bin.sub_bins[j].sink),
		    (gulong) windows[i]);
	    gst_video_overlay_expose (GST_VIDEO_OVERLAY (appCtx[i]->pipeline.
			instance_bins[0].sink_bin.sub_bins[j].sink));
	    if (!x_event_thread)
		x_event_thread = g_thread_new ("nvds-window-event-thread",
			nvds_x_event_thread, NULL);
	}
    }

    /* Dont try to set playing state if error is observed */
    if (return_value != -1) {
	for (i = 0; i < num_instances; i++) {
	    if (gst_element_set_state (appCtx[i]->pipeline.pipeline,
			GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {

		g_print ("\ncan't set pipeline to playing state.\n");
		return_value = -1;
		goto done;
	    }
	}
    }

    print_runtime_commands ();

    changemode (1);

    g_timeout_add (40, event_thread_func, NULL);
    g_main_loop_run (main_loop);

    changemode (0);

done:

    g_print ("Quitting\n");
    for (i = 0; i < num_instances; i++) {
	if (appCtx[i] == NULL)
	    continue;

	if (appCtx[i]->return_value == -1)
	    return_value = -1;

	destroy_pipeline (appCtx[i]);

	g_mutex_lock (&disp_lock);
	if (windows[i])
	    XDestroyWindow (display, windows[i]);
	windows[i] = 0;
	g_mutex_unlock (&disp_lock);

	g_free (appCtx[i]);
    }

    g_mutex_lock (&disp_lock);
    if (display)
	XCloseDisplay (display);
    display = NULL;
    g_mutex_unlock (&disp_lock);
    g_mutex_clear (&disp_lock);

    if (main_loop) {
	g_main_loop_unref (main_loop);
    }

    if (ctx) {
	g_option_context_free (ctx);
    }

    if (return_value == 0) {
	g_print ("App run successful\n");
    } else {
	g_print ("App run failed\n");
    }

    gst_deinit ();

    return return_value;

    g_free (testAppCtx);

    return 0;
}

    static gchar *
get_first_result_label (NvDsClassifierMeta * classifierMeta)
{
    GList *n;
    for (n = classifierMeta->label_info_list; n != NULL; n = n->next) {
	NvDsLabelInfo *labelInfo = (NvDsLabelInfo *) (n->data);
	if (labelInfo->result_label[0] != '\0') {
	    return g_strdup (labelInfo->result_label);
	}
    }
    return NULL;
}


