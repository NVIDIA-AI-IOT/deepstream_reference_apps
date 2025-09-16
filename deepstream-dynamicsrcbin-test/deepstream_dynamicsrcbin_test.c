/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gst/gst.h>
#include <pthread.h>

/* Configuration constants */
#define N_CHUNKS 2
#define VIDEO_FILE "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"

/**
 * @brief Application data structure to hold pipeline components and state
 */
typedef struct {
    GstElement *pipeline;    /* Main GStreamer pipeline */
    GstElement *src;         /* nvdsdynamicsrcbin source element */
    GstElement *sink;        /* fakesink element for testing */
    GMainLoop *loop;         /* Main event loop */
    pthread_t signal_thread;  /* Thread for sending dynamic source signals */
    gboolean thread_running;  /* Flag to track thread state */
} AppData;

/* Global variables for tracking frame processing */
guint frame_count = 0;
gint current_source_id = -1;

/**
 * @brief Pad probe callback for sink input to monitor frame processing
 * 
 * This callback is attached to the sink pad to:
 * - Count processed frames
 * - Handle custom stream start events from nvdsdynamicsrcbin
 * - Track source ID changes
 * 
 * @param pad The sink pad being probed
 * @param info Probe information containing buffer or event data
 * @param user_data User data (unused in this implementation)
 * @return GST_PAD_PROBE_OK to continue processing
 */
static GstPadProbeReturn sink_input_buffer_callback(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) {
        frame_count++;
        g_print("Frame Number: %d, Current Source ID: %d\n", frame_count, current_source_id);
    } else if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM) {
        GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);
        /* Check for custom stream start event from nvdsdynamicsrcbin */
        if (GST_EVENT_TYPE(event) == GST_EVENT_CUSTOM_DOWNSTREAM) {
            const GstStructure *structure = gst_event_get_structure(event);
            if (structure && gst_structure_has_name(structure, "custom-stream-start-event")) {
                gint source_id;
                if (gst_structure_get_int(structure, "source-id", &source_id)) {
                    g_print("Source change event received for source ID: %d\n", source_id);
                    current_source_id = source_id;
                    /* Reset frame count for new source */
                    frame_count = 0;
                }
            }
        }
    }
    return GST_PAD_PROBE_OK;
}

/**
 * @brief Utility function to find an element by factory name within a bin
 * 
 * @param bin The GStreamer bin to search in
 * @param factory_name The factory name of the element to find
 * @return Pointer to the found element (with increased ref count) or NULL if not found
 */
GstElement* get_element_by_factory_name(GstBin *bin, const gchar *factory_name) {
    GstElement *element = NULL;
    GstIterator *iterator = gst_bin_iterate_all_by_element_factory_name(GST_BIN(bin), factory_name);
    GValue item = G_VALUE_INIT;
    
    /* Get first matching element */
    if (gst_iterator_next(iterator, &item) == GST_ITERATOR_OK) {
        element = GST_ELEMENT(g_value_get_object(&item));
        gst_object_ref(element); /* Increase ref count since we'll be using it */
        g_value_reset(&item);
    }
    
    gst_iterator_free(iterator);
    return element;
}

/**
 * @brief Load a new file into the dynamic source bin
 * 
 * Creates and adds filesrc, queue, and parsebin elements to the nvdsdynamicsrcbin
 * and sets them to PLAYING state.
 * 
 * @param dynamicsrcbin The nvdsdynamicsrcbin element
 * @param current_file Path to the file to load
 * @return TRUE if successful, FALSE otherwise
 */
static gboolean load_file(GstElement *dynamicsrcbin, gchar *current_file) {
    /* Create new elements for the file */
    GstElement *filesrc = gst_element_factory_make("filesrc", "source");
    GstElement *queue = gst_element_factory_make("queue", "queue");
    GstElement *parsebin = gst_element_factory_make("parsebin", "parsebin");

    if (!filesrc || !queue || !parsebin) {
        g_print("One of the elements not successfully created\n");
        return FALSE;
    }

    /* Update filesrc location */
    g_object_set(filesrc, "location", current_file, NULL);
    g_print("Playing next file: %s\n", current_file);

    /* Add the elements to dynamicsrcbin */
    gst_bin_add_many(GST_BIN(dynamicsrcbin), filesrc, queue, parsebin, NULL);

    /* Link elements */
    gst_element_link_many(filesrc, queue, parsebin, NULL);

    /* Set elements to PLAYING state */
    gst_element_set_state(parsebin, GST_STATE_PLAYING);
    gst_element_set_state(queue, GST_STATE_PLAYING);
    gst_element_set_state(filesrc, GST_STATE_PLAYING);

    return TRUE;
}

/**
 * @brief Timeout callback to attempt loading a file
 * 
 * This function is called periodically to check if a file is ready to be loaded
 * into the nvdsdynamicsrcbin.
 * 
 * @param data Pointer to the nvdsdynamicsrcbin element
 * @return TRUE to continue calling, FALSE to stop
 */
static gboolean try_load_file(gpointer data) {
    GstElement *dynamicsrcbin = GST_ELEMENT(data);
    int current_id = -1;
    gchar *current_file = NULL;
    
    g_object_get(G_OBJECT(dynamicsrcbin), "current-id", &current_id, NULL);
    g_object_get(G_OBJECT(dynamicsrcbin), "current-file", &current_file, NULL);
    
    if (!current_file || !current_id) {
        g_print("Waiting for file to come...\n");
        return TRUE;
    }
    
    if (!load_file(dynamicsrcbin, current_file)) {
        g_print("Failed to load file: %s\n", current_file);
    }
    
    g_free(current_file);
    /* File loaded, no need to try again */
    return FALSE;
}

/**
 * @brief Handle dynamic source message from nvdsdynamicsrcbin
 * 
 * This function is called when the nvdsdynamicsrcbin sends a file change message.
 * It removes the old elements and loads the new file.
 * 
 * @param dynamicsrcbin The nvdsdynamicsrcbin element
 * @return TRUE if successful, FALSE otherwise
 */
static gboolean handle_dynamic_source_message(GstElement *dynamicsrcbin) {
    int current_id = -1;
    gchar *current_file = NULL;
    GstElement *filesrc = get_element_by_factory_name(GST_BIN(dynamicsrcbin), "filesrc");
    GstElement *queue = get_element_by_factory_name(GST_BIN(dynamicsrcbin), "queue");
    GstElement *parsebin = get_element_by_factory_name(GST_BIN(dynamicsrcbin), "parsebin");
  
    if (!filesrc || !queue || !parsebin) {
        g_print("One of the elements not found\n");
    } else {
        /* Stop and remove old elements */
        gst_element_set_state(parsebin, GST_STATE_NULL);
        gst_element_set_state(queue, GST_STATE_NULL);
        gst_element_set_state(filesrc, GST_STATE_NULL);
  
        gst_bin_remove(GST_BIN(dynamicsrcbin), filesrc);
        gst_bin_remove(GST_BIN(dynamicsrcbin), queue);
        gst_bin_remove(GST_BIN(dynamicsrcbin), parsebin);
        
        /* Get current file and ID from nvdsdynamicsrcbin */
        g_object_get(G_OBJECT(dynamicsrcbin), "current-id", &current_id, NULL);
        g_object_get(G_OBJECT(dynamicsrcbin), "current-file", &current_file, NULL);
        g_print("Current ID: %d, Current File: %s\n", current_id, current_file);
        
        if (current_file && current_id) {
            gboolean ret = load_file(dynamicsrcbin, current_file);
            g_free(current_file);
            return ret;
        }
  
        /* Schedule retry if file not ready */
        g_timeout_add(1000, try_load_file, dynamicsrcbin);
    }
    return TRUE;
}

/**
 * @brief Bus callback to handle GStreamer messages
 * 
 * Handles various GStreamer bus messages including:
 * - EOS (End of Stream)
 * - Warnings and Errors
 * - Application messages for dynamic source changes
 * - Latency messages
 * 
 * @param bus The GStreamer bus
 * @param msg The message to handle
 * @param data Pointer to AppData structure
 * @return TRUE to continue receiving messages
 */
static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) { 
    AppData *app = (AppData *)data;
    
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS: {
            g_print("End of stream\n");
            g_main_loop_quit(app->loop);
            break;
        }
        case GST_MESSAGE_WARNING: {
            gchar *debug = NULL;
            GError *error = NULL;
            gst_message_parse_warning(msg, &error, &debug);
            g_printerr("WARNING from element %s: %s\n",
                GST_OBJECT_NAME(msg->src), error->message);
            g_free(debug);
            g_printerr("Warning: %s\n", error->message);
            g_error_free(error);
            break;
        }
        case GST_MESSAGE_ERROR: {
            gchar *debug = NULL;
            GError *error = NULL;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("ERROR from element %s: %s\n",
                GST_OBJECT_NAME(msg->src), error->message);
            if (debug)
                g_printerr("Error details: %s\n", debug);
            g_free(debug);
            g_error_free(error);
            g_main_loop_quit(app->loop);
            break;
        }
        case GST_MESSAGE_APPLICATION: {
            /* Handle dynamic source change messages from nvdsdynamicsrcbin */
            const GstStructure *str = gst_message_get_structure(msg);
            if (gst_structure_has_name(str, "dynamic-src-bin-file-change")) {
                GstElement *dynamicsrcbin = get_element_by_factory_name(GST_BIN(app->pipeline), "nvdsdynamicsrcbin");
                if (!dynamicsrcbin) {
                    g_printerr("Failed to get dynamicsrcbin\n");
                    return FALSE;
                }
                handle_dynamic_source_message(dynamicsrcbin);
                gst_object_unref(dynamicsrcbin);
            }
            break;
        }
        case GST_MESSAGE_LATENCY: {
            /* Recalculate latency when receiving latency message */
            gst_bin_recalculate_latency(GST_BIN(app->pipeline));
            break;
        }
        default:
            break;
    }
    return TRUE;
}

/**
 * @brief Test function to add and immediately remove a source
 * 
 * Demonstrates how to add a source and then remove it before completion.
 * 
 * @param data Pointer to AppData structure
 * @param source_id The source ID to add and remove
 */
void add_remove_source_test(void *data, gint source_id) {
    AppData *app = (AppData *)data;
    g_signal_emit_by_name(app->src, "add-source", VIDEO_FILE, source_id);
    g_print("Thread: Emitted signal for ID: %u, File: %s\n", source_id, VIDEO_FILE);
    g_usleep(500000); /* 0.5 second delay */
    g_signal_emit_by_name(app->src, "remove-source", source_id);
    g_print("Thread: Emitted signal for ID: %u, File: %s\n", source_id, VIDEO_FILE);
    g_usleep(500000); /* 0.5 second delay */
}

/**
 * @brief Test function to add a source
 * 
 * @param data Pointer to AppData structure
 * @param source_id The source ID to add
 */
void add_source_test(void *data, gint source_id) {
    AppData *app = (AppData *)data;
    g_signal_emit_by_name(app->src, "add-source", VIDEO_FILE, source_id);
    g_print("Thread: Emitted signal for ID: %u, File: %s\n", source_id, VIDEO_FILE);
}

/**
 * @brief Thread function to send dynamic source signals
 * 
 * This thread demonstrates how to dynamically add sources to the nvdsdynamicsrcbin
 * element. It adds sources sequentially and then terminates the pipeline.
 * 
 * @param data Pointer to AppData structure
 * @return NULL
 */
static void* signal_thread_func(void *data) {
    AppData *app = (AppData *)data;
    
    /* Wait for the pipeline to be ready */
    g_usleep(1000000); /* 1 second delay */
    
    /* Add sources sequentially */
    for (gint source_id = 0; source_id < N_CHUNKS; source_id++) {
        add_source_test(app, source_id);
        /* Uncomment the line below to see how to remove source before it completes */
        /* add_remove_source_test(app, source_id); */
    }

    g_usleep(10000000); /* 10 second delay */
    /* Terminate the pipeline - for demonstration purposes only */
    /* Use when you don't need to add more sources and don't want to wait for completion */
    g_signal_emit_by_name(app->src, "terminate"); 

    g_print("Signal thread completed\n");
    app->thread_running = FALSE;
    return NULL;
}

/**
 * @brief Main function
 * 
 * Sets up the GStreamer pipeline with nvdsdynamicsrcbin and fakesink,
 * creates a signal thread for dynamic source management, and runs the main loop.
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return 0 on success, -1 on failure
 */
int main(int argc, char *argv[]) {
    AppData app;
    guint bus_watch_id;
    GstBus *bus;

    /* Initialize GStreamer */
    gst_init(NULL, NULL);
    app.loop = g_main_loop_new(NULL, FALSE);

    /* Create pipeline elements */
    app.pipeline = gst_pipeline_new("dynamicsrcbin-test");
    app.src = gst_element_factory_make("nvdsdynamicsrcbin", "src");
    app.sink = gst_element_factory_make("fakesink", "sink");

    if (!app.src || !app.sink) {
        g_printerr("Failed to create elements\n");
        return -1;
    }

    /* Configure sink element properties */
    g_object_set(app.sink, "sync", 0, NULL);
    g_object_set(app.sink, "qos", 0, NULL);

    /* Set up bus monitoring */
    bus = gst_pipeline_get_bus(GST_PIPELINE(app.pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, &app);

    /* Add elements to pipeline and link them */
    gst_bin_add_many(GST_BIN(app.pipeline), app.src, app.sink, NULL);

    if (!gst_element_link_many(app.src, app.sink, NULL)) {
        g_printerr("Failed to link nvdsdynamicsrcbin to fakesink\n");
        return -1;
    }

    /* Add probe to monitor frame processing */
    GstPad *sink_pad = gst_element_get_static_pad(app.sink, "sink");
    gst_pad_add_probe(sink_pad,
        GST_PAD_PROBE_TYPE_BUFFER | GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
        (GstPadProbeCallback)sink_input_buffer_callback,
        NULL,
        NULL);
    gst_object_unref(sink_pad);

    /* Set pipeline to PLAYING state */
    gst_element_set_state(app.pipeline, GST_STATE_PLAYING);

    /* Create and start the signal thread */
    app.thread_running = TRUE;
    if (pthread_create(&app.signal_thread, NULL, signal_thread_func, &app) != 0) {
        g_printerr("Failed to create signal thread\n");
        return -1;
    }
    g_print("Signal thread created and started\n");

    /* Add initial sources to the nvdsdynamicsrcbin */
    for (gint source_id = 0; source_id < N_CHUNKS; source_id++) {
        g_signal_emit_by_name(app.src, "add-source", VIDEO_FILE, source_id);
        g_print("Current ID: %u\n", source_id);
    }

    g_print("Running main loop...\n");
    g_main_loop_run(app.loop);

    /* Wait for signal thread to complete if it's still running */
    if (app.thread_running) {
        g_print("Waiting for signal thread to complete...\n");
        pthread_join(app.signal_thread, NULL);
    }

    /* Cleanup */
    g_print("Stopping playback\n");
    gst_element_set_state(app.pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(app.pipeline));
    gst_object_unref(bus);
    g_source_remove(bus_watch_id);
    g_main_loop_unref(app.loop);

    return 0;
}