/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <iostream>
#include <fstream>
#include <thread>
#include <string.h>
#include <queue>
#include <mutex>
#include <stdexcept>
#include <string>
#include <condition_variable>
#include <sstream>
#include <map>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gst-nvquery.h"
#include "gstnvdsmeta.h"
#include "gst-nvevent.h"

#include "nvdscustomlib_base.hpp"
#include <gstnvdsinfer.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "yaml_parser.h"
#include "data_conversion.h"

using namespace std;
using std::string;

#define FORMAT_NV12 "NV12"
#define FORMAT_RGBA "RGBA"
#define HARDWARE_ACCELERATION 1

/* Strcture used to share between the threads */
struct PacketInfo {
  GstBuffer *inbuf;
  guint frame_num;
};

class Algorithm : public DSCustomLibraryBase
{
public:
  Algorithm() {
    outputthread_stopped = false;
    m_cfgParams.m_tensor_width = 640;
    m_cfgParams.m_tensor_height = 360;
    m_pYbuf_cuda = NULL;
  }

  /* Set Init Parameters */
  virtual bool SetInitParams(DSCustom_CreateParams *params);

  /* Set Custom Properties  of the library */
  virtual bool SetProperty(Property &prop);

  /* Pass GST events to the library */
  virtual bool HandleEvent(GstEvent *event);

  virtual char *QueryProperties ();

  /* Process Incoming Buffer */
  virtual BufferResult ProcessBuffer(GstBuffer *inbuf);

  /* Retrun Compatible Caps */
  virtual GstCaps * GetCompatibleCaps (GstPadDirection direction,
        GstCaps* in_caps, GstCaps* othercaps);

  /* Deinit members */
  ~Algorithm();

private:
  /* Output Processing Thread, push buffer to downstream  */
  void OutputThread(void);

public:
  guint source_id = 0;
  guint m_frameNum = 0;
  bool outputthread_stopped = false;

  /* Output Thread Pointer */
  std::thread *m_outputThread = NULL;

  /* Queue and Lock Management */
  std::queue<PacketInfo> m_processQ;
  std::mutex m_processLock;
  std::condition_variable m_processCV;

  /* Aysnc Stop Handling */
  gboolean m_stop = FALSE;
  /*sr tensor width*/
  int m_tensor_width;
  /*sr tensor height*/
  int m_tensor_height;
  std::string m_config_file_path;
  cfg_params m_cfgParams;
  unsigned char* m_pYbuf_cuda;
};

// Create Custom Algorithm / Library Context
extern "C" IDSCustomLibrary *CreateCustomAlgoCtx(DSCustom_CreateParams *params)
{
  GST_DEBUG(" %d %s", __LINE__, __func__);
  return new Algorithm();
}

// Set Init Parameters
bool Algorithm::SetInitParams(DSCustom_CreateParams *params)
{
  DSCustomLibraryBase::SetInitParams(params);
  m_outputThread = new std::thread(&Algorithm::OutputThread, this);
  GST_DEBUG(" %d %s", __LINE__, __func__);

  return true;
}

// Return Compatible Output Caps based on input caps
GstCaps* Algorithm::GetCompatibleCaps (GstPadDirection direction,
        GstCaps* in_caps, GstCaps* othercaps)
{
  GstCaps* result = NULL;
  GstStructure *s1, *s2;
  gint width, height;
  gint i, num, denom;
  const gchar *inputFmt = NULL;

  printf ("\n----------\ndirection = %d (1=Src, 2=Sink) -> %s:\nCAPS ="
  " %s\n", direction, __func__, gst_caps_to_string(in_caps));
  printf ("%s : OTHERCAPS = %s\n", __func__, gst_caps_to_string(othercaps));

  othercaps = gst_caps_truncate(othercaps);
  othercaps = gst_caps_make_writable(othercaps);

  int num_output_caps = gst_caps_get_size (othercaps);
  printf("num_output_caps:%d\n", num_output_caps);
  num_output_caps = gst_caps_get_size (in_caps);
  printf("in_caps, num_output_caps:%d\n", num_output_caps);

  // TODO: Currently it only takes first caps
  s1 = gst_caps_get_structure(in_caps, 0);
  for (i=0; i<num_output_caps; i++)
  {
    s2 = gst_caps_get_structure(othercaps, i);
    inputFmt = gst_structure_get_string (s1, "format");

    printf ("InputFMT = %s \n\n", inputFmt);

    // Check for desired color format
    if ((strncmp(inputFmt, FORMAT_NV12, strlen(FORMAT_NV12)) == 0) ||
            (strncmp(inputFmt, FORMAT_RGBA, strlen(FORMAT_RGBA)) == 0))
    {
      //Set these output caps
      gst_structure_get_int (s1, "width", &width);
      gst_structure_get_int (s1, "height", &height);

      /* otherwise the dimension of the output heatmap needs to be fixated */

      // Here change the width and height on output caps based on the
      // information provided byt the custom library
      gst_structure_fixate_field_nearest_int(s2, "width", width);
      gst_structure_fixate_field_nearest_int(s2, "height", height);
      if (gst_structure_get_fraction(s1, "framerate", &num, &denom))
      {
        gst_structure_fixate_field_nearest_fraction(s2, "framerate", num,
          denom);
      }

      gst_structure_set (s2, "width", G_TYPE_INT, (gint)(width), NULL);
      gst_structure_set (s2, "height", G_TYPE_INT, (gint)(height) , NULL);
      gst_structure_set (s2, "format", G_TYPE_STRING, inputFmt, NULL);
      GstCapsFeatures *feature = gst_caps_features_new ("memory:NVMM", NULL);
      gst_caps_set_features (othercaps, 0, feature);

      result = gst_caps_ref(othercaps);
      gst_caps_unref(othercaps);
      printf ("%s : Updated OTHERCAPS = %s \n\n", __func__,
        gst_caps_to_string(othercaps));

      break;
    } else {
      continue;
    }
  }
  return result;
}

char *Algorithm::QueryProperties ()
{
    char *str = new char[1000];
    strcpy (str, "EMOTION LIBRARY PROPERTIES\n \t\t\tcustomlib-props=\"config-file\" : path of the model config file");
    return str;
}

bool Algorithm::HandleEvent (GstEvent *event)
{
  switch (GST_EVENT_TYPE(event))
  {
    case GST_EVENT_EOS:
        m_processLock.lock();
        m_stop = TRUE;
        m_processCV.notify_all();
        m_processLock.unlock();
        while (outputthread_stopped == FALSE)
        {
            //g_print ("waiting for processq to be empty, buffers in processq
            // = %ld\n", m_processQ.size());
            g_usleep (1000);
        }
        break;
    default:
        break;
  }
  if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_STREAM_EOS)
  {
      gst_nvevent_parse_stream_eos (event, &source_id);
  }
  return true;
}

// Set Custom Library Specific Properties
bool Algorithm::SetProperty(Property &prop)
{
  try
  {
    if (prop.key.compare("config-file") == 0) {
      m_config_file_path.assign(prop.value);
      printf("m_config_file_path:%s\n", m_config_file_path.c_str());
      if (!gst_parse_context_params_yaml (m_config_file_path.c_str(), m_cfgParams)) {
        GST_ERROR("parse new model config file: %s failed.", m_config_file_path.c_str());
      }
      printf("m_tensor_width:%d, m_tensor_height:%d\n",
        m_cfgParams.m_tensor_width, m_cfgParams.m_tensor_height);
      cudaMalloc((void**)&m_pYbuf_cuda, m_cfgParams.m_tensor_width * m_cfgParams.m_tensor_height);
    }
  }
  catch(std::invalid_argument& e)
  {
      std::cout << "Invalid engine file path" << std::endl;
      return false;
  }

  GST_DEBUG(" %d %s", __LINE__, __func__);
  return true;
}

/* Deinitialize the Custom Lib context */
Algorithm::~Algorithm()
{
  GST_DEBUG(" %d %s", __LINE__, __func__);
  std::unique_lock<std::mutex> lk(m_processLock);
  m_processCV.wait(lk, [&]{return m_processQ.empty();});
  m_stop = TRUE;
  m_processCV.notify_all();
  lk.unlock();

  /* Wait for OutputThread to complete */
  if (m_outputThread) {
    m_outputThread->join();
  }
  cudaFree(m_pYbuf_cuda);
}

/* Process Buffer */
BufferResult Algorithm::ProcessBuffer (GstBuffer *inbuf)
{
  GstMapInfo in_map_info;

  GST_DEBUG ("CustomLib: ---> Inside %s frame_num = %d\n", __func__,
  m_frameNum++);

  // Push buffer to process thread for further processing
  PacketInfo packetInfo;
  packetInfo.inbuf = inbuf;
  packetInfo.frame_num = m_frameNum;

  // Add custom preprocessing logic if required, here
  // Pass the buffer to output_loop for further processing and pusing to next component
  // Currently its just dumping few decoded video frames

  m_processLock.lock();
  m_processQ.push(packetInfo);
  m_processCV.notify_all();
  m_processLock.unlock();

  return BufferResult::Buffer_Async;
}

void PostProcess_cuda(NvDsInferTensorMeta *meta, unsigned char* pY) {
  for (unsigned int i = 0; i < meta->num_output_layers; i++) {
    NvDsInferLayerInfo *info = &meta->output_layers_info[i];
    if (meta->out_buf_ptrs_dev[i]) {
      int h = info->inferDims.d[1];
      int w = info->inferDims.d[2];
      Convert_FtFTensor((float*)meta->out_buf_ptrs_dev[i], pY, w, h);
      cudaDeviceSynchronize();
    }
  }
}

/* replace lumin part */
void replace_Y(unsigned char* pY, NvBufSurfaceParams *surParam) {
  unsigned char* pSrc = pY;
  int height = surParam->height;
  int width = surParam->width;
  int pitch = surParam->pitch;
  unsigned char * dataPtr = (unsigned char *)surParam->dataPtr;
  for (int i = 0; i < height; i++) {
    cudaMemcpy (dataPtr, pSrc, width, cudaMemcpyDeviceToDevice);
    pSrc += width;
    dataPtr += pitch;
  }
}

/* Output Processing Thread */
void Algorithm::OutputThread(void)
{
  GstFlowReturn flow_ret;
  GstBuffer *outBuffer = NULL;
  NvBufSurface *outSurf = NULL;
  NvDsBatchMeta *batch_meta = NULL;
  GstMapInfo in_map_info;
  std::unique_lock<std::mutex> lk(m_processLock);
  printf("in OutputThread\n");
  while(1){
    /* Wait if processing queue is empty. */
    if (m_processQ.empty()) {
      if (m_stop == TRUE) {
        break;
      }
      m_processCV.wait(lk);
      continue;
    }

    PacketInfo packetInfo = m_processQ.front();
    m_processQ.pop();

    m_processCV.notify_all();
    lk.unlock();

    NvBufSurface *in_surf = getNvBufSurface (packetInfo.inbuf);
    batch_meta = gst_buffer_get_nvds_batch_meta (packetInfo.inbuf);

    NvDsMetaList * l_frame = NULL;
    nvds_acquire_meta_lock (batch_meta);
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
      for (NvDsMetaList * l_user = frame_meta->frame_user_meta_list;
        l_user != NULL; l_user = l_user->next) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
        if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
          continue;
        NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;
        PostProcess_cuda(meta, m_pYbuf_cuda);
        NvBufSurfaceParams *surParam = &(in_surf->surfaceList[frame_meta->batch_id]);
        if (surParam->colorFormat == NVBUF_COLOR_FORMAT_NV12 ||
          surParam->colorFormat == NVBUF_COLOR_FORMAT_NV12_709 ) {
          if(in_surf->memType == NVBUF_MEM_CUDA_DEVICE)
            replace_Y(m_pYbuf_cuda, surParam);
        }
      }
    }

    nvds_release_meta_lock (batch_meta);

    // Transform IP case
    outSurf = in_surf;
    outBuffer = packetInfo.inbuf;

    // Output buffer parameters checking
    if (outSurf->numFilled != 0)
    {
      g_assert ((guint)m_outVideoInfo.width == outSurf->surfaceList->width);
      g_assert ((guint)m_outVideoInfo.height == outSurf->surfaceList->height);
    }

    flow_ret = gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (m_element),
      outBuffer);
    GST_DEBUG ("CustomLib: %s in_surf=%p, Pushing Frame %d to downstream..."
      " flow_ret = %d TS=%" GST_TIME_FORMAT " \n", __func__, in_surf,
      packetInfo.frame_num, flow_ret,
      GST_TIME_ARGS(GST_BUFFER_PTS(outBuffer)));

    lk.lock();
  }
  outputthread_stopped = true;
  printf("exit OutputThread\n");
}
