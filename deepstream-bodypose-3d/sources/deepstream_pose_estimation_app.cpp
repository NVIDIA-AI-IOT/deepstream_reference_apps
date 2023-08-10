/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <signal.h>
#include <bits/stdc++.h>

#include "cuda_runtime_api.h"
#include "gstnvdsinfer.h"
#include "gstnvdsmeta.h"
#include "nvdsgstutils.h"
#include "nvbufsurface.h"
#include "nvdsmeta_schema.h"
#include "deepstream_common.h"
#include "deepstream_perf.h"
#include "deepstream_app_version.h"
#include <sys/time.h>
#include <vector>
#include <array>
#include <queue>
#include <cmath>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>
#include <ctime>

GST_DEBUG_CATEGORY_STATIC (NVDS_APP);  // define category (statically)
#define GST_CAT_DEFAULT NVDS_APP       // set as default

#define EPS 1e-6
#define MAX_TIME_STAMP_LEN 32
#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2
#define PGIE_CLASS_ID_PRODUCT 4

// Default camera attributes
#define MUXER_OUTPUT_WIDTH 1280
#define MUXER_OUTPUT_HEIGHT 720
#define FOCAL_LENGTH 800.79041f

/* Padding due to AR SDK model requires bigger bboxes*/
#define PAD_DIM 128

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"
#define CONFIG_GPU_ID "gpu-id"

#define PGIE_CONFIG_FILE "../configs/config_infer_primary_peoplenet.txt"
#define SGIE_CONFIG_FILE "../configs/config_infer_secondary_bodypose3dnet.txt"
#define TRACKER_CONFIG_FILE "../configs/config_tracker.txt"
#define CONFIG_GROUP_TRACKER "tracker"
#define CONFIG_GROUP_TRACKER_WIDTH "tracker-width"
#define CONFIG_GROUP_TRACKER_HEIGHT "tracker-height"
#define CONFIG_GROUP_TRACKER_LL_CONFIG_FILE "ll-config-file"
#define CONFIG_GROUP_TRACKER_LL_LIB_FILE "ll-lib-file"
#define CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS "enable-batch-process"
#define MAX_TRACKING_ID_LEN 16

#define CHECK_ERROR(error) \
    if (error) { \
        g_printerr ("Error while parsing config file: %s\n", error->message); \
        goto done; \
    }

//---Global variables derived from program arguments---
static gboolean _print_version = FALSE;
static gboolean _print_dependencies_version = FALSE;
static gboolean _print_fps = FALSE;
static gchar *_input = NULL;
static gchar *_output = NULL;
static gchar *_nvmsgbroker_conn_str = NULL;
static gchar *_pose_filename = NULL;
static gchar *_tracker = NULL;
static gchar *_publish_pose = NULL;
static guint _cintr = FALSE;
static gboolean _quit = FALSE;
FILE *_pose_file = NULL;
double _focal_length_dbl = FOCAL_LENGTH;
float _focal_length = (float)_focal_length_dbl;
int _image_width = MUXER_OUTPUT_WIDTH;
int _image_height = MUXER_OUTPUT_HEIGHT;
int _pad_dim = PAD_DIM;// A scaled version of PAD_DIM
Eigen::Matrix3f _K;// Camera intrinsic matrix
//---Global variables derived from program arguments---

static GstElement *pipeline = NULL;
static gint _fps_interval=1;
static gint _osd_process_mode = 0;

gint frame_number = 0;

#if (DS_VERSION_MAJOR < 6) || ((DS_VERSION_MAJOR == 6) && (DS_VERSION_MINOR < 2))
typedef struct NvDsJoint {
  gdouble confidence;
  gdouble x;
  gdouble y;
  gdouble z;
}NvDsJoint;

typedef struct NvDsJoints {

  gint pose_type;
  gint num_joints;
  NvDsJoint *joints;

}NvDsJoints;
#endif

typedef struct NvDsPersonPoseExt {
  gint num_poses;
  NvDsJoints *poses;
}NvDsPersonPoseExt;


class OneEuroFilter {
public:
  /// Default constructor
  OneEuroFilter() {
    reset(30.0f /* Hz */, 0.1f /* Hz */, 0.09f /* ??? */, 0.5f /* Hz */);
  }
  /// Constructor
  /// @param dataUpdateRate   the sampling rate, i.e. the number of samples per unit of time.
  /// @param minCutoffFreq    the lowest bandwidth filter applied.
  /// @param cutoffSlope      the rate at which the filter adapts: higher levels reduce lag.
  /// @param derivCutoffFreq  the bandwidth of the filter applied to smooth the derivative, default 1 Hz.
  OneEuroFilter(float dataUpdateRate, float minCutoffFreq, float cutoffSlope, float derivCutoffFreq) {
    reset(dataUpdateRate, minCutoffFreq, cutoffSlope, derivCutoffFreq);
  }
  /// Reset all parameters of the filter.
  /// @param dataUpdateRate   the sampling rate, i.e. the number of samples per unit of time.
  /// @param minCutoffFreq    the lowest bandwidth filter applied.
  /// @param cutoffSlope      the rate at which the filter adapts: higher levels reduce lag.
  /// @param derivCutoffFreq  the bandwidth of the filter applied to smooth the derivative, default 1 Hz.
  void reset(float dataUpdateRate, float minCutoffFreq, float cutoffSlope, float derivCutoffFreq) {
    reset(); _rate = dataUpdateRate; _minCutoff = minCutoffFreq; _beta = cutoffSlope; _dCutoff = derivCutoffFreq;
  }
  /// Reset only the initial condition of the filter, leaving parameters the same.
  void reset() { _firstTime = true; _xFilt.reset(); _dxFilt.reset(); }
  /// Apply the one euro filter to the given input.
  /// @param x  the unfiltered input value.
  /// @return   the filtered output value.
  float filter(float x)
  {
    float dx, edx, cutoff;
    if (_firstTime) {
      _firstTime = false;
      dx = 0;
    } else {
      dx = (x - _xFilt.hatXPrev()) * _rate;
    }
    edx = _dxFilt.filter(dx, alpha(_rate, _dCutoff));
    cutoff = _minCutoff + _beta * fabsf(edx);
    return _xFilt.filter(x, alpha(_rate, cutoff));
  }


private:
  class LowPassFilter {
  public:
    LowPassFilter() { reset(); }
    void reset() { _firstTime = true; }
    float hatXPrev() const { return _hatXPrev; }
    float filter(float x, float alpha){
      if (_firstTime) {
        _firstTime = false;
        _hatXPrev = x;
      }
      float hatX = alpha * x + (1.f - alpha) * _hatXPrev;
      _hatXPrev = hatX;
      return hatX;

    }
  private:
    float _hatXPrev;
    bool _firstTime;
  };
  inline float alpha(float rate, float cutoff) {
  const float kOneOverTwoPi = 0.15915494309189533577f;  // 1 / (2 * pi)
  // The paper has 4 divisions, but we only use one
  // float tau = kOneOverTwoPi / cutoff, te = 1.f / rate;
  // return 1.f / (1.f + tau / te);
  return cutoff / (rate * kOneOverTwoPi + cutoff);
}
  bool _firstTime;
  float _rate, _minCutoff, _dCutoff, _beta;
  LowPassFilter _xFilt, _dxFilt;
};

//===Global variables===
std::unordered_map<gint , std::vector<OneEuroFilter>> g_filter_pose25d;
OneEuroFilter m_filterRootDepth; // Root node in pose25d.

fpos_t g_fp_25_pos;

//===Global variables===

#define ACQUIRE_DISP_META(dmeta)  \
  if (dmeta->num_circles == MAX_ELEMENTS_IN_DISPLAY_META  || \
      dmeta->num_labels == MAX_ELEMENTS_IN_DISPLAY_META ||  \
      dmeta->num_lines == MAX_ELEMENTS_IN_DISPLAY_META) \
        { \
          dmeta = nvds_acquire_display_meta_from_pool(bmeta);\
          nvds_add_display_meta_to_frame(frame_meta, dmeta);\
        }\

#define GET_LINE(lparams) \
        ACQUIRE_DISP_META(dmeta)\
        lparams = &dmeta->line_params[dmeta->num_lines];\
        dmeta->num_lines++;\

static float _sgie_classifier_threshold = FLT_MIN;

typedef struct NvAR_Point3f {
  float x, y, z;
} NvAR_Point3f;

static Eigen::Matrix3f m_K_inv_transpose;
const float m_scale_ll[] = {
  0.5000, 0.5000, 1.0000, 0.8175, 0.9889, 0.2610, 0.7942, 0.5724, 0.5078,
  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3433, 0.8171,
  0.9912, 0.2610, 0.8259, 0.5724, 0.5078, 0.0000, 0.0000, 0.0000, 0.0000,
  0.0000, 0.0000, 0.0000, 0.3422, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000};
const float m_mean_ll[] = {
  246.3427f, 246.3427f, 492.6854f, 402.4380f, 487.0321f, 128.6856f, 391.6295f,
  281.9928f, 249.9478f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,
    0.0000f,   0.0000f, 169.1832f, 402.2611f, 488.1824f, 128.6848f, 407.5836f,
  281.9897f, 249.9489f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,
    0.0000f,   0.0000f, 168.6137f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,
    0.0000f};

/* Given 2D and ZRel, we need to find the depth of the root to reconstruct the scale normalized 3D Pose.
   While there exists many 3D poses that can have the same 2D projection, given the 2.5D pose and intrinsic camera parameters,
   there exists a unique 3D pose that satisfies (Xˆn − Xˆm)**2 + (Yˆn − Yˆm)**2 + (Zˆn − Zˆm)**2 = C**2.
   Refer Section 3.3 of https://arxiv.org/pdf/1804.09534.pdf for more details.
*/
std::vector<float> calculateZRoots(const std::vector<float>& X0, const std::vector<float>& X1,
    const std::vector<float>& Y0, const std::vector<float>& Y1,
    const std::vector<float>& Zrel0,
    const std::vector<float>& Zrel1, const std::vector<float>& C) {
    std::vector<float> zRoots(X0.size());
    for (int i = 0; i < X0.size(); i++) {
        double x0 = (double)X0[i], x1 = (double)X1[i], y0 = (double)Y0[i], y1 = (double)Y1[i],
            z0 = (double)Zrel0[i], z1 = (double)Zrel1[i];
        double a = ((x1 - x0) * (x1 - x0)) + ((y1 - y0) * (y1 - y0));
        double b = 2 * (z1 * ((x1 * x1) + (y1 * y1) - x1 * x0 - y1 * y0) +
                    z0 * ((x0 * x0) + (y0 * y0) - x1 * x0 - y1 * y0));
        double c = ((x1 * z1 - x0 * z0) * (x1 * z1 - x0 * z0)) +
                   ((y1 * z1 - y0 * z0) * (y1 * z1 - y0 * z0)) +
                   ((z1 - z0) * (z1 - z0)) - (C[i] * C[i]);
        double d = (b * b) - (4 * a * c);

        // make sure the solutions are valid
        a = fmax(DBL_EPSILON, a);
        d = fmax(DBL_EPSILON, d);
        zRoots[i] = (float) ((-b + sqrt(d)) / (2 * a + 1e-8));
    }
    return zRoots;
}

float median(std::vector<float>& v) {
    size_t n = v.size() / 2;
    nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}

/* Given 2D keypoints and the relative depth of each keypoint w.r.t the root, we find the depth of the root
   to reconstruct the scale normalized 3D pose.
*/
std::vector<NvAR_Point3f> liftKeypoints25DTo3D(const float* p2d,
    const float* pZRel,
    const int numKeypoints,
    const Eigen::Matrix3f& KInv,
    const float limbLengths[]) {

    const int ROOT = 0;

    // Contains the relative depth values of each keypoints
    std::vector<float> zRel(numKeypoints, 0.f);

    // Matrix containing the 2D keypoints.
    Eigen::MatrixXf XY1 = Eigen::MatrixXf(numKeypoints, 3);

    // Mean distance between a specific pair and its parent.
    std::vector<float> C;

    // Indices representing keypoints and its parents for limb lengths > 0.
    // In our dataset, we only have limb length information for few keypoints.
    std::vector<int> idx0 = { 0, 3, 6, 8, 5, 2, 2, 21, 23, 21, 7, 4, 1, 1, 20, 22, 20 };
    std::vector<int> idx1 = { 3, 6, 0, 5, 2, 0, 21, 23, 25, 6, 4, 1, 0, 20, 22, 24, 6 };

    std::vector<float> X0(idx0.size(), 0.f), Y0(idx0.size(), 0.f), X1(idx0.size(), 0.f), Y1(idx0.size(), 0.f),
        zRel0(idx0.size(), 0.f), zRel1(idx0.size(), 0.f);

    for (int i = 0; i < numKeypoints; i++) {
        zRel[i] = pZRel[i];

        XY1.row(i) << p2d[i * 2], p2d[(i * 2) + 1], 1.f;

        if (limbLengths[i] > 0.f) C.push_back(limbLengths[i]);
    }

    // Set relative depth of root to be 0 as the relative depth is measure w.r.t the root.
    zRel[ROOT] = 0.f;

    for (int i = 0; i < XY1.rows(); i++) {
        float x = XY1(i, 0);
        float y = XY1(i, 1);
        float z = XY1(i, 2);
        XY1.row(i) << x, y, z;
    }

    XY1 = XY1 * KInv;

    for (int i = 0; i < idx0.size(); i++) {
        X0[i] = XY1(idx0[i], 0);
        Y0[i] = XY1(idx0[i], 1);
        X1[i] = XY1(idx1[i], 0);
        Y1[i] = XY1(idx1[i], 1);
        zRel0[i] = zRel[idx0[i]];
        zRel1[i] = zRel[idx1[i]];
    }

    std::vector<float> zRoots = calculateZRoots(X0, X1, Y0, Y1, zRel0, zRel1, C);

    float zRootsMedian = median(zRoots);

    zRootsMedian = m_filterRootDepth.filter(zRootsMedian);

    std::vector<NvAR_Point3f> p3d(numKeypoints, { 0.f, 0.f, 0.f });

    for (int i = 0; i < numKeypoints; i++) {
        p3d[i].x = XY1(i, 0) * (zRel[i] + zRootsMedian);
        p3d[i].y = XY1(i, 1) * (zRel[i] + zRootsMedian);
        p3d[i].z = XY1(i, 2) * (zRel[i] + zRootsMedian);
    }

    return p3d;
}

/* Once we have obtained the scale normalized 3D pose, we use the mean limb lengths of keypoint-keypointParent pairs
* to find the scale of the whole body. We solve for
*  s^ = argmin sum((s * L2_norm(P_k - P_l) - meanLimbLength_k_l)**2), solve for s.
*  meanLimbLength_k_l = mean length of the bone between keypoints k and l in the training data
*  P_k and P_l are the keypoint location of k and l.
*
*  We have a least squares minimization, where we are trying to minimize the magnitude of the error:
     (target - scale * unit_length). Thus, we're minimizing T - sL. By the normal equations, the optimal solution is:
      s = inv([L'L]) * L'T

*/
float recoverScale(const std::vector<NvAR_Point3f>& p3d, const float* scores,
    const float targetLengths[]) {
    std::vector<int> validIdx;

    // Indices of keypoints for which we have the length information.
    for (int i = 0; i < p3d.size(); i++) {
        if (targetLengths[i] > 0.f) validIdx.push_back(i);
    }

    Eigen::MatrixXf targetLenMatrix = Eigen::MatrixXf(validIdx.size(), 1);

    for (int i = 0; i < validIdx.size(); i++) {
        targetLenMatrix(i, 0) = targetLengths[validIdx[i]];
    }

    // Indices representing keypoints and its parents for limb lengths > 0.
    // In our dataset, we have only have limb length information for few keypoints.
    std::vector<int> idx0 = { 0, 3, 6, 8, 5, 2, 2, 21, 23, 21, 7, 4, 1, 1, 20, 22, 20 };
    std::vector<int> idx1 = { 3, 6, 0, 5, 2, 0, 21, 23, 25, 6, 4, 1, 0, 20, 22, 24, 6 };

    Eigen::MatrixXf unitLength = Eigen::MatrixXf(idx0.size(), 1);
    Eigen::VectorXf limbScores(unitLength.size());
    float squareNorms = 0.f;
    float limbScoresSum = 0.f;
    for (int i = 0; i < idx0.size(); i++) {
        unitLength(i, 0) = sqrtf((p3d[idx0[i]].x - p3d[idx1[i]].x) * (p3d[idx0[i]].x - p3d[idx1[i]].x) +
            (p3d[idx0[i]].y - p3d[idx1[i]].y) * (p3d[idx0[i]].y - p3d[idx1[i]].y) +
            (p3d[idx0[i]].z - p3d[idx1[i]].z) * (p3d[idx0[i]].z - p3d[idx1[i]].z));

        limbScores[i] = scores[idx0[i]] * scores[idx1[i]];
        limbScoresSum += limbScores[i];
    }

    for (int i = 0; i < limbScores.size(); i++) {
        limbScores[i] /= limbScoresSum;
        squareNorms += ((unitLength(i, 0) * unitLength(i, 0)) * limbScores[i]);
    }

    auto limbScoreDiag = limbScores.asDiagonal();

    //Eigen::MatrixXf numerator1 = ;
    Eigen::MatrixXf numerator = (unitLength.transpose() * limbScoreDiag) * targetLenMatrix;

    return numerator(0, 0) / squareNorms;
}

void generate_ts_rfc3339 (char *buf, int buf_size)
{
  time_t tloc;
  struct tm tm_log;
  struct timespec ts;
  char strmsec[6]; //.nnnZ\0

  clock_gettime(CLOCK_REALTIME,  &ts);
  memcpy(&tloc, (void *)(&ts.tv_sec), sizeof(time_t));
  gmtime_r(&tloc, &tm_log);
  strftime(buf, buf_size,"%Y-%m-%dT%H:%M:%S", &tm_log);
  int ms = ts.tv_nsec/1000000;
  g_snprintf(strmsec, sizeof(strmsec),".%.3dZ", ms);
  strncat(buf, strmsec, buf_size);
}

static
gpointer copy_bodypose_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;
  NvDsEventMsgMeta *dstMeta = NULL;
  NvDsPersonObject *srcExt = NULL;
  NvDsPersonObject *dstExt = NULL;

  dstMeta = (NvDsEventMsgMeta *)g_memdup ((gpointer)srcMeta, sizeof(NvDsEventMsgMeta));

  // pose
  dstMeta->pose.num_joints = srcMeta->pose.num_joints;
  dstMeta->pose.pose_type = srcMeta->pose.pose_type;
  dstMeta->pose.joints = (NvDsJoint *)g_memdup ((gpointer)srcMeta->pose.joints,
                                    sizeof(NvDsJoint)*srcMeta->pose.num_joints);

  if (srcMeta->ts)
    dstMeta->ts = g_strdup (srcMeta->ts);

  if (srcMeta->sensorStr)
    dstMeta->sensorStr = g_strdup (srcMeta->sensorStr);

  if (srcMeta->objSignature.size > 0) {
    dstMeta->objSignature.signature = (gdouble *)g_memdup ((gpointer)srcMeta->objSignature.signature,
                                                sizeof(gdouble)*srcMeta->objSignature.size);
    dstMeta->objSignature.size = srcMeta->objSignature.size;
  }

  if(srcMeta->objectId) {
    dstMeta->objectId = g_strdup (srcMeta->objectId);
  }

  if (srcMeta->extMsg){
    dstMeta->extMsg = g_memdup(srcMeta->extMsg, srcMeta->extMsgSize);
    dstMeta->extMsgSize = srcMeta->extMsgSize;
    srcExt = (NvDsPersonObject *)srcMeta->extMsg;
    dstExt = (NvDsPersonObject *)dstMeta->extMsg;
    dstExt->gender = g_strdup(srcExt->gender);
    dstExt->hair = g_strdup(srcExt->hair);
    dstExt->cap = g_strdup(srcExt->cap);
    dstExt->apparel = g_strdup(srcExt->apparel);
    dstExt->age = srcExt->age;
  }
  return dstMeta;
}

static void
release_bodypose_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;

  // pose
  g_free (srcMeta->pose.joints);
  g_free (srcMeta->ts);
  g_free (srcMeta->sensorStr);

  if (srcMeta->objSignature.size > 0) {
    g_free (srcMeta->objSignature.signature);
    srcMeta->objSignature.size = 0;
  }

  if(srcMeta->objectId) {
    g_free (srcMeta->objectId);
  }

  g_free (srcMeta->extMsg);
  srcMeta->extMsgSize = 0;
  srcMeta->extMsg = NULL;

  g_free (user_meta->user_meta_data);
  user_meta->user_meta_data = NULL;
}

void build_msg_meta(NvDsFrameMeta *frame_meta,
      NvDsObjectMeta *obj_meta,
      const int numKeyPoints,
      const float *keypoints,
      const float *keypointsZRel,
      const float *keypoints_confidence,
      const std::vector<NvAR_Point3f> &p3dLifted)
{
  NvDsEventMsgMeta *msg_meta = (NvDsEventMsgMeta *) g_malloc0 (sizeof (NvDsEventMsgMeta));
  NvDsPersonObject *msg_meta_ext = (NvDsPersonObject *) g_malloc0 (sizeof (NvDsPersonObject));

  msg_meta->type = NVDS_EVENT_ENTRY; //Should this be ENTRY
  msg_meta->objType = (NvDsObjectType) NVDS_OBJECT_TYPE_PERSON;
  msg_meta->bbox.top = obj_meta->rect_params.top;
  msg_meta->bbox.left = obj_meta->rect_params.left;
  msg_meta->bbox.width = obj_meta->rect_params.width;
  msg_meta->bbox.height = obj_meta->rect_params.height;
  msg_meta->extMsg = msg_meta_ext;
  msg_meta->extMsgSize =  sizeof(NvDsPersonObject);
  msg_meta_ext->gender  = g_strdup("");
  msg_meta_ext->hair    = g_strdup("");
  msg_meta_ext->cap     = g_strdup("");
  msg_meta_ext->apparel = g_strdup("");
  msg_meta_ext->age = 0;

  //---msg_meta->poses---
  if (1) {
    int pose_types[8];

    if (_publish_pose) {
      if (!strcmp(_publish_pose, "pose3d"))
        msg_meta->pose.pose_type = 2;// pdatapose3D
      else if (!strcmp(_publish_pose, "pose25d"))
        msg_meta->pose.pose_type = 1;// pdatapose3D
    }
    else {
      msg_meta->pose.pose_type = 2;// pdatapose3D
    }

    msg_meta->pose.num_joints = numKeyPoints;
    msg_meta->pose.joints = (NvDsJoint *)g_malloc0(sizeof(NvDsJoint) * numKeyPoints);

    if (msg_meta->pose.pose_type == 0) {// pose25d without zRel
      for(int i = 0; i < msg_meta->pose.num_joints; i++){
        msg_meta->pose.joints[i].x = keypoints[2*i  ];
        msg_meta->pose.joints[i].y = keypoints[2*i+1];
        msg_meta->pose.joints[i].confidence = keypoints_confidence[i];
      }
    }
    else if (msg_meta->pose.pose_type == 2) {// pose3d
      for(int i = 0; i < msg_meta->pose.num_joints; i++){
        msg_meta->pose.joints[i].x = p3dLifted[i].x;
        msg_meta->pose.joints[i].y = p3dLifted[i].y;
        msg_meta->pose.joints[i].z = p3dLifted[i].z;
        msg_meta->pose.joints[i].confidence = keypoints_confidence[i];
      }
    }
    else if (msg_meta->pose.pose_type == 1) {// pose25d
      for(int i = 0; i < msg_meta->pose.num_joints; i++){
        msg_meta->pose.joints[i].x = keypoints[2*i  ];
        msg_meta->pose.joints[i].y = keypoints[2*i+1];
        msg_meta->pose.joints[i].z = keypointsZRel[i];
        msg_meta->pose.joints[i].confidence = keypoints_confidence[i];
      }
    }
  }
  // // DEBUG
  // g_message("Metadata poses are built.\n");
  //---msg_meta->poses---
  // msg_meta->embedding =
  // msg_meta->location =
  // msg_meta->coordinate =
  // msg_meta->objSignature =
  //msg_meta->objClassId = PGIE_CLASS_ID_PERSON;
  msg_meta->objClassId = obj_meta->class_id;
  // msg_meta->sensorId =
  // msg_meta->moduleId =
  // msg_meta->placeId =
  // msg_meta->componentId =
  msg_meta->frameId = frame_meta->frame_num;
  msg_meta->confidence = obj_meta->confidence;
  msg_meta->trackingId = obj_meta->object_id;
  msg_meta->ts = (gchar *) g_malloc0 (MAX_TIME_STAMP_LEN + 1);
  generate_ts_rfc3339(msg_meta->ts, MAX_TIME_STAMP_LEN);
  msg_meta->objectId = (gchar *) g_malloc0 (MAX_LABEL_SIZE);
  strncpy(msg_meta->objectId, obj_meta->obj_label, MAX_LABEL_SIZE);
  // msg_meta->sensorStr =
  // msg_meta->otherAttr =
  // msg_meta->videoPath =
  // // DEBUG
  // g_message("Metadata is built.");

  NvDsBatchMeta *batch_meta = frame_meta->base_meta.batch_meta;
  NvDsUserMeta *user_event_meta = nvds_acquire_user_meta_from_pool (batch_meta);
  if (user_event_meta) {
    user_event_meta->user_meta_data = (void *) msg_meta;
    user_event_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
    user_event_meta->base_meta.copy_func = (NvDsMetaCopyFunc) copy_bodypose_meta;
    user_event_meta->base_meta.release_func = (NvDsMetaReleaseFunc)release_bodypose_meta;
    nvds_add_user_meta_to_frame(frame_meta, user_event_meta);
  } else {
    g_printerr("Error in attaching event meta to buffer\n");
  }
}

void osd_upper_body(NvDsFrameMeta* frame_meta,
      NvDsBatchMeta *bmeta,
      NvDsDisplayMeta *dmeta,
      const int numKeyPoints,
      const float keypoints[],
      const float keypoints_confidence[])
{
  const int keypoint_radius = 3 * _image_width / MUXER_OUTPUT_WIDTH;//6;//3;
  const int keypoint_line_width = 2 * _image_width / MUXER_OUTPUT_WIDTH;//4;//2;

  const int num_joints = 24;
  const int idx_joints[] = { 0,  1,  2,  3,  6, 15, 16, 17, 18, 19, 20, 21,
                            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33};
  const int num_bones = 25;
  const int idx_bones[] = { 21,  6, 20,  6, 21, 23, 20, 22, 24, 22, 23, 25,
                            27, 25, 31, 25, 33, 25, 29, 25, 24, 30, 24, 26,
                            24, 32, 24, 28,  2, 21,  1, 20,  3,  6,  6, 15,
                            15, 16, 15, 17, 19, 17, 18, 16,  0,  1,  0,  2,
                             0,  3};
  const NvOSD_ColorParams bone_colors[] = {
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1}};

  for (int ii = 0; ii < num_joints; ii++) {
    int i = idx_joints[ii];

    if (keypoints_confidence[i] < _sgie_classifier_threshold)
      continue;

    ACQUIRE_DISP_META(dmeta);
    NvOSD_CircleParams &cparams = dmeta->circle_params[dmeta->num_circles];
    cparams.xc = keypoints[2 * i    ];
    cparams.yc = keypoints[2 * i + 1];
    cparams.radius = keypoint_radius;
    cparams.circle_color =  NvOSD_ColorParams{0.96, 0.26, 0.21, 1};
    cparams.has_bg_color = 1;
    cparams.bg_color =  NvOSD_ColorParams{0.96, 0.26, 0.21, 1};
    dmeta->num_circles++;
  }

  for (int i = 0; i < num_bones; i++) {
    int i0 = idx_bones[2 * i    ];
    int i1 = idx_bones[2 * i + 1];

    if ((keypoints_confidence[i0] < _sgie_classifier_threshold) ||
        (keypoints_confidence[i1] < _sgie_classifier_threshold))
        continue;

    ACQUIRE_DISP_META(dmeta);
    NvOSD_LineParams *lparams = &dmeta->line_params[dmeta->num_lines];
    lparams->x1 = keypoints[2 * i0];
    lparams->y1 = keypoints[2 * i0 + 1];
    lparams->x2 = keypoints[2 * i1];
    lparams->y2 = keypoints[2 * i1 + 1];
    lparams->line_width = keypoint_line_width;
    lparams->line_color = bone_colors[i];
    dmeta->num_lines++;
  }

  return;
}

void osd_lower_body(NvDsFrameMeta* frame_meta,
      NvDsBatchMeta *bmeta,
      NvDsDisplayMeta *dmeta,
      const int numKeyPoints,
      const float keypoints[],
      const float keypoints_confidence[])
{
  const int keypoint_radius = 3 * _image_width / MUXER_OUTPUT_WIDTH;//6;//3;
  const int keypoint_line_width = 2 * _image_width / MUXER_OUTPUT_WIDTH;//4;//2;

  const int num_joints = 10;
  const int idx_joints[] = { 4,  5,  7,  8,  9, 10, 11, 12, 13, 14};
  const int num_bones = 10;
  const int idx_bones[] = {  2,  5,  5,  8,  1,  4,  4,  7,  7, 13,
                             8, 14,  8, 10,  7,  9, 11,  9, 12, 10};
  const NvOSD_ColorParams bone_colors[] = {
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1}};

  for (int ii = 0; ii < num_joints; ii++) {
    int i = idx_joints[ii];

    if (keypoints_confidence[i] < _sgie_classifier_threshold)
      continue;

    ACQUIRE_DISP_META(dmeta);
    NvOSD_CircleParams &cparams = dmeta->circle_params[dmeta->num_circles];
    cparams.xc = keypoints[2 * i    ];
    cparams.yc = keypoints[2 * i + 1];
    cparams.radius = keypoint_radius;
    cparams.circle_color = NvOSD_ColorParams{0.96, 0.26, 0.21, 1};
    cparams.has_bg_color = 1;
    cparams.bg_color = NvOSD_ColorParams{0.96, 0.26, 0.21, 1};
    dmeta->num_circles++;
  }

  for (int i = 0; i < num_bones; i++) {
    int i0 = idx_bones[2 * i    ];
    int i1 = idx_bones[2 * i + 1];

    if ((keypoints_confidence[i0] < _sgie_classifier_threshold) ||
        (keypoints_confidence[i1] < _sgie_classifier_threshold))
        continue;

    ACQUIRE_DISP_META(dmeta);
    NvOSD_LineParams *lparams = &dmeta->line_params[dmeta->num_lines];
    lparams->x1 = keypoints[2 * i0];
    lparams->y1 = keypoints[2 * i0 + 1];
    lparams->x2 = keypoints[2 * i1];
    lparams->y2 = keypoints[2 * i1 + 1];
    lparams->line_width = keypoint_line_width;
    lparams->line_color = bone_colors[i];
    dmeta->num_lines++;
  }

  return;
}

void parse_25dpose_from_tensor_meta(NvDsInferTensorMeta *tensor_meta,
      NvDsFrameMeta *frame_meta, NvDsObjectMeta *obj_meta)
{
  // const int pelvis = 0;
  // const int left_hip = 1;
  // const int right_hip = 2;
  // const int torso = 3;
  // const int left_knee = 4;
  // const int right_knee = 5;
  // const int neck = 6;
  // const int left_ankle = 7;
  // const int right_ankle = 8;
  // const int left_big_toe = 9;
  // const int right_big_toe = 10;
  // const int left_small_toe = 11;
  // const int right_small_toe = 12;
  // const int left_heel = 13;
  // const int right_heel = 14;
  // const int nose = 15;
  // const int left_eye = 16;
  // const int right_eye = 17;
  // const int left_ear = 18;
  // const int right_ear = 19;
  // const int left_shoulder = 20;
  // const int right_shoulder = 21;
  // const int left_elbow = 22;
  // const int right_elbow = 23;
  // const int left_wrist = 24;
  // const int right_wrist = 25;
  // const int left_pinky_knuckle = 26;
  // const int right_pinky_knuckle = 27;
  // const int left_middle_tip = 28;
  // const int right_middle_tip = 29;
  // const int left_index_knuckle = 30;
  // const int right_index_knuckle = 31;
  // const int left_thumb_tip = 32;
  // const int right_thumb_tip = 33;

  float threshold = 0.1;
  int window_size = 5;
  int max_num_parts = 20;
  int num_integral_samples = 7;
  float link_threshold = 0.1;
  int max_num_objects = 100;

  void *cmap_data = tensor_meta->out_buf_ptrs_host[0];
  NvDsInferDims &cmap_dims = tensor_meta->output_layers_info[0].inferDims;
  void *paf_data = tensor_meta->out_buf_ptrs_host[1];
  NvDsInferDims &paf_dims = tensor_meta->output_layers_info[1].inferDims;

  const int numKeyPoints = 34;
  // std::vector<float> keypoints(68) ;
  // std::vector<float> keypointsZRel(34);
  // std::vector<float> keypoints_confidence(34);
  float keypoints[2 * numKeyPoints];
  float keypointsZRel[numKeyPoints];
  float keypoints_confidence[numKeyPoints];

  m_K_inv_transpose = _K.inverse().eval();
  m_K_inv_transpose = m_K_inv_transpose.transpose().eval();

  NvDsBatchMeta *bmeta = frame_meta->base_meta.batch_meta;
  NvDsDisplayMeta *dmeta = nvds_acquire_display_meta_from_pool(bmeta);
  nvds_add_display_meta_to_frame(frame_meta, dmeta);

  for (unsigned int m=0; m < tensor_meta->num_output_layers;m++){
    NvDsInferLayerInfo *info = &tensor_meta->output_layers_info[m];

    if (!strcmp(info->layerName, "pose25d")) {
      float *data = (float *)tensor_meta->out_buf_ptrs_host[m];
      /*  for (int j =0 ; j < 34; j++){
          printf ("a=%f b=%f c=%f d=%f\n",data[j*4],data[j*4+1],data[j*4+2], data[j*4+3]);
          }*/

      // Initialize
      if (g_filter_pose25d.find(obj_meta->object_id) == g_filter_pose25d.end()) {
        const float m_oneEuroSampleRate = 30.0f;
        // const float m_oneEuroMinCutoffFreq = 0.1f;
        // const float m_oneEuroCutoffSlope = 0.05f;
        const float m_oneEuroDerivCutoffFreq = 1.0f;// Hz

        //std::vector <SF1eFilter*> filter_vec;
        std::vector <OneEuroFilter> filter_vec;

        for (int j=0; j < numKeyPoints*3; j++) {
            //TODO:Pending delete especially when object goes out of view, or ID switch
            //will cause memleak, cleanup required wrap into class
         //   filter_vec.push_back(SF1eFilterCreate(30, 1.0, 0.0, 1.0));

          // filters for x and y
          // for (auto& fil : m_filterKeypoints2D) fil.reset(m_oneEuroSampleRate, 0.1f, 0.05, m_oneEuroDerivCutoffFreq);
          filter_vec.push_back(OneEuroFilter(m_oneEuroSampleRate, 0.1f, 0.05, m_oneEuroDerivCutoffFreq));
          filter_vec.push_back(OneEuroFilter(m_oneEuroSampleRate, 0.1f, 0.05, m_oneEuroDerivCutoffFreq));

          // filters for z (depth)
          // for (auto& fil : m_filterKeypointsRelDepth) fil.reset(m_oneEuroSampleRate, 0.5f, 0.05, m_oneEuroDerivCutoffFreq);
          filter_vec.push_back(OneEuroFilter(m_oneEuroSampleRate, 0.5f, 0.05, m_oneEuroDerivCutoffFreq));
        }
        g_filter_pose25d[obj_meta->object_id] = filter_vec;

        // Filters depth of root keypoint
        m_filterRootDepth.reset(m_oneEuroSampleRate, 0.1f, 0.05f, m_oneEuroDerivCutoffFreq);
      }

      int batchSize_offset = 0;

      //std::vector<SF1eFilter*> &filt_val = g_filter_pose25d[obj_meta->object_id];
      std::vector<OneEuroFilter> &filt_val = g_filter_pose25d[obj_meta->object_id];

      // x,y,z,c
      for (int i = 0; i < numKeyPoints; i++) {
        int index = batchSize_offset + i * 4;

        // Update with filtered results
        keypoints[2 * i    ] = filt_val[3 * i    ].filter(data[index    ] *
                                (obj_meta->rect_params.width / 192.0)  + obj_meta->rect_params.left);
        keypoints[2 * i + 1] = filt_val[3 * i + 1].filter(data[index + 1] *
                                (obj_meta->rect_params.height / 256.0) + obj_meta->rect_params.top);
        keypointsZRel[i]     = filt_val[3 * i + 2].filter(data[index + 2]);

        keypoints_confidence[i] = data[index + 3];
      }

      // Since we have cropped and resized the image buffer provided to the SDK from the app,
      // we scale and offset the points back to the original resolution
      float scaleOffsetXY[] = {1.0f, 0.0f, 1.0f, 0.0f};

      // Render upper body
      if (1) {
        osd_upper_body(frame_meta, bmeta, dmeta, numKeyPoints, keypoints, keypoints_confidence);
      }
      // Render lower body
      if (1) {
        osd_lower_body(frame_meta, bmeta, dmeta, numKeyPoints, keypoints, keypoints_confidence);
      }

      // SGIE operates on an enlarged/padded image buffer.
      // const int muxer_output_width_pad = _pad_dim * 2 + _image_width;
      // const int muxer_output_height_pad = _pad_dim * 2 + _image_height;
      // Before outputting result, the image frame with overlay is cropped by removing _pad_dim.
      // The final pose estimation result should counter the padding before deriving 3D keypoints.
      for (int i = 0; i < numKeyPoints; i++) {
        keypoints[2 * i    ]-= _pad_dim;
        keypoints[2 * i + 1]-= _pad_dim;
      }

      // Recover pose 3D
      std::vector<NvAR_Point3f> p3dLifted;
      p3dLifted = liftKeypoints25DTo3D(keypoints, keypointsZRel, numKeyPoints, m_K_inv_transpose, m_scale_ll);
      float scale = recoverScale(p3dLifted, keypoints_confidence, m_mean_ll);
      // printf("scale = %f\n", scale);
      for (auto i = 0; i < p3dLifted.size(); i++) {
        p3dLifted[i].x *= scale;
        p3dLifted[i].y *= scale;
        p3dLifted[i].z *= scale;
      }

      if (_nvmsgbroker_conn_str) {// Prepare metadata to message broker
        build_msg_meta(frame_meta, obj_meta,
          numKeyPoints, keypoints, keypointsZRel,
          keypoints_confidence, p3dLifted);
        g_debug("Sent metadata of frame %6d to message broker.", frame_meta->frame_num);
      }

      // Output pose25d and pose3d tensors
      if (_pose_file) {
        fprintf(_pose_file,
          "{\n"
          "      \"object_id\": %lu,\n",
          obj_meta->object_id);

        // Write pose25d
        fprintf(_pose_file,
          "      \"pose25d\": [");
        for (int i = 0; i < p3dLifted.size(); i++) {
          // Remember the position of "," so that we can remove it on the last entry.
          fprintf(_pose_file, "%f, %f, %f, %f", keypoints[2*i], keypoints[2*i+1],
            keypointsZRel[i], keypoints_confidence[i]);
          fgetpos(_pose_file, &g_fp_25_pos);
          fprintf(_pose_file, ", ");
        }
        fsetpos(_pose_file, &g_fp_25_pos);
        fprintf(_pose_file, "],\n");

        // Write the recovered pose3d
        fprintf(_pose_file,
          "      \"pose3d\": [");
        for (int i = 0; i < p3dLifted.size(); i++) {
          // Remember the position of "," so that we can remove it on the last entry.
          fprintf(_pose_file, "%f, %f, %f, %f", p3dLifted[i].x, p3dLifted[i].y, p3dLifted[i].z, keypoints_confidence[i]);
          fgetpos(_pose_file, &g_fp_25_pos);
          fprintf(_pose_file, ", ");
        }
        fsetpos(_pose_file, &g_fp_25_pos);
        fprintf(_pose_file, "]\n");

        // Remember the position of "," so that we can remove it on the last entry.
        fprintf(_pose_file, "    }");
        fgetpos(_pose_file, &g_fp_25_pos);
        fprintf(_pose_file, ", ");
      }
    }
  }
}

/* pgie_src_pad_buffer_probe will extract metadata received from pgie
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
pgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  gchar *msg = NULL;
  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsMetaList *l_user = NULL;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  /* Padding due to AR SDK model requires bigger bboxes*/
  const int muxer_output_width_pad = _pad_dim * 2 + _image_width;
  const int muxer_output_height_pad = _pad_dim * 2 + _image_height;

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next)
    {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
      float sizex = obj_meta->rect_params.width * .5f;
      float sizey = obj_meta->rect_params.height * .5f;
      float centrx = obj_meta->rect_params.left  + sizex;
      float centry = obj_meta->rect_params.top  + sizey;
      sizex *= (1.25f);
      sizey *= (1.25f);
      if (sizex < sizey)
        sizex = sizey;
      else
        sizey = sizex;

      obj_meta->rect_params.width = roundf(2.f *sizex);
      obj_meta->rect_params.height = roundf(2.f *sizey);
      obj_meta->rect_params.left   = roundf (centrx - obj_meta->rect_params.width/2.f);
      obj_meta->rect_params.top    = roundf (centry - obj_meta->rect_params.height/2.f);

      sizex= obj_meta->rect_params.width * .5f, sizey = obj_meta->rect_params.height * .5f;
      centrx = obj_meta->rect_params.left + sizex, centry = obj_meta->rect_params.top + sizey;
      // Make sure box has same aspect ratio as 3D Body Pose model's input dimensions
      // (e.g 192x256 -> 0.75 aspect ratio) by enlarging in the appropriate dimension.
      float xScale = (float)192.0 / (float)sizex, yScale = (float)256.0 / (float)sizey;
      if (xScale < yScale) { // expand on height
          sizey = (float)256.0/ xScale;
      }
      else { // expand on width
          sizex = (float)192.0 / yScale;
      }

      obj_meta->rect_params.width = roundf(2.f *sizex);
      obj_meta->rect_params.height = roundf(2.f *sizey);
      obj_meta->rect_params.left   = roundf (centrx - obj_meta->rect_params.width/2.f);
      obj_meta->rect_params.top    = roundf (centry - obj_meta->rect_params.height/2.f);
      if (obj_meta->rect_params.left < 0.0) {
          obj_meta->rect_params.left = 0.0;
      }
      if (obj_meta->rect_params.top < 0.0) {
        obj_meta->rect_params.top = 0.0;
      }
      if (obj_meta->rect_params.left + obj_meta->rect_params.width > muxer_output_width_pad -1){
        obj_meta->rect_params.width = muxer_output_width_pad - 1 - obj_meta->rect_params.left;
      }
      if (obj_meta->rect_params.top + obj_meta->rect_params.height > muxer_output_height_pad -1){
        obj_meta->rect_params.height = muxer_output_height_pad - 1 - obj_meta->rect_params.top;
      }

    }
  }
  return GST_PAD_PROBE_OK;
}

/* sgie_src_pad_buffer_probe  will extract metadata received from pgie
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
sgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  gchar *msg = NULL;
  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsMetaList *l_user = NULL;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  if (_pose_file) {// Write batch header
    if (batch_meta->frame_meta_list) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(batch_meta->frame_meta_list->data);
      if (frame_meta->obj_meta_list) {
        fprintf(_pose_file,
          "{\n"
          "  \"num_frames_in_batch\": %d,\n"
          "  \"batches\": [",
          batch_meta->num_frames_in_batch);
      }
    }
  }

  // g_mutex_lock(&str->struct_lock);
  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

    if (_pose_file) {// Write frame header
      if (frame_meta->obj_meta_list) {
        fprintf(_pose_file,
          "{\n"
          "    \"batch_id\": %d,\n"
          "    \"frame_num\": %d,\n"
          "    \"ntp_timestamp\": %ld,\n"
          "    \"num_obj_meta\": %d,\n"
          "    \"objects\": [",
          frame_meta->batch_id, frame_meta->frame_num, frame_meta->ntp_timestamp, frame_meta->num_obj_meta);
          fgetpos(_pose_file, &g_fp_25_pos);
      }
    }

    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next)
    {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;

      // Set below values to 0 in order to disable bbox and text output
      obj_meta->rect_params.border_width = 0;//2;
      obj_meta->text_params.font_params.font_size = 0;//10;

      for (l_user = obj_meta->obj_user_meta_list; l_user != NULL;
           l_user = l_user->next)
      {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
        if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
        {
          NvDsInferTensorMeta *tensor_meta =
              (NvDsInferTensorMeta *)user_meta->user_meta_data;
          parse_25dpose_from_tensor_meta(tensor_meta, frame_meta, obj_meta) ;
        }
      }
    }

    if (_pose_file) {// closing off "objects" key.
      if (frame_meta->obj_meta_list) {
        fsetpos(_pose_file, &g_fp_25_pos);
        fprintf(_pose_file, "]\n");// closing off "objects" key.

        fprintf(_pose_file, "  }");
        fgetpos(_pose_file, &g_fp_25_pos);
        fprintf(_pose_file, ", ");
      }
    }
  }

  if (_pose_file) {// closing off "batches" key.
    if (batch_meta->frame_meta_list) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(batch_meta->frame_meta_list->data);
      if (frame_meta->obj_meta_list) {
        fsetpos(_pose_file, &g_fp_25_pos);
        fprintf(_pose_file, "]\n");// closing off "batches" key.

        fprintf(_pose_file, "}");
        fgetpos(_pose_file, &g_fp_25_pos);
        fprintf(_pose_file, ", ");
      }
    }
  }
  // g_mutex_unlock (&str->struct_lock);

  return GST_PAD_PROBE_OK;
}

/* osd_sink_pad_buffer_probe  will extract metadata received from OSD
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *)info->data;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsDisplayMeta *display_meta = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    int offset = 0;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
    {
      obj_meta = (NvDsObjectMeta *)(l_obj->data);
    }
    display_meta = nvds_acquire_display_meta_from_pool(batch_meta);

    /* Parameters to draw text onto the On-Screen-Display */
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    display_meta->num_labels = 1;
    txt_params->display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
    offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Frame Number =  %d", frame_number);
    offset = snprintf(txt_params->display_text + offset, MAX_DISPLAY_LEN, " ");

    txt_params->x_offset = 10;
    txt_params->y_offset = 12;

    char font_name[] = "Mono";
    txt_params->font_params.font_name = font_name;
    txt_params->font_params.font_size = 10;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;

    txt_params->set_bg_clr = 1;
    txt_params->text_bg_clr.red = 0.0;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 1.0;

    nvds_add_display_meta_to_frame(frame_meta, display_meta);
  }
  frame_number++;
  return GST_PAD_PROBE_OK;
}

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *)data;
  switch (GST_MESSAGE_TYPE(msg))
  {
  case GST_MESSAGE_EOS:
    g_print("End of Stream\n");
    g_main_loop_quit(loop);
    break;

  case GST_MESSAGE_ERROR:
  {
    gchar *debug;
    GError *error;
    gst_message_parse_error(msg, &error, &debug);
    g_printerr("ERROR from element %s: %s\n",
               GST_OBJECT_NAME(msg->src), error->message);
    if (debug)
      g_printerr("Error details: %s\n", debug);
    g_free(debug);
    g_error_free(error);
    g_main_loop_quit(loop);
    break;
  }

  default:
    break;
  }
  return TRUE;
}

gboolean
link_element_to_tee_src_pad(GstElement *tee, GstElement *sinkelem)
{
  gboolean ret = FALSE;
  GstPad *tee_src_pad = NULL;
  GstPad *sinkpad = NULL;
  GstPadTemplate *padtemplate = NULL;

  padtemplate = (GstPadTemplate *)gst_element_class_get_pad_template(GST_ELEMENT_GET_CLASS(tee), "src_%u");
  tee_src_pad = gst_element_request_pad(tee, padtemplate, NULL, NULL);

  if (!tee_src_pad)
  {
    g_printerr("Failed to get src pad from tee");
    goto done;
  }

  sinkpad = gst_element_get_static_pad(sinkelem, "sink");
  if (!sinkpad)
  {
    g_printerr("Failed to get sink pad from '%s'",
               GST_ELEMENT_NAME(sinkelem));
    goto done;
  }

  if (gst_pad_link(tee_src_pad, sinkpad) != GST_PAD_LINK_OK)
  {
    g_printerr("Failed to link '%s' and '%s'", GST_ELEMENT_NAME(tee),
               GST_ELEMENT_NAME(sinkelem));
    goto done;
  }
  ret = TRUE;

done:
  if (tee_src_pad)
  {
    gst_object_unref(tee_src_pad);
  }
  if (sinkpad)
  {
    gst_object_unref(sinkpad);
  }
  return ret;
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
      if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
              decoder_src_pad)) {
        g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref (bin_ghost_pad);
    } else {
      g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  g_print ("Decodebin child added: %s\n", name);
  if (g_strrstr (name, "urisourcebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
  /* Commented to suppress warnings*/
  /*if (!(g_strcmp0 (source_type, "rtmp"))) {
     g_object_set (G_OBJECT (object), "do-timestamp", 1, NULL);
     g_object_set (G_OBJECT (object), "timeout", 10000, NULL);
  }*/
}

// Imported from deepstream_test3_app.c
static GstElement *
create_source_bin (guint index, gchar * uri)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = { };

  g_snprintf (bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");

  if (!bin || !uri_decode_bin) {
    g_printerr ("One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  g_object_set (G_OBJECT (uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);

  gst_bin_add (GST_BIN (bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}


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

  _cintr = TRUE;
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

/**
 * Loop function to check the status of interrupts.
 * It comes out of loop if application got interrupted.
 */
static gboolean
check_for_interrupt (gpointer data)
{
  if (_quit) {
    return FALSE;
  }

  if (_cintr) {
    _cintr = FALSE;

    _quit = TRUE;
    GMainLoop *loop = (GMainLoop *) data;
    g_main_loop_quit (loop);

    return FALSE;
  }
  return TRUE;
}
//===from deepstream_test5_app_main.c===

bool verify_arguments()
{
  if (!_input) {
    g_printerr("--input option is not specified. Exiting...\n");
    return false;
  }
  else {
    if (strncmp(_input, "rtsp://", 7) && strncmp(_input, "file://", 7)) {
      g_printerr("--input value is not a valid URI address. Exiting...\n");
      return false;
    }
  }

  if (_tracker) {
    if (strcmp(_tracker, "accuracy") && strcmp(_tracker, "perf")) {
      g_printerr("--tracker value is neither \"accuracy\", nor \"perf\". Exiting...\n");
      return false;
    }
  }
  else {// default value
    _tracker = (gchar *)g_malloc0(64);
    strcpy(_tracker, "perf");
  }

  if (_pose_filename) {
    _pose_file = fopen(_pose_filename, "wt");
    if (!_pose_file) {
      g_printerr("Cannot open file %s. Exiting...\n", _pose_filename);
      return false;
    }
    fprintf(_pose_file, "[");
    fgetpos(_pose_file, &g_fp_25_pos);
  }

  if (_publish_pose) {
    if (strcmp(_publish_pose, "pose3d") && strcmp(_publish_pose, "pose25d")) {
      g_printerr("--publish-pose value is neither \"pose3d\", nor \"pose25d\". Exiting...\n");
      return false;
    }
  }

  if (_image_width <= 0) {
    g_printerr("--width value %d is non-positive. Exiting...\n", _image_width);
    return false;
  }
  if (_image_height <= 0) {
    g_printerr("--height value %d is non-positive. Exiting...\n", _image_height);
    return false;
  }
  _focal_length = (float)_focal_length_dbl;
  if (_focal_length <= 0) {
    g_printerr("--focal value %f is non-positive. Exiting...\n", _focal_length);
    return false;
  }

  _K.row(0) << _focal_length, 0,              _image_width / 2.f;
  _K.row(1) << 0,             _focal_length,  _image_height / 2.f;
  _K.row(2) << 0,             0,              1.f;

  _pad_dim = PAD_DIM * _image_width / MUXER_OUTPUT_WIDTH;

  return true;
}

int main(int argc, char *argv[])
{
  const guint num_sources = 1;

  GMainLoop *loop = NULL;
  GstCaps *caps = NULL;
  GstElement *source = NULL, *streammux_pgie = NULL;
  GstElement *sink = NULL, *pgie = NULL;
  // Padding the image and removing the padding
  GstElement *nvvideoconvert_enlarge = NULL, *nvvideoconvert_reduce = NULL,
    *capsFilter_enlarge = NULL, *capsFilter_reduce = NULL;
  GstElement *nvvidconv = NULL, *nvosd = NULL, *tracker = NULL, *nvdslogger = NULL,
    *filesink = NULL, *nvvideoencfilesinkbin = NULL,
    *nvrtspoutsinkbin = NULL;
  GstElement *tee = NULL, *msgbroker = NULL, *msgconv = NULL;// msg broker and converter
  GstBus *bus = NULL;
  guint bus_watch_id;

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  //---Parse command line options---
  {
    const GOptionEntry entries[] = {
      {"version", 'v', 0, G_OPTION_ARG_NONE, &_print_version,
        "Print DeepStreamSDK version.", NULL}
      ,
      {"version-all", 0, 0, G_OPTION_ARG_NONE, &_print_dependencies_version,
        "Print DeepStreamSDK and dependencies version.", NULL}
      ,
      {"input", 0, 0, G_OPTION_ARG_STRING, &_input,
        "[Required] Input video address in URI format by starting \
with \"rtsp://\" or \"file://\".",
        NULL}
      ,
      {"output", 0, 0, G_OPTION_ARG_STRING, &_output,
        "Output video address. Either \"rtsp://\" or a file path or \"fakesink\" is \
acceptable. If the value is \"rtsp://\", then the result video is \
published at \"rtsp://localhost:8554/ds-test\".",
        NULL}
      ,
      {"save-pose", 0, 0, G_OPTION_ARG_STRING, &_pose_filename,
        "The file path to save both the pose25d and the recovered \
pose3d in JSON format.",
        NULL}
      ,
      {"conn-str", 0, 0, G_OPTION_ARG_STRING, &_nvmsgbroker_conn_str,
        "Connection string for Gst-nvmsgbroker, e.g. <ip address>;<port>;<topic>.",
        NULL}
      ,
      {"publish-pose", 0, 0, G_OPTION_ARG_STRING, &_publish_pose,
        "Specify the type of pose to publish. Acceptable \
value is either \"pose3d\" or \"pose25d\". If not specified, both \"pose3d\" and \"pose25d\" \
are published to the message broker.",
        NULL}
      ,
      {"tracker", 0, 0, G_OPTION_ARG_STRING, &_tracker,
        "Specify the NvDCF tracker mode. The acceptable value is either \
\"accuracy\" or \"perf\". The default value is \"perf\"  \"accuracy\" mode"\
" requires DeepSORT model to be installed. Please refer to [Setup Official Re-ID Model]"\
"(https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html) section for details.",
        NULL}
      ,
      {"fps", 0, 0, G_OPTION_ARG_NONE, &_print_fps,
        "Print FPS in the format of current_fps (averaged_fps).",
        NULL}
      ,
      {"fps-interval", 0, 0, G_OPTION_ARG_INT, &_fps_interval,
        "Interval in seconds to print the fps, applicable only with --fps flag.",
        NULL}
      ,
      {"width", 0, 0, G_OPTION_ARG_INT, &_image_width,
        "Input video width in pixels. The default value is 1280.",//MUXER_OUTPUT_WIDTH
        NULL}
      ,
      {"height", 0, 0, G_OPTION_ARG_INT, &_image_height,
        "Input video height in pixels. The default value is 720.",//MUXER_OUTPUT_HEIGHT
        NULL}
      ,
      {"focal", 0, 0, G_OPTION_ARG_DOUBLE, &_focal_length_dbl,
        "Camera focal length in millimeters. The default value is 800.79041.",//FOCAL_LENGTH
        NULL}
      ,
      {"osd-process-mode", 0, 0, G_OPTION_ARG_INT, &_osd_process_mode,
        "OSD process mode CPU - 0 or GPU 1.",
        NULL}
      ,
      {NULL}
      ,
    };

    GOptionContext *ctx = NULL;
    GOptionGroup *group = NULL;
    GError *error = NULL;
    guint i;

    ctx = g_option_context_new ("Deepstream BodyPose3DNet App");
    group = g_option_group_new ("arguments", NULL, NULL, NULL, NULL);
    g_option_group_add_entries (group, entries);

    g_option_context_set_main_group (ctx, group);
    g_option_context_add_group (ctx, gst_init_get_option_group ());

    GST_DEBUG_CATEGORY_INIT (NVDS_APP, "Deepstream BodyPose3DNet App", 0, NULL);

    if (!g_option_context_parse (ctx, &argc, &argv, &error)) {
      NVGSTDS_ERR_MSG_V ("%s", error->message);
      g_printerr ("%s",g_option_context_get_help (ctx, TRUE, NULL));
      return -1;
    }

    if (_print_version) {
      g_print ("deepstream-bodypose3dnet-app version %d.%d.%d\n",
          NVDS_APP_VERSION_MAJOR, NVDS_APP_VERSION_MINOR, NVDS_APP_VERSION_MICRO);
      return 0;
    }

    if (_print_dependencies_version) {
      g_print ("deepstream-bodypose3dnet-app version %d.%d.%d\n",
          NVDS_APP_VERSION_MAJOR, NVDS_APP_VERSION_MINOR, NVDS_APP_VERSION_MICRO);
      return 0;
    }

    if (!verify_arguments()) {
      g_printerr ("%s",g_option_context_get_help (ctx, TRUE, NULL));
      return -1;
    }
  }
  //---Parse command line options---


  /* Standard GStreamer initialization */
  // signal(SIGINT, sigintHandler);
  gst_init(&argc, &argv);
  loop = g_main_loop_new(NULL, FALSE);

  _intr_setup ();
  g_timeout_add (400, check_for_interrupt, loop);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new("deepstream-bodypose3dnet");
  if (!pipeline) {
    g_printerr ("Pipeline could not be created. Exiting.\n");
    return -1;
  }

  /* we add a message handler */
  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux_pgie = gst_element_factory_make ("nvstreammux", "streammux-pgie");
  if (!streammux_pgie) {
    g_printerr ("PGIE streammux could not be created. Exiting.\n");
    return -1;
  }
  //---Set properties of streammux_pgie---
  g_object_set(G_OBJECT(streammux_pgie), "batch-size", num_sources, NULL);
  g_object_set(G_OBJECT(streammux_pgie), "width", _image_width, "height",
      _image_height,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  gst_bin_add(GST_BIN(pipeline), streammux_pgie);
  //---Set properties of streammux_pgie---

  // !!!TODO: support >1 input streams!!!
  /* Source element for reading from the file/uri */
  {
    GstPad *sinkpad, *srcpad;
    gchar pad_name[16] = { };

    source = create_source_bin(0, const_cast<char*>(_input));
    if (!source) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }
    gst_bin_add(GST_BIN(pipeline), source);

    g_snprintf (pad_name, 15, "sink_%u", 0);
    sinkpad = gst_element_get_request_pad(streammux_pgie, pad_name);
    if (!sinkpad) {
      g_printerr ("Source Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad(source, "src");
    if (!srcpad) {
      g_printerr ("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref (srcpad);
    gst_object_unref (sinkpad);
  }

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
  if (!pgie) {
    g_printerr ("PGIE element could not be created. Exiting.\n");
    return -1;
  }
  //---Set pgie properties---
  /* Configure the nvinfer element using the nvinfer config file. */
  g_object_set(G_OBJECT(pgie), "config-file-path", PGIE_CONFIG_FILE, NULL);

  /* Override the batch-size set in the config file with the number of sources. */
  guint pgie_batch_size = 0;
  g_object_get(G_OBJECT(pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);

    g_object_set(G_OBJECT(pgie), "batch-size", num_sources, NULL);
  }

  //---Set pgie properties---

  /* We need to have a tracker to track the identified objects */
  tracker = gst_element_factory_make ("nvtracker", "tracker");
  if (!tracker) {
    g_printerr ("Nvtracker could not be created. Exiting.\n");
    return -1;
  }
  g_object_set (G_OBJECT(tracker), "ll-lib-file",
      "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
      NULL);

  if (!strcmp(_tracker, "accuracy")) {
    g_object_set(G_OBJECT(tracker), "ll-config-file",
      "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_accuracy.yml",
      NULL);
  }
  else if (!strcmp(_tracker, "perf")) {
    g_object_set(G_OBJECT(tracker), "ll-config-file",
      "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml",
      NULL);
  }

  if (_print_fps){
      nvdslogger = gst_element_factory_make ("nvdslogger", "nvdslogger");
      if (!nvdslogger) {
          g_printerr ("Nvdslogger could not be created. Exiting.\n");
          return -1;
      }
      if (_fps_interval){
        g_object_set (G_OBJECT(nvdslogger), "fps-measurement-interval-sec",
              _fps_interval,
              NULL);
      }
  }
  else {
      nvdslogger = gst_element_factory_make ("queue", NULL);
      if (!nvdslogger) {
          g_printerr ("queue could not be created. Exiting.\n");
          return -1;
      }
  }

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  GstPad* pgie_src_pad = gst_element_get_static_pad(tracker, "src");
  if (!pgie_src_pad)
    g_printerr ("Unable to get src pad for pgie\n");
  else
    gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        pgie_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref (pgie_src_pad);

  /* 3d bodypose secondary gie */
  GstElement* sgie = gst_element_factory_make("nvinfer", "secondary-nvinference-engine");
  if (!sgie) {
    g_printerr ("Secondary nvinfer could not be created. Exiting.\n");
    return -1;
  }
  //---Set sgie properties---
  /* Configure the nvinfer element using the nvinfer config file. */
  g_object_set(G_OBJECT(sgie),
    "output-tensor-meta", TRUE,
    "config-file-path", SGIE_CONFIG_FILE,
    NULL);

  /* Override the batch-size set in the config file with the number of sources. */
  guint sgie_batch_size = 0;
  g_object_get(G_OBJECT(sgie), "batch-size", &sgie_batch_size, NULL);
  if (sgie_batch_size < num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        sgie_batch_size, num_sources);

    g_object_set(G_OBJECT(sgie), "batch-size", num_sources, NULL);
  }

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  GstPad* sgie_src_pad = gst_element_get_static_pad(sgie, "src");
  if (!sgie_src_pad)
    g_printerr("Unable to get src pad for sgie\n");
  else
    gst_pad_add_probe(sgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        sgie_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref(sgie_src_pad);
  //---Set sgie properties---

  /* Create tee to render buffer and send message simultaneously*/
  tee = gst_element_factory_make ("tee", "nvsink-tee");

  /* Add queue elements between every two elements */
  GstElement* queue_nvvidconv = NULL;
  queue_nvvidconv = gst_element_factory_make("queue", "queue_nvvidconv");
  if (!queue_nvvidconv) {
    g_printerr ("queue_nvvidconv could not be created. Exiting.\n");
    return -1;
  }

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
  if (!nvvidconv) {
    g_printerr ("nvvidconv could not be created. Exiting.\n");
    return -1;
  }

  //---Manipulate image size so that PGIE bbox is large enough---
  // Enlarge image so that PeopleNet detected bbox is larger which would fully cover the
  // detected object in the original sized image.
  nvvideoconvert_enlarge = gst_element_factory_make("nvvideoconvert", "nvvideoconvert_enlarge");
  if (!nvvideoconvert_enlarge) {
    g_printerr ("nvvideoconvert_enlarge could not be created. Exiting.\n");
    return -1;
  }
  capsFilter_enlarge = gst_element_factory_make("capsfilter", "capsFilter_enlarge");
  if (!capsFilter_enlarge) {
    g_printerr ("capsFilter_enlarge could not be created. Exiting.\n");
    return -1;
  }

  // Reduce the previously enlarged image frame so that the final output video retains the
  // same dimension as the pipeline's input video dimension.
  nvvideoconvert_reduce = gst_element_factory_make("nvvideoconvert", "nvvideoconvert_reduce");
  if (!nvvideoconvert_reduce) {
    g_printerr ("nvvideoconvert_reduce could not be created. Exiting.\n");
    return -1;
  }
  capsFilter_reduce = gst_element_factory_make("capsfilter", "capsFilter_reduce");
  if (!capsFilter_reduce) {
    g_printerr ("capsFilter_reduce could not be created. Exiting.\n");
    return -1;
  }

  gchar *string1 = NULL;
  asprintf (&string1, "%d:%d:%d:%d", _pad_dim, _pad_dim, _image_width, _image_height);
  // "dest-crop" - input size < output size
  g_object_set(G_OBJECT(nvvideoconvert_enlarge), "dest-crop", string1,"interpolation-method",1 ,NULL);
  // "src-crop" - input size > output size
  g_object_set(G_OBJECT(nvvideoconvert_reduce), "src-crop", string1,"interpolation-method",1 ,NULL);
  free(string1);

  /* Padding due to AR SDK model requires bigger bboxes*/
  const int muxer_output_width_pad = _pad_dim * 2 + _image_width;
  const int muxer_output_height_pad = _pad_dim * 2 + _image_height;
  asprintf (&string1, "video/x-raw(memory:NVMM),width=%d,height=%d",
      muxer_output_width_pad, muxer_output_height_pad);
  GstCaps *caps1 = gst_caps_from_string (string1);
  g_object_set(G_OBJECT(capsFilter_enlarge),"caps", caps1, NULL);
  free(string1);
  gst_caps_unref(caps1);

  asprintf (&string1, "video/x-raw(memory:NVMM),width=%d,height=%d",
      _image_width, _image_height);
  caps1 = gst_caps_from_string (string1);
  g_object_set(G_OBJECT(capsFilter_reduce),"caps", caps1, NULL);
  free(string1);
  gst_caps_unref(caps1);
  //---Manipulate image size so that PGIE bbox is large enough---

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");
  if (!nvosd) {
    g_printerr ("Nvdsosd could not be created. Exiting.\n");
    return -1;
  }

  g_object_set (G_OBJECT(nvosd), "process-mode", _osd_process_mode, NULL);

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  GstPad* osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
  if (!osd_sink_pad)
    g_print("Unable to get sink pad\n");
  else
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      osd_sink_pad_buffer_probe, (gpointer)sink, NULL);
  gst_object_unref(osd_sink_pad);

  /* Set output file location */
  if (_output) {
    if (!strcmp(_output, "rtsp://")) {
      filesink = gst_element_factory_make("nvrtspoutsinkbin", "nv-filesink");
    }
    else if (!strcmp(_output,"fakesink")){
      filesink = gst_element_factory_make("fakesink", "nv-sink");
    }
    else {
      filesink = gst_element_factory_make("nvvideoencfilesinkbin", "nv-filesink");
    }
    if (!filesink) {
      g_printerr ("Filesink could not be created. Exiting.\n");
      return -1;
    }

    if (strcmp(_output,"fakesink")){
      g_object_set(G_OBJECT(filesink), "output-file", _output, NULL);
      g_object_set(G_OBJECT(filesink), "bitrate", 4000000, NULL);
      //g_object_set(G_OBJECT(filesink), "profile", 3, NULL);
      g_object_set(G_OBJECT(filesink), "codec", 2, NULL);//hevc
      // g_object_set(G_OBJECT(filesink), "control-rate", 0, NULL);//hevc
    }
  }
  else {
    if (prop.integrated) {
      filesink = gst_element_factory_make("nv3dsink", "nv-sink");
    } else {
      filesink = gst_element_factory_make("nveglglessink", "nv-sink");
    }
  }

  /* Add all elements to the pipeline */
  // streammux_pgie has been added into pipeline already.
  gst_bin_add_many(GST_BIN(pipeline),
    nvvideoconvert_enlarge, capsFilter_enlarge,
    pgie, tracker, sgie, tee,
    queue_nvvidconv, nvvidconv, nvosd, filesink, nvdslogger,
    nvvideoconvert_reduce, capsFilter_reduce, NULL);

  // Link elements
  if (!gst_element_link_many(streammux_pgie,
      nvvideoconvert_enlarge, capsFilter_enlarge,
      pgie, tracker, sgie, nvdslogger, tee, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }
  if (prop.integrated && _output) { // Jetson
    if (!gst_element_link_many(queue_nvvidconv, nvvidconv, nvosd,
          nvvideoconvert_reduce, capsFilter_reduce,
          filesink, NULL)) {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }
  }
  else { // dGPU & Jetson Display
    if (!gst_element_link_many(queue_nvvidconv, nvvidconv, nvosd,
          nvvideoconvert_reduce, capsFilter_reduce,
          filesink, NULL)) {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }
  }

  // Link tee and queue_nvvidconv
  {
    GstPad *sinkpad, *srcpad;

    srcpad = gst_element_get_request_pad (tee, "src_%u");
    sinkpad = gst_element_get_static_pad (queue_nvvidconv, "sink");
    if (!srcpad || !sinkpad) {
      g_printerr ("Unable to get request pads\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Unable to link tee and queue_nvvidconv.\n");
      gst_object_unref (srcpad);
      gst_object_unref (sinkpad);
      return -1;
    }

    gst_object_unref (srcpad);
    gst_object_unref (sinkpad);
  }

  if (_nvmsgbroker_conn_str) {// Publish metadata to a broker as well
    GstElement* queue_msgconv = gst_element_factory_make("queue", "queue_msgconv");
    if (!queue_msgconv) {
      g_printerr ("queue_msgconv could not be created. Exiting.\n");
      return -1;
    }

    /* Set up message broker */
    /* Create msg converter to generate payload from buffer metadata */
    msgconv = gst_element_factory_make ("nvmsgconv", "nvmsg-converter");
    // g_object_set (G_OBJECT(msgconv), "config", MSGCONV_CONFIG_FILE, NULL);
    g_object_set (G_OBJECT(msgconv), "payload-type", 1, NULL);// Minimal schema
    g_object_set (G_OBJECT(msgconv), "msg2p-newapi", 0, NULL);// Event Msg meta

    /* Create msg broker to send payload to server */
    msgbroker = gst_element_factory_make ("nvmsgbroker", "nvmsg-broker");
    g_object_set (G_OBJECT(msgbroker),
      "proto-lib", "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so",
      "conn-str", _nvmsgbroker_conn_str,
      "sync", FALSE,
      NULL);

    gst_bin_add_many(GST_BIN(pipeline), queue_msgconv, msgconv, msgbroker, NULL);

    if (!gst_element_link_many(queue_msgconv, msgconv, msgbroker, NULL)) {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }

    // Link tee with queue_msgconv
    GstPad *sinkpad, *srcpad;

    srcpad = gst_element_get_request_pad (tee, "src_%u");
    sinkpad = gst_element_get_static_pad (queue_msgconv, "sink");
    if (!srcpad || !sinkpad) {
      g_printerr ("Unable to get request pads\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Unable to link tee and queue_msgconv.\n");
      gst_object_unref (srcpad);
      gst_object_unref (sinkpad);
      return -1;
    }

    gst_object_unref (srcpad);
    gst_object_unref (sinkpad);
  }

  /* Set the pipeline to "playing" state */
  g_print("Now playing: %s\n", _input);
  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  GST_DEBUG_BIN_TO_DOT_FILE((GstBin*)pipeline, GST_DEBUG_GRAPH_SHOW_ALL, "pipeline");

  /* Wait till pipeline encounters an error or EOS */
  g_print("Running...\n");
  g_main_loop_run(loop);

  /* Out of the main loop, clean up nicely */
  g_print("Returned, stopping playback\n");
  gst_element_set_state(pipeline, GST_STATE_NULL);
  g_print("Deleting pipeline\n");
  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);
  if (_pose_file) {
    fsetpos(_pose_file, &g_fp_25_pos);
    fprintf(_pose_file, "]\n");

    fclose(_pose_file);
  }


  return 0;
}
