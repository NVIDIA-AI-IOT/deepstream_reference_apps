/*******************************************************************************
 * MIT License
 * 
 * Copyright (C) 2019 NVIDIA CORPORATION
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

#ifndef _GST_DSDIRECTION_H_
#define _GST_DSDIRECTION_H_

#include <gst/gst.h>
#include <gst/base/gstcollectpads.h>

G_BEGIN_DECLS

/* #defines don't like whitespacey bits */
#define GST_TYPE_DS_DIRECTION \
  (gst_ds_direction_get_type())
#define GST_DS_DIRECTION(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DS_DIRECTION,GstDsDirection))
#define GST_DS_DIRECTION_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DS_DIRECTION,GstDsDirectionClass))
#define GST_IS_DS_DIRECTION(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DS_DIRECTION))
#define GST_IS_DS_DIRECTION_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DS_DIRECTION))

typedef struct _GstDsDirection      GstDsDirection;
typedef struct _GstDsDirectionClass GstDsDirectionClass;

struct _GstDsDirection
{
    GstElement element;

    GstPad *srcpad;
    GstPad *sinkpads[2];
    gboolean silent;
    guint active_passthru; /*< index of the sink pad that is being connected to 
                               the source pad */
    GstCollectPads *collect;
};

struct _GstDsDirectionClass 
{
  GstElementClass parent_class;
};

GType gst_ds_direction_get_type (void);

G_END_DECLS


#endif
