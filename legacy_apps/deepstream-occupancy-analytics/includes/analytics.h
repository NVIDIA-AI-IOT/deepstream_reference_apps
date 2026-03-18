#ifndef _ANALYTICS_H_
#define _ANALYTICS_H_

#include <gst/gst.h>

/* User defined */
typedef struct 
{

    guint32 lcc_cnt_exit;
    guint32 lccum_cnt;
    guint32 lcc_cnt_entry;
    guint32 source_id;

} AnalyticsUserMeta;

#endif
