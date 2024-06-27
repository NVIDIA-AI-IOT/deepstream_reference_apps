CREATE STREAM DETECTIONS_STREAM (
    messageid varchar,
    mdsversion varchar,
    timestamp varchar,
    object struct<
        id varchar,
        speed int,
        direction int,
        orientation int,
        detection varchar,
        obj_prop struct<
            hasBasket varchar,
            confidence double>,
        bbox struct<
            topleftx int,
            toplefty int,
            bottomrightx int,
            bottomrighty int>>,
    event_des struct<
        id varchar,
        type varchar>,
    videopath varchar) WITH (
        KAFKA_TOPIC='detections', 
        VALUE_FORMAT='JSON', 
        TIMESTAMP='timestamp',
        TIMESTAMP_FORMAT='yyyy-MM-dd''T''HH:mm:ss.SSS''Z''');