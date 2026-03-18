/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef _NVDSTILER_CONFIG_H_
#define _NVDSTILER_CONFIG_H_

/**
 * @defgroup  ds_nvtiler_custom_api NvTiler Custom canvas API module
 * NvTiler CustomTile configuration structures
 */
 
#include <stdint.h>

/**
 * Holds information about an individual tile in the custom canvas.
 * One tile is the space on canvas where nvmultistreamtiler
 * render an individual source
 */
typedef struct
{
    /** sourceId identifying the video source in a DeepStream pipeline */
    uint32_t sourceId;
    /** offset from left in the unit - 1/100 or
     * percentage of canvas-width */
    float x;
    /** offset from top in the unit - 1/100 or
     * percentage of canvas-height */
    float y;
    /** tile width in the unit - 1/100 or
     * percentage of canvas-width */
    float width;
    /** tile height in the unit - 1/100 or
     * percentage of canvas-height */
    float height;
}CustomTile;

/**
 * Holds information about the custom tile canvas
 * A pointer to this memory (transfer-none) shall be
 * set on custom-tile-config property on nvmultistreamtiler plugin.
 * Please check `gst-inspect-1.0 nvmultistreamtiler` for more info.
 * Note 1: This data structure shall be filled with
 *         individual tile resolution for all involved sources.
 * Note 2: custom-tile-config property can be configured dynamically
 *         while the pipeline is running.
 * Note 3: To remove sources from the canvas (example: EOS), user shall set the
 *         custom-tile-config property again by removing entry in
 *         the array CustomTileConfig->tiles and update CustomTileConfig->length
 */
typedef struct
{
    /** custom tile-level config array
     * NOTE: nullable
     * Used to customize the tile resolution per source
     * If used, user shall pass configuration for all the
     * individual tiles in [rows X columns] canvas
     */
    CustomTile*    tiles;
    uint32_t length; /**< length of tiles config array */
}CustomTileConfig;

/**
 * Holds the Tiler Canvas Configuration
 * Note: The user cannot directly set this on nvmultistreamtiler
 *       User shall leverage properties on nvmultistreamtiler plugin
 *       to configure the canvas.
 *       Please check `gst-inspect-1.0 nvmultistreamtiler` for more info.
 */
typedef struct
{
    uint32_t width;   /**< canvas width */
    uint32_t height;  /**< canvas height */
    uint32_t columns; /**< #columns of tiles in canvas */
    uint32_t rows;    /**< #rows of tiles in canvas */
    uint32_t gpuId;   /**< #gpuId to be used for stream creation */
    CustomTileConfig customTileConfig;
}TilerConfig;


#endif /**<  _NVDSTILER_CONFIG_H_ */
