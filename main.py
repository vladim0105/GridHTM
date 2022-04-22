import argparse
import json
import os
import pickle

import model
import numpy as np
import progressbar
import cv2.cv2 as cv2
import utils

def concat_seg(frame, success):
    if not success:
        return None
    seg_1 = frame[:frame.shape[0] // 2, :]
    seg_2 = frame[frame.shape[0] // 2:, :]
    # out = np.maximum(seg_1, seg_2)
    out = seg_1  # For simplicity, we only look at one class of object
    return out


def get_divisible_shape(current_shape, cell_size):
    width = current_shape[0]
    height = current_shape[1]
    new_width = (width + cell_size) - (width % cell_size)
    new_height = (height + cell_size) - (height % cell_size)
    return new_width, new_height


def force_divisible(frame, cell_size):
    new_width, new_height = get_divisible_shape(frame.shape, cell_size)
    out = np.zeros(shape=(new_width, new_height, 3))
    out[:frame.shape[0], :frame.shape[1], :] = frame
    return out


def anomaly_detection(video_file: str, parameters_file: str, output_file: str):
    vidcap = cv2.VideoCapture(video_file)
    parameters = json.load(open(parameters_file, "rb"))

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_scale = parameters["video_scale"]
    sp_grid_size = parameters["spatial_pooler"]["grid_size"]
    tm_grid_size = parameters["temporal_memory"]["grid_size"]

    success, orig_frame = vidcap.read()
    orig_frame = concat_seg(orig_frame, success)
    scaled_frame_shape = (int(orig_frame.shape[0] * video_scale), int(orig_frame.shape[1] * video_scale))
    new_width, new_height = get_divisible_shape(scaled_frame_shape, sp_grid_size)
    scaled_sdr_shape = (
        int(new_width * 1), int(new_height * 1))
    sp_args = model.SpatialPoolerArgs()
    sp_args.seed = parameters["seed"]
    sp_args.inputDimensions = (sp_grid_size, sp_grid_size)
    sp_args.columnDimensions = (tm_grid_size, tm_grid_size)
    sp_args.potentialPct = parameters["spatial_pooler"]["potential_pct"]
    sp_args.potentialRadius = parameters["spatial_pooler"]["potential_radius"]
    sp_args.localAreaDensity = parameters["spatial_pooler"]["local_area_density"]
    sp_args.globalInhibition = parameters["spatial_pooler"]["global_inhibition"] == "True"
    sp_args.wrapAround = parameters["spatial_pooler"]["wrap_around"] == "True"
    sp_args.synPermActiveInc = parameters["spatial_pooler"]["syn_perm_active_inc"]
    sp_args.synPermInactiveDec = parameters["spatial_pooler"]["syn_perm_inactive_dec"]
    sp_args.stimulusThreshold = parameters["spatial_pooler"]["stimulus_threshold"]
    sp_args.boostStrength = parameters["spatial_pooler"]["boost_strength"]
    sp_args.dutyCyclePeriod = parameters["spatial_pooler"]["duty_cycle_period"]

    tm_args = model.TemporalMemoryArgs()

    tm_args.columnDimensions = (tm_grid_size, tm_grid_size)
    tm_args.predictedSegmentDecrement = parameters["temporal_memory"]["predicted_segment_decrement"]
    tm_args.permanenceIncrement = parameters["temporal_memory"]["permanence_increment"]
    tm_args.permanenceDecrement = parameters["temporal_memory"]["permanence_decrement"]
    tm_args.minThreshold = parameters["temporal_memory"]["min_threshold"]
    tm_args.activationThreshold = parameters["temporal_memory"]["activation_threshold"]
    tm_args.cellsPerColumn = parameters["temporal_memory"]["cells_per_column"]
    tm_args.seed = parameters["seed"]

    aggr_func = np.mean if parameters["grid_htm"]["aggr_func"] == "mean" else model.grid_mean_aggr_func
    grid_htm = model.GridHTM((new_width, new_height), sp_grid_size, tm_grid_size, sp_args, tm_args,
                             min_sparsity=parameters["grid_htm"]["min_sparsity"], sparsity=parameters["grid_htm"]["sparsity"],
                             aggr_func=aggr_func, temporal_size=parameters["grid_htm"]["temporal_size"])
    frame_skip = parameters["frame_skip"]
    frame_repeats = parameters["frame_repeats"]
    frame_repeat_start_idx = parameters["frame_repeat_start_idx"]

    out = cv2.VideoWriter(f'{output_file}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10,
                          (new_height, new_width*2), True)
    anoms = []
    raw_anoms = []
    x_vals = []

    with progressbar.ProgressBar(max_value=total_frames,
                                 widgets=["Processing Frame #", progressbar.SimpleProgress(), " | ",
                                          progressbar.ETA()]) as bar:
        bar.update(0)
        while success:
            # Encode --------------------------------------------------------------------
            frame = cv2.resize(orig_frame, dsize=(scaled_frame_shape[1], scaled_frame_shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            frame = frame
            frame = force_divisible(frame, sp_grid_size)
            frame = (frame > 200) * 255
            frame = frame.astype(np.uint8)
            encoded_input = (frame == 255)[:, :, 0].astype(np.uint8)
            # Run HTM -------------------------------------------------------------------
            anom, colored_sp_output, raw_anom = grid_htm(encoded_input)
            anoms.append(anom)
            raw_anoms.append(raw_anom)
            x_vals.append(bar.value)
            # Create output -------------------------------------------------------------
            frame_out = np.zeros(shape=(frame.shape[0] * 2, frame.shape[1], 3), dtype=np.uint8)
            colored_sp_output = cv2.resize(colored_sp_output, dsize=(scaled_sdr_shape[1], scaled_sdr_shape[0]),
                                           interpolation=cv2.INTER_NEAREST)

            frame_out[frame.shape[0]:frame.shape[0] + scaled_sdr_shape[0], 0:, :] = frame
            frame_out[0: frame.shape[0], 0:] = colored_sp_output
            frame_number = utils.text_phantom(str(bar.value), 12)
            frame_out[0:12, -(12 * 5):] = frame_number
            out.write(frame_out)

            # Get next frame -------------------------------------------------------------
            # Do not get next frame if it is currently set to be repeating the same frame
            for i in range(frame_skip):
                if bar.value < frame_repeat_start_idx or bar.value >= frame_repeat_start_idx + frame_repeats:
                    success, orig_frame = vidcap.read()
                    orig_frame = concat_seg(orig_frame, success)

                bar.update(bar.value + 1)
                if bar.value == total_frames:
                    break
            if bar.value == total_frames:
                break
    dump_data = {"anom_scores": anoms, "raw_anoms": raw_anoms, "x_vals": x_vals}
    return dump_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="The segmented video on which to perform anomaly detection.")
    parser.add_argument("params", type=str, help="The parameters file.")
    parser.add_argument("-o", "--output", type=str, help="Output name.", default="result")
    args = parser.parse_args()

    data = anomaly_detection(args.video, args.params, args.output)
    pickle.dump(data, open(f'{args.output}.pkl', 'wb'))
