#!/usr/bin/env python
"""
demo single ecg display

the script assumes that the data is sent in the following format:
-1, sample_from_A4, sample_from_A2, sample_from_A0, -1, sample_from_A4, ...
where -1 indicates the start of a new sample

only the samples from A4 are used for ecg processing

note to self : red on left arm, white on right arm, black on right leg
"""

import numpy as np
import asyncio
import matplotlib
import serial
from detector import MaxPicker, TimeDomainPeakDetector, QRSDetector
from scipy.signal import butter, sosfilt, sosfilt_zi, iirnotch
from tracker import RobustTracker
from rr_filter import RRPDAF
from utils import Buffer, Spectrogram
import argparse

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description="Since Channel PPG demo")
    parser.add_argument(
        "--comport",
        type=str,
        default="/dev/tty.usbmodem34B7DA6613DC2",
        help="COM port for serial communication",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="Baud rate for serial communication",
    )
    args = parser.parse_args()
    return args


async def fillbuffer_serial(
    comport, baudrate, buffer_A0=None, buffer_A2=None, buffer_A4=None, period=1 / 20
):
    """
    Fill the buffers with data from serial port.
    """
    ser = serial.Serial(comport, baudrate, timeout=1 / 200)
    while True:
        data = ser.readline().decode().strip()
        if data:
            data = [int(x) for x in data.split(",")]
            next_type = None
            for x in data:
                if x == -1:
                    next_type = "A4"
                elif next_type == "A4":
                    buffer_A4.append(-x)
                    next_type = "A2"
                elif next_type == "A2":
                    if buffer_A2 is not None:
                        buffer_A2.append(x)
                    next_type = "A0"
                elif next_type == "A0":
                    if buffer_A0 is not None:
                        buffer_A0.append(x)
                    next_type = None
                else:
                    print("unexpected type")
                    next_type = None
        await asyncio.sleep(period)


async def update_figure(
    ppg_buffer,
    spect_computer,
    detector,
    tracker,
    fig,
    line_object_for_ppg,
    peaks_object,
    image_object_for_spect,
    line_object_for_track,
    period=0.05,
    hr_display_thold_for_tracker_variance=-7.0,
):
    """
    Update the figure with new data from the buffer.
    """
    # get the number of samples for time-domain plot
    num_samples = line_object_for_ppg.get_xdata().size
    fs = 200
    t_plot = np.arange(-num_samples + 1, 1, 1) / fs

    old_spec = image_object_for_spect.get_array().data
    old_data = line_object_for_ppg.get_ydata()
    sos = butter(N=8, Wn=[0.5,55], btype="bandpass", fs=fs, output="sos")
    zi = sosfilt_zi(sos)*0
    b_notch, a_notch = iirnotch(w0=60, Q=30, fs=fs)
    sos_notch = np.array([b_notch.tolist() + a_notch.tolist()])
    zi_notch = sosfilt_zi(sos_notch)*0
    # td_detector = TimeDomainPeakDetector(fs=fs)
    td_detector = QRSDetector(fs=fs,thold = 5e-7, max_level=5, winsize=0.33)

    # set up IBI filter
    filter_memory = 10
    start_mean = 1.0
    params = {
        "a": start_mean * 0.5 * filter_memory,
        "b": filter_memory,
        "c": 0.5 * filter_memory / start_mean,
        "d": filter_memory / 2,
        "gam": (filter_memory - 1) / filter_memory,
    }

    rf_pdaf = RRPDAF(**params, prob_error=0.09)
    latest_peak = 0
    delay = td_detector.delay
    normalize_window = np.hanning(10)
    while True:
        new_ppg = ppg_buffer()
        
        if new_ppg.size > 0:
            new_ppg = (new_ppg - 512) / 512
            new_data, zi_notch = sosfilt(sos_notch, new_ppg, zi=zi_notch)
            new_data, zi = sosfilt(sos, new_data, zi=zi)
            peak_labels = np.array([td_detector(x) for x in new_data])
            # construct ibis
            latest_peak -= new_data.size/fs
            
            # new_data = td_detector(new_ppg)
            new_pts = new_data.size
            num_pts = old_data.size
            if peak_labels.size > num_pts:
                peak_labels = peak_labels[-num_pts:]
            new_peaks = t_plot[-peak_labels.size:][peak_labels]
            if new_peaks.size > 0:
                new_segments = [np.array([[p - delay/fs, -1], [p - delay/fs, 1]]) for p in new_peaks] 
                if True:
                    for p in new_peaks:
                        ibi = p - latest_peak
                        latest_peak = p
                        mu, lam, prob_anomalous = rf_pdaf(ibi)
            else:
                new_segments = []
                mu = rf_pdaf.mu
            
            if new_pts < num_pts:
                data = np.concatenate([old_data[new_pts:], new_data])
                # update vertical lines for peaks
                # remove segments that are out of range
                slide_by = new_pts/fs
                segments = peaks_object.get_segments()
                # move segments by slide_by
                dif = np.array([[slide_by, 0], [slide_by, 0]])
                segments = [(s - dif) for s in segments if s[0,0] > t_plot[0] + slide_by]
                # add new segments
                segments += new_segments
                peaks_object.set_segments(segments)
            else:
                data = new_data[-num_pts:]
                # peaks_object.set_segments(new_segments)
            old_data = data
            nmin = np.min(data)
            nmax = np.max(data)
            if nmax - nmin > 0:
                data = 1.95 * (data - nmin) / (nmax - nmin) - 1.95 / 2
            line_object_for_ppg.set_ydata(data)
            new_spec = spect_computer(new_data)

            if new_spec.size > 0:
                est = []
                old_track = line_object_for_track.get_ydata()
                new_track = []
                for col_ind in np.arange(new_spec.shape[1]):
                    # detection = detector(new_spec[:, col_ind])
                    detection = 1/mu
                    temp = 60 * tracker(detection)
                    new_track.append(temp)
                    var = np.log(tracker.state.variance)
                    if var > hr_display_thold_for_tracker_variance:
                        temp = "--"
                    else:
                        temp = f"{temp:.1f}"
                    print(f"log-variance : {var:.2f}")
                    fig.suptitle(f"HR : {temp} BPM")
                    est.append(temp)
                temp = np.concatenate(
                    [old_spec[:, new_spec.shape[1] :], new_spec], axis=1
                )

                old_spec = temp
                temp = np.abs(temp)
                smooth = np.convolve(
                    np.mean(temp, axis=0), normalize_window, mode="same"
                )
                temp = temp / (1e-10 + smooth)
                temp = temp / np.max(np.abs(temp))
                temp = 20 * np.log10(temp + 1e-10)
                image_object_for_spect.set_array(temp)
                image_object_for_spect.set_clim([-50, 0])

                len_new = len(new_track)
                if len_new > 0:
                    detections = np.concatenate(
                        [old_track[len_new:], np.array(new_track)]
                    )
                    line_object_for_track.set_ydata(detections)

            fig.canvas.draw_idle()

        await asyncio.sleep(period)


async def main(
    comport,
    baudrate,
    fs=200,
    timespan_timedomain=10,
    window_size_for_spect=10,
    hopsize_for_spect=0.5,
    timespan_for_spect=30,
    spect_freq_range=[0.5, 3],
):
    """
    Main function to set up buffers, figure, spectrogram, detector, and tracker.

    Parameters:
    comport: COM port for serial communication
    baudrate: Baud rate for serial communication
    fs: Sampling frequency of the input signal (Hz)
    timespan_timedomain: Time span for time-domain plot (seconds)
    window_size_for_spect: Window size for spectrogram (seconds)
    hopsize_for_spect: Hop size for spectrogram (seconds)
    timespan_for_spect: Time span for spectrogram (seconds)
    spect_freq_range: Frequency range for spectrogram ([min_freq, max_freq] in Hz
    """

    # initialize buffers
    buffer_ppg = Buffer()

    # create figure
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 10),
        gridspec_kw={"height_ratios": [1, 2]},
    )

    # initialize plotting elements
    num_samples = int(timespan_timedomain * fs)
    ax = axes[0]
    t0 = np.arange(-num_samples + 1, 1, 1) / fs
    ppg_line_object = ax.plot(
        t0, np.zeros(num_samples), "k-"
    )[0]

    peaks = np.zeros(num_samples, dtype=bool)
    peaks[::200] = True
    peaks_object = ax.vlines(t0[peaks], -1, 1, lw=5, color="y", alpha=0.25)


    # peaks_object = ax.plot(t0[peaks], np.zeros_like(t0[peaks]), "r.", alpha=0.5)
    # format axis
    ax.set_ylim([-1.1, 1.1])
    ax.set_xlabel("Time (sec)")
    ax.set_xlim([-timespan_timedomain, 0])
    ax.set_yticks([])
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax = axes[1]

    # set up Spectrogram
    num_cols = int(timespan_for_spect / hopsize_for_spect)
    spect_params = {}
    spect_params["fs"] = fs
    spect_params["window_size"] = int(fs * window_size_for_spect)
    spect_params["hopsize"] = int(hopsize_for_spect * fs)
    spect_params["nfft"] = np.max([window_size_for_spect, 2**14])
    spect_params["freq_range"] = np.array(spect_freq_range)

    spect = Spectrogram(
        **spect_params,
    )

    # set up detector and tracker
    detector = MaxPicker(freqs=spect.freqs, f_low=0.6, f_high=3)
    
    tracker = RobustTracker(probability_of_detection=0.75)

    # initialize the spectrogram plot
    t = np.arange(num_cols) * spect_params["hopsize"] / spect_params["fs"]
    t -= np.max(t)
    im = ax.pcolormesh(t, 60 * spect.freqs, np.zeros((spect.freqs.size, num_cols)))

    # format the spectrogram axis
    ax.tick_params(axis="y", right=True, labelright=True)
    ax.set_yticks(np.arange(*(60 * spect_params["freq_range"]), 20), minor=False)
    ax.set_yticks(np.arange(*(60 * spect_params["freq_range"]), 10), minor=True)
    ax.grid("--", linewidth=1, axis="y", which="major", color="w", alpha=0.75)
    ax.grid("--", linewidth=0.5, axis="y", which="minor", color="w", alpha=0.5)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel(
        "Freq (BPM)",
    )

    track_line_object = ax.plot(
        t, int(timespan_for_spect) * np.ones_like(t), "r.-", alpha=0.5
    )[0]

    ax.set_ylim([30, 180])
    plt.pause(0.01)
    fig.canvas.draw_idle()
    print("Starting")
    update_fig = asyncio.create_task(
        update_figure(
            buffer_ppg,
            spect,
            detector,
            tracker,
            fig,
            ppg_line_object,
            peaks_object,
            im,
            track_line_object,
            period=1 / 20,
        )
    )
    update_data = asyncio.create_task(
        fillbuffer_serial(
            comport=comport,
            baudrate=baudrate,
            buffer_A4=buffer_ppg,
            period=1 / 500,
        )
    )
    while True:
        fig.canvas.flush_events()
        await asyncio.sleep(1 / 30)


if __name__ == "__main__":
    args = get_args()
    comport = args.comport
    baudrate = args.baudrate
    print(f"Using comport {comport} at baudrate {baudrate}")
    asyncio.run(main(comport=comport, baudrate=baudrate))
