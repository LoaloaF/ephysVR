import os
import sys
import random
import glob
import datetime
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# to import logger, VR-wide constants and device paths
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

from mea1k_modules.mea1k_config_utils import create_stim_sine_sequence
from mea1k_modules.mea1k_config_utils import start_saving, stop_saving, shift_DAC
from mea1k_modules.mea1k_config_utils import turn_on_stimulation_units, turn_off_stimulation_units
from mea1k_modules.mea1k_config_utils import setup_stim_unit_characterization, find_stim_unit_amplifier
from mea1k_modules.mea1k_config_utils import get_maxlab_saving

from mea1k_connectivity_scripts.signal_helpers import estimate_frequency_power
from mea1k_modules.mea1k_raw_preproc import read_stim_DAC, read_raw_data

def _sweep_DAC(dirname, array, stim_unit, ampl_id, set_id, DAC_values, debug):
    s = get_maxlab_saving()
        
    # first measurement is always off, skip saving and run it twice
    DAC_values = [DAC_values[0]] + DAC_values
    for i, DAC_val in enumerate(DAC_values):
        fname = f"config_StimUnit{int(stim_unit):02d}_Ampl{ampl_id:04d}_Set{set_id}_DAC{DAC_val}"
        if i != 0:
            channels_args = {'channels': [ampl_id]} if not debug else {'legacy': True}
            start_saving(s, dir_name=dirname, fname=fname, **channels_args)
        time.sleep(.02)
        
        # Indicator 1
        # large effect, trace should come back to baseline
        print(f"Turning on stimulation unit {stim_unit}...\n\n")
        turn_on_stimulation_units([stim_unit], mode='small_current')
        time.sleep(.4)
        
        # Indicator 2
        shift_DAC(DAC_val)
        time.sleep(.3)
            
        # Indicator 3
        # has large effect on trace, what we care about
        array.connect_amplifier_to_ringnode(int(ampl_id))
        array.download()
        time.sleep(.3)

        print(f"Turning on stim units...\n\n")
        array.connect_amplifier_to_stimulation(ampl_id)
        array.download()
        # usually no dislocation, but required for prv offset?
        time.sleep(.3)

        print(f"Turning off stim units...\n\n")
        turn_off_stimulation_units([stim_unit])
        array.disconnect_amplifier_from_stimulation(ampl_id)
        array.disconnect_amplifier_from_ringnode(ampl_id)

        if i != 0:
            stop_saving(s)

def _get_DAC_candidate_values(centered_around, delta, n_points=32):
    # enforce max n_points values
    resolution = max(1, int((2*delta) / n_points))
    DAC_candidates = np.arange(centered_around-delta, centered_around+delta+1, resolution)
    # ensure between 0 and 1023 – shift the whole window rather than hard-clipping
    if DAC_candidates[0] < 0:
        DAC_candidates -= DAC_candidates[0]
    if DAC_candidates[-1] > 1023:
        DAC_candidates -= (DAC_candidates[-1] - 1023)
    DAC_candidates = np.clip(DAC_candidates, 0, 1023)
    return np.unique(DAC_candidates).astype(int).tolist()

def _extract_fname_info(fname):
    # config_StimUnit00_Ampl0828_Set0_DAC480.raw.h5
    stimunit_id = int(fname.split("_")[1].replace("StimUnit", ""))
    ampl_id = int(fname.split("_")[2].replace("Ampl", ""))
    set_id = int(fname.split("_")[3].replace("Set", ""))
    DAC_code = int(fname.split("_")[4].replace("DAC", "").replace(".raw.h5", ""))
    return stimunit_id, ampl_id, set_id, DAC_code

def _extract_DAC_transient_set(dirname, ampl_id, stim_unit, set_id, debug=False):
    if debug:
        fig, ax = plt.subplots(figsize=(20, 8), nrows=2, sharex=True)
    
    # go through all files in the directory, skip those that do not match the set_id, ampl_id, stim_unit
    dac_transients_res = []
    for fname in sorted(os.listdir(dirname)):
        print(fname)
        if not fname.endswith(".raw.h5"):
            print("Skipping, not raw.h5")
            continue
        if (f"Set{set_id}_" not in fname) or ("DAC" not in fname):
            print("Skipping, set_id or DAC missing", (f"Set{set_id}_" not in fname), ("DAC" not in fname), fname)
            continue
        if f"Ampl{ampl_id:04d}_" not in fname:
            print("Skipping, ampl_id mismatch", ampl_id,  f"Ampl{ampl_id:04d}_", fname)
            continue
        if f"StimUnit{stim_unit:02d}_" not in fname:
            print("Skipping, stim_unit mismatch")
            continue
        
        print("Matched file:\n\n")
        # extract dac_val
        _, ampl_id, _, DAC_val = _extract_fname_info(fname)
        
        print(f"Processing {fname} ", ampl_id, stim_unit)
        data = read_raw_data(dirname, fname, convert2uV=True,
                             subtract_dc_offset=False,)
        print(data.shape)
        dac = read_stim_DAC(dirname, fname)
        # just one row in the filet
        if dac is None:
            ampl_id_idx = 0
        else:
            ampl_id_idx = ampl_id
        
        # TODO didn't see proper peaks last time?
        from_t, to_t = 12_500, 20_000 # interval contains the transient
        if dac is not None and dac.shape[0] < to_t:
            print(f"WARNING! DAC trace too short {dac.shape[0]} < {to_t}, adjusting to available length")
            to_t = dac.shape[0]
            from_t = 0
        # from_t, to_t = 0, 10_000 # interval contains the transient
        dac_transient = data[ampl_id_idx, from_t:to_t].astype(float)
        if dac is not None:
            dac = dac[from_t:to_t]

        baseline = np.median(dac_transient[:2_000])
        dac_transient -= baseline

        # calculate derivative
        diff_dac_transient = np.diff(dac_transient)
        
        # find peak closely after the fast rise
        fast_rise_at = np.argmax(np.abs(diff_dac_transient)) 
        peak_at = fast_rise_at + np.argmax(np.abs(dac_transient[fast_rise_at: fast_rise_at + 8]))
        peak = dac_transient[peak_at]
        print(f"Baseline: {baseline:.1f}, Peak: {peak:.1f}, DAC_val: {DAC_val}, at {peak_at}")
        
        if peak_at <2000 or peak_at >3000:
            print("\n\nWARNING! Peak at unexpected location, no peak?")
            # continue
        
        dac_transients_res.append(pd.Series({
            "stimunit_id": stim_unit,
            "ampl_id": ampl_id,
            "set_id": set_id,
            "DAC_val": DAC_val,
            "peak_uV": peak,
            "peak_at": peak_at,
            "baseline": baseline,
        }))
        
        if debug:
            ax[1].plot(dac_transient, linewidth=1, alpha=.7)
            col = ax[1].lines[-1].get_color()
            
            ax[1].hlines(peak, 0, len(dac_transient), alpha=0.4, linestyle='--', color=col)
            ax[1].vlines(peak_at, -200000, 200000, alpha=0.4, linestyle='--', color=col)
            ax[1].scatter([peak_at], [peak], color=col, s=50, edgecolor=col, 
                          marker='o', zorder=10, label=f'DAC {DAC_val}, Peak {peak/1000:.0f} mV')
            
            if dac is not None:
                ax[0].plot(dac.astype(float), linewidth=1, color='black', 
                        alpha=0.4, label='DAC')
                
                ax[0].set_ylim(512-340, 512+340)
                ax[0].set_yticks(np.array((512-140, 512, 512+140)))
                ax[0].set_ylabel("DAC value")
        
            plt.title(f"Raw data from {fname}")
            plt.legend(fontsize=6, ncol=2)
    
    dac_transients_res = pd.DataFrame(dac_transients_res)
    fullfname = os.path.join(dirname, f"results_StimUnit{stim_unit:02d}_Ampl{ampl_id:04d}_Set{set_id}.csv")
    dac_transients_res.to_csv(fullfname, index=False)        
    print(dac_transients_res)
    if debug:
        plt.show()
    return dac_transients_res


# ── Parabola fitting helpers ──────────────────────────────────────────────────

def _parabola(x, a, b, c):
    """Upward-opening parabola: a*(x-b)^2 + c"""
    return a * (x - b)**2 + c

def _fit_parabola(dac_vals, abs_peaks):
    """Fit a convex parabola to (dac_vals, abs_peaks).

    Returns (vertex_dac, r_squared, popt) where popt=(a,b,c).
    vertex_dac and popt are None on failure or if fit is concave.
    """
    if len(dac_vals) < 3:
        return None, 0.0, None
    dac_vals  = np.asarray(dac_vals,  dtype=float)
    abs_peaks = np.asarray(abs_peaks, dtype=float)

    b0 = float(dac_vals[np.argmin(abs_peaks)])
    a0 = float(np.ptp(abs_peaks) / max(np.ptp(dac_vals)**2, 1e-6))
    c0 = float(np.min(abs_peaks))
    try:
        popt, _ = curve_fit(_parabola, dac_vals, abs_peaks,
                            p0=[a0, b0, c0],
                            bounds=([0, 0, 0], [np.inf, 1023, np.inf]),
                            maxfev=10_000)
        a, b, c = popt
        if a <= 0:                          # concave – not useful
            return None, 0.0, popt
        predicted = _parabola(dac_vals, *popt)
        ss_res    = np.sum((abs_peaks - predicted)**2)
        ss_tot    = np.sum((abs_peaks - np.mean(abs_peaks))**2)
        r2        = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else 0.0
        return float(b), r2, popt
    except Exception as e:
        print(f"  Parabola fit failed: {e}")
        return None, 0.0, None


def _sliding_window_parabolas(all_data, dac_range=100):
    """Fit parabolas to windows defined by a DAC range rather than point count."""
    sorted_data = all_data.sort_values('DAC_val').reset_index(drop=True)
    dac_vals = sorted_data.DAC_val.values.astype(float)
    abs_peaks = sorted_data.peak_uV.abs().values.astype(float)

    window_results = []
    for i in range(len(dac_vals)):
        # Include all points within dac_range of the starting point
        mask = (dac_vals >= dac_vals[i]) & (dac_vals <= dac_vals[i] + dac_range)
        x, y = dac_vals[mask], abs_peaks[mask]

        if len(x) < 4 or np.ptp(x) < 5: continue

        vertex, r2, popt = _fit_parabola(x, y)
        if popt is None: continue

        # Score favors high R2 and low peak voltage (c) at vertex
        score = r2 / (popt[2] + 1.0) 

        window_results.append({
            'x': x, 'y': y, 'vertex': vertex, 'r2': r2, 
            'score': score, 'popt': popt, 'curvature': abs(popt[0])
        })

    window_results.sort(key=lambda d: d['score'], reverse=True)
    return window_results, (window_results[0] if window_results else None)


def _next_search_params(all_data, centered_around, delta, prev_vertices,
                        r2_good=0.70, r2_acceptable=0.40,
                        min_delta=16, vertex_tol=8, vertex_window=3):
    """Decide the next (centered_around, delta) using sliding-window parabolas.

    Strategy
    --------
    1.  Fit local parabolas to windows of 3-4 consecutive sorted measurements.
    2.  Score = R² × |curvature| so flat regions get suppressed automatically.
    3.  If the best window is trustworthy (R² ≥ r2_good), zoom toward its
        vertex and halve delta.
    4.  If it is mediocre (R² ≥ r2_acceptable), move toward the vertex but
        keep delta – we need more data there.
    5.  If no window passes even the acceptable threshold, the region is too
        flat to locate the dip; double delta (or keep the broad range).
    6.  Convergence is declared when the last *vertex_window* vertex estimates
        all agree within *vertex_tol* DAC units.
    """
    window_results, best = _sliding_window_parabolas(all_data)

    new_center  = centered_around
    new_delta   = delta
    vertex_dac  = None
    r2          = 0.0
    popt        = None
    converged   = False

    if best is not None:
        vertex_dac = best['vertex']
        r2         = best['r2']
        popt       = best['popt']

        print(f"  Best window: DAC {best['x'][0]:.0f}–{best['x'][-1]:.0f}  "
              f"vertex={vertex_dac:.1f}  R²={r2:.3f}  "
              f"curvature={best['curvature']:.2e}  score={best['score']:.2e}")

        if r2 >= r2_acceptable and vertex_dac is not None:
            prev_vertices.append(vertex_dac)

        if r2 >= r2_good:
            # Good fit: zoom in on the vertex
            new_center = int(np.clip(round(vertex_dac), 0, 1023))
            new_delta  = max(min_delta, delta // 2)
            print(f"  Good fit → zoom to center={new_center}, delta={new_delta}")

        elif r2 >= r2_acceptable:
            # Mediocre: move toward vertex, keep resolution
            new_center = int(np.clip(round(vertex_dac), 0, 1023))
            new_delta  = delta
            print(f"  Mediocre fit → move to center={new_center}, keep delta={new_delta}")

        else:
            # Poor: landscape still flat, widen to find the dip
            new_delta  = min(256, delta * 2)
            print(f"  Poor fit (R²={r2:.3f}) → widen delta to {new_delta}, stay at center={new_center}")

    else:
        # No valid window at all – widen
        new_delta = min(256, delta * 2)
        print(f"  No valid window found → widen delta to {new_delta}")

    # Convergence: last vertex_window estimates agree within vertex_tol DAC units
    if len(prev_vertices) >= vertex_window:
        recent = prev_vertices[-vertex_window:]
        if (max(recent) - min(recent)) <= vertex_tol:
            converged = True
            print(f"  Vertices converged: {[f'{v:.1f}' for v in recent]}")

    return new_center, new_delta, vertex_dac, r2, popt, converged


def _save_parabola_plot(all_data, set_id, window_results, best_window,
                        next_center, next_delta,
                        dirname, stim_unit, ampl_id):
    """Save a per-set diagnostic: all measurements, scored local parabolas,
    and the next sampling window.  Called after every set so you can watch
    the search converge.
    """
    processed_dir = os.path.join(dirname, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    fig, axes = plt.subplots(figsize=(16, 5), ncols=3)
    ax_main, ax_score, ax_hist = axes

    # ── left: measurements + top parabola fits ────────────────────────────
    for sid, grp in all_data.groupby("set_id"):
        alpha = 0.35 + 0.5 * (sid / max(all_data.set_id.max(), 1))
        ax_main.scatter(grp.DAC_val, grp.peak_uV.abs() / 1_000,
                        alpha=alpha, s=25, zorder=3,
                        label=f"Set {int(sid)}")

    # Draw the top-3 parabola windows (faint) and the best one (bold)
    colors_w = plt.cm.autumn(np.linspace(0.1, 0.9, min(3, len(window_results))))
    for k, (wr, col) in enumerate(zip(window_results[:3], colors_w)):
        x_fit = np.linspace(wr['x'][0], wr['x'][-1], 80)
        y_fit = _parabola(x_fit, *wr['popt']) / 1_000
        lw    = 2.5 if k == 0 else 1.0
        ls    = '-'  if k == 0 else '--'
        label = (f"Best (R²={wr['r2']:.2f}, score={wr['score']:.1e})"
                 if k == 0 else f"Top-{k+1} (R²={wr['r2']:.2f})")
        ax_main.plot(x_fit, y_fit, color=col, lw=lw, ls=ls,
                     alpha=0.85, label=label)
        if k == 0 and wr['vertex'] is not None:
            vx = wr['vertex']
            vy = _parabola(vx, *wr['popt']) / 1_000
            ax_main.scatter([vx], [vy], color=col, s=90, zorder=6,
                            marker='*', label=f"Vertex={vx:.1f}")

    # next sampling window
    lo = max(0,    next_center - next_delta)
    hi = min(1023, next_center + next_delta)
    ax_main.axvspan(lo, hi, alpha=0.12, color='steelblue',
                    label=f"Next [{lo},{hi}]")
    ax_main.axvline(next_center, color='steelblue', ls=':', lw=1.2)
    ax_main.axhline(23, color='forestgreen', ls='--', lw=1, alpha=0.7,
                    label="23 mV threshold")

    ax_main.set_xlabel("DAC value")
    ax_main.set_ylabel("|Peak transient| [mV]")
    ax_main.set_title(f"StimUnit {stim_unit:02d}  Ampl {ampl_id:04d}  "
                      f"after Set {set_id}  ({len(all_data)} pts)")
    ax_main.legend(fontsize=6, loc='upper right')
    ax_main.set_xlim(0, 1023)
    ax_main.set_ylim(bottom=0)
    ax_main.grid(True, alpha=0.3)

    # ── middle: window score landscape ────────────────────────────────────
    if window_results:
        win_centers = [float(np.mean(w['x'])) for w in window_results]
        win_scores  = [w['score']              for w in window_results]
        win_r2      = [w['r2']                 for w in window_results]
        # sort by center for a clean line plot
        order = np.argsort(win_centers)
        wc    = np.array(win_centers)[order]
        ws    = np.array(win_scores) [order]
        wr2   = np.array(win_r2)     [order]
        ax_score.plot(wc, ws,  color='darkorange', lw=1.5, label='Score (R²×curv)')
        ax_score.plot(wc, wr2, color='steelblue',  lw=1.0, ls='--', label='R²')
        ax_score.axvline(next_center, color='steelblue', ls=':', lw=1.2)
        ax_score.axvspan(lo, hi, alpha=0.12, color='steelblue')
    ax_score.set_xlabel("Window centre (DAC)")
    ax_score.set_ylabel("Score / R²")
    ax_score.set_title("Sliding-window parabola scores")
    ax_score.legend(fontsize=7)
    ax_score.set_xlim(0, 1023)
    ax_score.grid(True, alpha=0.3)

    # ── right: histogram of |peak_uV| ─────────────────────────────────────
    ax_hist.hist(all_data.peak_uV.abs() / 1_000, bins=20,
                 color='steelblue', edgecolor='white', alpha=0.8)
    ax_hist.axvline(23, color='forestgreen', ls='--', lw=1,
                    label="23 mV threshold")
    ax_hist.set_xlabel("|Peak transient| [mV]")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Distribution of |peaks|")
    ax_hist.legend(fontsize=7)
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(processed_dir,
                       f"StimUnit{stim_unit:02d}_Ampl{ampl_id:04d}_Set{set_id:02d}_parabola.png")
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved parabola plot → {out}")

    
def char_stim_units(dirname, n_amplifiers=2, stim_units=list(range(32)), debug=False):
    array = setup_stim_unit_characterization(dirname)
    
    for stim_unit in stim_units:
        # initial DAC sweep parameters – start broad so the flat landscape
        # outside the dip is sampled and we can detect where it starts curving
        centered_around = 512
        delta = 256   # covers [256, 768] at 32 points, step≈16 DAC units
        for which_amplifier in range(n_amplifiers):
            # iterate random mea1k el until we find the right stim unit + amplifier
            ampl_id = find_stim_unit_amplifier(array, stim_unit=stim_unit, 
                                               which_amplifier=which_amplifier)

            # new amplifier starts with the same broad parameters
            set_id = 0
            all_data = pd.DataFrame()   # accumulate measurements across sets
            prev_vertices = []          # track consecutive vertex estimates for convergence

            while True:
                print(f"\n==========={centered_around}+-{delta}================")
                DAC_values = _get_DAC_candidate_values(centered_around, delta)
                _sweep_DAC(dirname, array, stim_unit, ampl_id, set_id, DAC_values, debug=debug)
                results = _extract_DAC_transient_set(dirname, ampl_id, stim_unit, 
                                                     set_id, debug=debug)

                # accumulate all measurements so the parabola scorer has a
                # richer picture of the landscape with every set
                all_data = pd.concat([all_data, results], axis=0, ignore_index=True)

                # Fit sliding-window parabolas to the full accumulated dataset
                # and decide where to look next
                window_results, best_window = _sliding_window_parabolas(all_data)
                centered_around, delta, vertex_dac, r2, popt, converged = \
                    _next_search_params(all_data, centered_around, delta, prev_vertices)

                # Per-set diagnostic: shows measurements, scored windows, next window
                _save_parabola_plot(all_data, set_id, window_results, best_window,
                                    next_center=centered_around,
                                    next_delta=delta,
                                    dirname=dirname,
                                    stim_unit=stim_unit,
                                    ampl_id=ampl_id)

                # ── stopping criteria ──────────────────────────────────────
                peak = all_data.peak_uV.abs().min()

                if converged and peak < 23_000:
                    best = all_data.iloc[all_data.peak_uV.abs().argmin()]
                    print(f"Stim unit {stim_unit} amplifier {ampl_id}: "
                          f"vertices converged + peak < 23 mV  →  done.\n"
                          f"Best DAC={best.DAC_val}  peak={best.peak_uV:.1f} uV")
                    print(all_data.sort_values(by='peak_uV', key=abs, ascending=True).head(3))
                    break

                if delta <= 16 and peak < 23_000:
                    best = all_data.iloc[all_data.peak_uV.abs().argmin()]
                    print(f"Stim unit {stim_unit} amplifier {ampl_id}: "
                          f"delta=16 (minimum) + peak < 23 mV  →  done.\n"
                          f"Best DAC={best.DAC_val}  peak={best.peak_uV:.1f} uV")
                    print(all_data.sort_values(by='peak_uV', key=abs, ascending=True).head(3))
                    break

                if set_id > 4:
                    print("Too many sets, stopping here.\n\n\n")
                    break

                set_id += 1

def eval_char_stim_units(dirname,  stim_units, debug=False):
    all_results = []
    for stim_unit in stim_units:
        res_fullfnames = glob.glob(os.path.join(dirname, f"results_StimUnit{stim_unit:02d}_*.csv"))
        print(f"\nFound {len(res_fullfnames)} result files for stim unit {stim_unit}")
        
        stimunit_res = pd.DataFrame()
        for fullfname in res_fullfnames:
            print(f"Reading {fullfname}")
            stimunit_res = pd.concat([stimunit_res, pd.read_csv(fullfname)], axis=0)
            print(stimunit_res)
        
        dac_median = stimunit_res[['DAC_val', 'peak_uV']].groupby('DAC_val').apply(lambda x: x.abs().mean())
        dac_median = dac_median.rolling(window=2, center=True, min_periods=1).mean()
        min_dac_value = dac_median.peak_uV.abs().idxmin()
        peak_uV = dac_median.loc[min_dac_value].peak_uV
        print(f"Stim unit {stim_unit} minimum |peak_uV| at DAC value {min_dac_value}: {peak_uV:.1f} uV")
        all_results.append(pd.Series({
            "stimunit_id": stim_unit,
            "zero_current_DAC_code": min_dac_value,
            "median_peak_uV": peak_uV,
            "n_measurements": len(stimunit_res),
            "peak_std_uV": stimunit_res.loc[stimunit_res.DAC_val==min_dac_value].peak_uV.std(),
        }))

        # final summary plot – overlay the global parabola fit too
        dac_vals_all  = stimunit_res.DAC_val.values.astype(float)
        abs_peaks_all = stimunit_res.peak_uV.abs().values.astype(float)
        vertex_f, r2_f, popt_f = _fit_parabola(dac_vals_all, abs_peaks_all)

        plt.vlines(min_dac_value, 0, 200_000, color='green', linestyle='--', label='Min |peak_uV|')
        plt.axhline(peak_uV, color='green', linestyle='--')
        plt.text(min_dac_value+2, 180_000, f"{peak_uV/1000:.1f} mV at DAC {min_dac_value}", 
                color='green', ha='left', fontsize=8)
        
        plt.scatter(stimunit_res.DAC_val, abs(stimunit_res.peak_uV), alpha=.4, s=20)
        plt.plot(dac_median.DAC_val, dac_median.peak_uV, color='blue', linewidth=1, 
                alpha=.8, label='Median |peak_uV|')

        if popt_f is not None:
            x_fit = np.linspace(dac_vals_all.min(), dac_vals_all.max(), 400)
            plt.plot(x_fit, _parabola(x_fit, *popt_f), color='crimson',
                     lw=1.5, ls='--', label=f'Global parabola fit (R²={r2_f:.3f})')

        plt.title(f"Stim unit {stim_unit}")
        plt.xlabel("DAC value")
        plt.ylabel("Peak uV")
        plt.legend()
        if not os.path.exists(os.path.join(dirname, "processed")):
            os.makedirs(os.path.join(dirname, "processed"), exist_ok=True)
        plt.savefig(os.path.join(dirname, "processed", f"StimUnit{stim_unit:02d}_characterization.png"),
                    dpi=300)
        if debug:
            plt.show()
        plt.close()
        
    all_results = pd.DataFrame(all_results)
    all_results.to_csv(os.path.join(dirname, "processed", "StimUnits_characterization.csv"), index=False)    
    print("\n\nAll results:\n", all_results)

def char_current_lsb(dirname, R, sine_ampl_DAC_units, stim_units, n_amplifiers, debug=False):
    array = setup_stim_unit_characterization(dirname)
    s = get_maxlab_saving()
    dac_settings = pd.read_csv(os.path.join(dirname, "processed", "StimUnits_characterization.csv"))
    
    for stim_unit in stim_units:
        for which_amplifier in range(n_amplifiers):
            ampl_id = find_stim_unit_amplifier(array, stim_unit=stim_unit, which_amplifier=which_amplifier)

            print(dac_settings.loc[dac_settings.stimunit_id==stim_unit])
            dac_code = int(dac_settings.loc[dac_settings.stimunit_id==stim_unit].zero_current_DAC_code)
            print(f"\n\nStim unit {stim_unit} amplifier {ampl_id}, zero current DAC code: {dac_code}")

            sine_seq = create_stim_sine_sequence(dac_id=0, amplitude=sine_ampl_DAC_units, 
                                                 f=1000, ncycles=400,
                                                 center_around=dac_code)
            
            fname = f"config_StimUnit{int(stim_unit):02d}_Ampl{ampl_id:04d}_CurrentLSB"
            start_saving(s, dir_name=dirname, fname=fname, legacy=True)
            
            turn_on_stimulation_units([stim_unit], mode='small_current')
            shift_DAC(dac_code)
            array.connect_amplifier_to_ringnode(int(ampl_id))
            array.connect_amplifier_to_stimulation(ampl_id)
            array.download()
            
            sine_seq.send()
            time.sleep(0.5)

            turn_off_stimulation_units([stim_unit])
            array.disconnect_amplifier_from_stimulation(ampl_id)
            array.disconnect_amplifier_from_ringnode(ampl_id)
            stop_saving(s)

def eval_current_lsb(dirname, R, sine_ampl_DAC_units, stim_units, debug=False):
    aggr = []
    for fname in sorted(os.listdir(dirname)):
        if not fname.endswith("_CurrentLSB.raw.h5"):
            continue
        stimunit_id = int(fname.split("_")[1].replace("StimUnit", ""))
        ampl_id = int(fname.split("_")[2].replace("Ampl", ""))

        print(f"\nProcessing {fname} ", ampl_id, stimunit_id)
        data = read_raw_data(dirname, fname, convert2uV=True, )#row_slice=[ampl_id, ],)
        dac = read_stim_DAC(dirname, fname)
        
        interv = 1200, 9400
        zero_current_DAC_code = dac[interv[0]]
        mean_ampl, phase_shift = estimate_frequency_power(data[ampl_id, interv[0]:interv[1]].astype(float), 
                                                          sampling_rate=20_000, 
                                                          debug=debug, 
                                                          min_band=960, max_band=1040,
                                                          dac=dac[interv[0]:interv[1]].astype(float) if dac is not None else None)
        aggr.append(pd.Series({
            "stimunit_id": stimunit_id,
            "mean_amplitude_uV": mean_ampl,
            "zero_current_DAC": zero_current_DAC_code,
            "resistance": R,
            "sine_amplitude_DAC_units": sine_ampl_DAC_units,
            "LSB_small_current_nA": (mean_ampl / R) / (sine_ampl_DAC_units / 1_000),  # in nA
            "phase_shift_deg": phase_shift,
        }))
        
    fig, ax = plt.subplots(figsize=(14, 8), ncols=2, nrows=2, sharex=True)
    ax = ax.flatten()
    
    ax[0].set_title("Mean Amplitude uV")
    ax[0].set_xlabel("Stim Unit ID")
    ax[0].set_ylabel("Mean Amplitude [uV]")
    ax[0].set_xticks(stim_units)
    ax[0].grid(True)
    ax[0].set_ylim(0, 60_000)
    ax[0].scatter([a.stimunit_id for a in aggr], [a.mean_amplitude_uV for a in aggr], s=50)
    
    # Circular plot for phase shift
    ax[1].set_title("Phase Shift (Circular)")
    ax[1].set_xlabel("Stim Unit ID")
    ax[1].set_ylabel("Phase Shift [deg]")
    ax[1].set_xticks(stim_units)
    # Convert degrees to radians for circular plot
    phase_radians = np.deg2rad([a.phase_shift_deg for a in aggr])
    ax[1].scatter([a.stimunit_id for a in aggr], np.mod(phase_radians, 2 * np.pi), s=50)
    ax[1].set_yticks(np.deg2rad([0, 90, 180, 270, 360]))
    ax[1].set_yticklabels(['0°', '90°', '180°', '270°', '360°'])
    ax[1].set_ylim(0, 2 * np.pi)
    ax[1].tick_params(axis='x', which='both', rotation=90)
    
    ax[2].set_title("LSB, small_current mode in [nA]")
    ax[2].set_xlabel("Stim Unit ID")
    ax[2].set_ylabel("LSB [nA]")
    ax[2].set_xticks(stim_units)
    ax[2].scatter([a.stimunit_id for a in aggr], [a.LSB_small_current_nA for a in aggr], s=50)
    ax[2].tick_params(axis='x', which='both', rotation=90)
    
    ax[3].set_title("Zero current DAC code")
    ax[3].set_xlabel("Stim Unit ID")
    ax[3].set_ylabel("Zero current DAC code")
    ax[3].set_xticks(stim_units)
    ax[3].scatter([a.stimunit_id for a in aggr], [a.zero_current_DAC for a in aggr], s=50)
    ax[3].tick_params(axis='x', which='both', rotation=90)

    result_fullfname = os.path.join(dirname, "processed", "smallcurrent_lsb_characterization.csv")
    aggr = pd.DataFrame(aggr)
    aggr.to_csv(result_fullfname, index=False)
    # copy to devices/headstage_devices basedir
    aggr.to_csv(os.path.join(dirname, "..", "..", "smallcurrent_lsb_characterization.csv"), index=False)
    plt.savefig(result_fullfname.replace(".csv", ".png"))
    if debug:
        plt.show()
    plt.close()
            
def main():
    random.seed(42)
    debug = True
    
    nas_dir = device_paths()[0]
    # nas_dir = "/home/houmanjava/nas_imitation"
    # device_dir = "devices/headstage_devices/MEA1K22/recordings"
    device_dir = "devices/well_devices/4983/recordings"
    # R = 1_000_000  # 1 MOhm
    R = 100_000  # 100 KOhm
    sine_ampl_DAC_units = 10 # in DAC units
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M")
    rec_dir = f"{t}_{R=}_CharStimUnits2026_Claude"
    # rec_dir = "2026-03-10_18.58_R=100000_CharStimUnits2026"
    # rec_dir = "2025-09-23_17.00_R=1000000_CharStimUnits"
    # rec_dir = "2025-10-14_19.52_R=1000000_CharStimUnits"
    # rec_dir = "2026-02-13_12.31_R=1000000_CharStimUnits2026"
    # take the newest directory
    # print(sorted(os.listdir(os.path.join(nas_dir, device_dir))))
    # rec_dir = sorted(os.listdir(os.path.join(nas_dir, device_dir)))[-1]
    full_path = os.path.join(nas_dir, device_dir, rec_dir)

    n_amplifiers = 1
    # stim_units = [1,2,8,9,16,17,24,30]
    stim_units = list(range(32))
    
    # Iterate over all stim units * n_amplifiers (samples) and sweep the DAC values
    # until we find the one that gives the lowest parasitic current (lowest transient peak)
    char_stim_units(full_path, n_amplifiers=n_amplifiers, stim_units=stim_units, debug=debug)
    
    # Evaluate the stim unit characterization results, aggregating all sets,
    # fitting a curve and finding the minimum
    eval_char_stim_units(full_path, stim_units=stim_units, debug=debug)
    
    # Now that we know the DAC value that gives 0 current for each stim unit,
    # we can measure the actual current that corresponds to a certain DAC amplitude
    # in small_current mode + R=1MOhm, which gives us the current LSB
    char_current_lsb(full_path, R=R, sine_ampl_DAC_units=sine_ampl_DAC_units,
                     stim_units=stim_units, n_amplifiers=n_amplifiers, debug=debug)
    
    # Evaluate the current LSB characterization results, this table is saved in 
    # the headstage_devices directory and its needed for stimulation configuration
    eval_current_lsb(full_path, R=R, sine_ampl_DAC_units=sine_ampl_DAC_units, 
                     stim_units=stim_units, debug=debug)
    
if __name__ == "__main__":
    main()