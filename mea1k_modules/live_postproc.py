"""Postprocess recordings in a background thread while the stim loop records the next one.

Every recording X.raw.h5 gets a sidecar X.csv (written after stop_saving) with whatever
postprocessing needs. The output of each recording goes to processed/X.csv; recordings
that already have an output are skipped unless reprocess=True. So the same code covers
live runs, resuming after a crash, and pure reprocessing (just never call add()).

The thread never prints to the terminal: it logs to live_figures/<name>.log (overwritten
each run) and appends to <rec_dir>/processed/<name>.log. Follow it from a second terminal
with the tail command printed at start.
"""
import os
import queue
import threading
import traceback
from datetime import datetime

import pandas as pd

LIVE_FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "live_figures")
LIVE_FIG_DIR = os.path.normpath(LIVE_FIG_DIR)

def save_live_fig(fig, name, **savefig_kwargs):
    """Write live_figures/<name>.png via a temp file, a viewer never sees a half-written png."""
    os.makedirs(LIVE_FIG_DIR, exist_ok=True)
    fullfname = os.path.join(LIVE_FIG_DIR, f"{name}.png")
    tmp_fullfname = os.path.join(LIVE_FIG_DIR, f".{name}.tmp.png")
    fig.savefig(tmp_fullfname, **savefig_kwargs)
    os.replace(tmp_fullfname, fullfname)
    return fullfname

class LivePostproc(threading.Thread):
    """process_fn(rec_dir, fname, log) -> DataFrame or None (= failed), runs in the thread.
    summarize_fn(results_df, last_result) -> str, the live readout shown in status(), from
    all results so far and/or from the last processed recording.
    Results from earlier runs (existing processed/X.csv) count in status() and finish().
    match: only recordings with this in their name (rec_dir holds other recordings too)."""
    def __init__(self, rec_dir, process_fn, summarize_fn=None, name="postproc",
                 reprocess=False, match=None):
        super().__init__(daemon=True)
        self.rec_dir = rec_dir
        self.processed_dir = os.path.join(rec_dir, "processed")
        self.process_fn = process_fn
        self.summarize_fn = summarize_fn
        self.reprocess = reprocess
        self.match = match
        self.live_log_fullfname = os.path.join(LIVE_FIG_DIR, f"{name}.log")
        self.rec_log_fullfname = os.path.join(self.processed_dir, f"{name}.log")

        self.results = {}       # fname -> DataFrame
        self.failed = []
        self.n_done_this_run = 0
        self.n_queued = 0
        self.last_result = None
        self._already_done = []
        self._q = queue.Queue()
        self._lock = threading.Lock()   # results, n_queued: shared with the main thread

    def _out_fullfname(self, fname):
        return os.path.join(self.processed_dir, fname.replace(".raw.h5", ".csv"))

    def log(self, msg):
        line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}\n"
        for log_file in self._log_files:
            log_file.write(line)
            log_file.flush()

    def add(self, fname):
        """Queue a finished recording (call after its sidecar csv is written)."""
        with self._lock:
            self.n_queued += 1
        self._q.put(fname)

    def run_in_thread(self, fn):
        """Queue fn() to run in the postproc thread, after the recordings added so far, e.g. a
        live summary plot. Keeps all plotting in one thread (pyplot is not thread-safe)."""
        self._q.put(fn)

    def start(self):
        """Queue the recordings already in rec_dir, print the tail command, start the thread.
        Called before the first recording, so nothing gets queued twice."""
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(LIVE_FIG_DIR, exist_ok=True)
        self._log_files = [open(self.live_log_fullfname, "w"),
                           open(self.rec_log_fullfname, "a")]
        n_todo = 0
        for fname in sorted(os.listdir(self.rec_dir)):
            if not fname.endswith(".raw.h5") or (self.match and self.match not in fname):
                continue
            if os.path.exists(self._out_fullfname(fname)) and not self.reprocess:
                self._already_done.append(fname)
            else:
                self.add(fname)
                n_todo += 1
        self.log(f"=== {self.rec_dir}")
        self.log(f"{n_todo} existing recordings to process, "
                 f"{len(self._already_done)} already processed (reprocess={self.reprocess})")

        bar = "=" * 80
        print(f"\n\033[1;36m{bar}\n  POSTPROC LOG - follow it in a 2nd terminal:\n\n"
              f"tail -F {self.live_log_fullfname}\n\n{bar}\033[0m\n", flush=True)
        super().start()

    def run(self):
        for fname in self._already_done:
            res = pd.read_csv(self._out_fullfname(fname))
            with self._lock:
                self.results[fname] = res
        while (fname := self._q.get()) is not None:
            if callable(fname):
                try:
                    fname()
                except Exception:
                    self.log(f"run_in_thread {fname} crashed:\n{traceback.format_exc()}")
                continue
            try:
                res = self.process_fn(self.rec_dir, fname, self.log)
            except Exception:
                res = None
                self.log(f"{fname} crashed:\n{traceback.format_exc()}")
            with self._lock:
                self.n_queued -= 1
            if res is None:
                self.failed.append(fname)
                self.log(f"FAILED {fname}")
                continue
            res.to_csv(self._out_fullfname(fname), index=False)
            with self._lock:
                self.results[fname] = res
                self.last_result = res
            self.n_done_this_run += 1
            self.log(f"{fname} | {self.status()}")

    def results_df(self):
        with self._lock:
            dfs = [self.results[fname] for fname in sorted(self.results)]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def status(self):
        msg = (f"{len(self.results)} processed ({self.n_done_this_run} this run), "
               f"{self.n_queued} queued, {len(self.failed)} failed")
        results_df = self.results_df()
        if self.summarize_fn is not None and len(results_df):
            msg += f" | {self.summarize_fn(results_df, self.last_result)}"
        return msg

    def finish(self):
        """Process what is still queued, stop the thread, return all results as one df."""
        self._q.put(None)
        self.join()
        self.log(f"=== done: {self.status()}")
        if self.failed:
            self.log(f"failed: {self.failed}")
        for log_file in self._log_files:
            log_file.close()
        print(f"\nPostproc done: {self.status()}\n  log: {self.rec_log_fullfname}", flush=True)
        return self.results_df()
