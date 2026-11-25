#!/usr/bin/env python3
"""
run_batch.py -- transcribe WHOLE podcast episodes with free faster-whisper (base.en) and save the text in this repository.
Black Podcasting Awards.

  python run_batch.py --auto        RECOMMENDED. Claims the next free batch by itself, finishes it, claims the next, and so on,
                                    until about 6 hours have passed or no batch is left. Paste the same command into as many
                                    cloud sessions as you like: each one claims different batches.
  python run_batch.py 7             one chosen batch (20 episodes, about an hour)
  python run_batch.py 7 --count 4   batches 7, 8, 9 and 10 one after another

Reads  queue.tsv  (columns: batch, record_id, guid, url, pub_date, category, tier, minutes)
Writes results/batch_007.jsonl   one line per episode: text with a [H:MM:SS] stamp about every minute, plus timings
       claims/batch_007.claim    who is working on that batch, refreshed at every save (auto mode only)
Commits and pushes to main every 4 episodes and at the end of each batch. A claim with no refresh for 40 minutes counts as
abandoned, and another session picks the batch up where it stopped (finished episodes are skipped).
Prints only a few lines, on purpose.
"""
import argparse, csv, json, os, ssl, subprocess, sys, tempfile, threading, queue, time, uuid, urllib.request

UA = {"User-Agent": "BlackPodAwardsBot/1.0 (+https://blackpodawards.com)"}
MAX_SEC, MAX_BYTES = 130 * 60, 300_000_000
STALE_SEC = 40 * 60
ME = uuid.uuid4().hex[:8]


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def commit(msg):
    sh("git -c user.name='BPAwards cloud batch' -c user.email='batch@blackpodawards.com' commit -q -m '%s'" % msg)


def push_paths(paths, msg):
    """Add paths, commit, and push to main, retrying around other sessions' pushes."""
    for p in paths:
        sh("git add -A '%s'" % p)
    commit(msg)
    p = None
    for attempt in range(5):
        sh("git pull --rebase origin main")
        p = sh("git push origin HEAD:main")
        if p.returncode == 0:
            return True, ""
        time.sleep(2 + attempt * 4)
    return False, (p.stderr or p.stdout)[-300:]


def paragraphs(seg):
    out, cur, start = [], [], 0.0
    for s in seg:
        if not cur:
            start = s.start
        cur.append(s.text.strip())
        if s.end - start >= 60:
            out.append((start, " ".join(cur))); cur = []
    if cur:
        out.append((start, " ".join(cur)))
    return "\n".join("[%d:%02d:%02d] %s" % (int(st // 3600), int(st % 3600 // 60), int(st % 60), tx) for st, tx in out)


def batch_status(allrows, b):
    """(done_count, total) for batch b from results/ as it is now on disk."""
    guids = {r[2] for r in allrows if r[0] == str(b)}
    got = set()
    p = "results/batch_%03d.jsonl" % b
    if os.path.exists(p):
        for ln in open(p):
            try: got.add(json.loads(ln)["guid"])
            except Exception: pass
    return len(guids & got), len(guids)


def claim_age(b):
    p = "claims/batch_%03d.claim" % b
    if not os.path.exists(p):
        return None
    try:
        return time.time() - float(open(p).read().split()[1])
    except Exception:
        return STALE_SEC + 1


def claim_next(allrows, batches):
    """Pick the lowest free batch and claim it by pushing a claim file; return the batch number or None."""
    for _ in range(3):                                           # make sure finished results are pushed before we reset anything
        sh("git fetch -q origin main")
        if sh("git rev-list --count origin/main..HEAD").stdout.strip() in ("0", ""):
            break
        sh("git pull --rebase origin main"); sh("git push origin HEAD:main")
    for attempt in range(12):
        sh("git fetch -q origin main && git reset -q --hard origin/main")
        pick = None
        for b in batches:
            d, t = batch_status(allrows, b)
            if d >= t:
                continue                                         # finished
            age = claim_age(b)
            if age is not None and age < STALE_SEC:
                continue                                         # someone is working on it
            pick = b; break
        if pick is None:
            return None
        os.makedirs("claims", exist_ok=True)
        open("claims/batch_%03d.claim" % pick, "w").write("%s %d\n" % (ME, time.time()))
        sh("git add claims"); commit("claim batch %d (%s)" % (pick, ME))
        if sh("git push origin HEAD:main").returncode == 0:
            return pick                                          # our claim is the one that landed
        time.sleep(1 + attempt)                                  # someone else pushed first: look again
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", type=int, nargs="?")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=360)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.batch is None:
        a.auto = True
    t_start = time.time()
    allrows = [r for r in csv.reader(open("queue.tsv"), delimiter="\t") if r]
    batches = sorted({int(r[0]) for r in allrows})
    try:
        import faster_whisper, av, certifi  # noqa
    except Exception:
        sh("%s -m pip install -q faster-whisper av certifi" % sys.executable)
    import av, certifi, numpy as np
    from faster_whisper import WhisperModel
    model = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=a.threads)
    ctx = ssl.create_default_context(cafile=certifi.where())
    os.makedirs("results", exist_ok=True)
    stats = {"batches": 0, "ok": 0, "bad": 0, "audio": 0.0}

    def fetch(url):
        err = "request_error"
        for rng in (True, False):
            h = dict(UA)
            if rng:
                h["Range"] = "bytes=0-%d" % MAX_BYTES
            try:
                r = urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60, context=ctx)
                data, got = [], 0
                while got < MAX_BYTES:
                    b = r.read(min(1 << 20, MAX_BYTES - got))
                    if not b:
                        break
                    data.append(b); got += len(b)
                if got >= MAX_BYTES:
                    return None, "too_big"
                return b"".join(data), None
            except urllib.error.HTTPError as e:
                err = "http_%d" % e.code
                if e.code in (404, 410):
                    break
            except Exception as e:
                err = type(e).__name__
        return None, err

    def decode(raw):
        tmp = os.path.join(tempfile.gettempdir(), "ep_%d.audio" % threading.get_ident())
        open(tmp, "wb").write(raw)
        out, n, rs = [], 0, av.AudioResampler(format="s16", layout="mono", rate=16000)
        try:
            with av.open(tmp) as f:
                try:
                    for fr in f.decode(audio=0):
                        for x in rs.resample(fr):
                            arr = x.to_ndarray().reshape(-1); out.append(arr); n += len(arr)
                        if n > 16000 * MAX_SEC:
                            break
                except Exception:
                    pass
        except Exception:
            return None
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        if not out:
            return None
        return np.concatenate(out).astype(np.float32)[: 16000 * MAX_SEC] / 32768.0

    def do_batch(b):
        rows = [r for r in allrows if r[0] == str(b)]
        outp = "results/batch_%03d.jsonl" % b
        claim = "claims/batch_%03d.claim" % b
        done = set()
        if os.path.exists(outp):
            for ln in open(outp):
                try: done.add(json.loads(ln)["guid"])
                except Exception: pass
        todo = [r for r in rows if r[2] not in done]
        pipe = queue.Queue(maxsize=1)

        def producer():
            for r in todo:
                if time.time() - t_start > a.max_minutes * 60:
                    break
                raw, err = fetch(r[3])
                audio = decode(raw) if raw else None
                raw = None
                if not err and (audio is None or len(audio) < 16000 * 300):
                    err = "bad_audio"
                pipe.put((r, None if err else audio, err))
            pipe.put(None)

        threading.Thread(target=producer, daemon=True).start()
        ok = bad = since = 0; audio_s = tr_s = 0.0
        f = open(outp, "a")
        while True:
            item = pipe.get()
            if item is None:
                break
            r, audio, err = item
            rec = {"record_id": r[1], "guid": r[2], "url": r[3], "pub_date": r[4], "category": r[5], "tier": r[6],
                   "scope": "whole", "model": "faster-whisper-base.en (cloud)"}
            if err:
                rec["error"] = err; bad += 1
            else:
                w0 = time.time()
                segs, info = model.transcribe(audio, language="en", beam_size=1, vad_filter=True, condition_on_previous_text=False)
                seg = list(segs)
                plain = " ".join(s.text.strip() for s in seg).strip()
                dt = time.time() - w0
                if len(plain) < 200:
                    rec["error"] = "too_little_speech"; bad += 1
                else:
                    rec.update(text=paragraphs(seg), words=len(plain.split()), clip_seconds=len(audio) / 16000.0, work_seconds=round(dt, 1),
                               avg_logprob=sum(s.avg_logprob for s in seg) / max(len(seg), 1),
                               low_conf_share=sum(1 for s in seg if s.avg_logprob < -1.0 or s.no_speech_prob > 0.6) / max(len(seg), 1))
                    ok += 1; audio_s += rec["clip_seconds"]; tr_s += dt
            audio = None
            f.write(json.dumps(rec) + "\n"); f.flush(); since += 1
            if since >= 4:
                if a.auto:
                    open(claim, "w").write("%s %d\n" % (ME, time.time()))        # heartbeat
                push_paths([outp] + ([claim] if a.auto else []), "batch %d checkpoint" % b); since = 0
        f.close()
        finished = batch_status(allrows, b)[0] >= batch_status(allrows, b)[1]
        paths = [outp]
        if a.auto:
            if not finished and os.path.exists(claim):
                os.unlink(claim)                                    # leave early: free the batch for another session
            paths.append("claims")
        pushed, perr = push_paths(paths, "batch %d %s" % (b, "done" if finished else "stopped early"))
        stats["batches"] += 1; stats["ok"] += ok; stats["bad"] += bad; stats["audio"] += audio_s
        print("BATCH %d: %d episodes saved, %d failed, %.1fx real time, %.0f minutes so far, %s, push %s%s" % (
            b, ok, bad, audio_s / max(tr_s, 1), (time.time() - t_start) / 60, "finished" if finished else "stopped early",
            "OK" if pushed else "FAILED", "" if pushed else " " + perr), flush=True)

    if a.auto:
        while time.time() - t_start < (a.max_minutes - 50) * 60:     # a batch needs about an hour
            b = claim_next(allrows, batches)
            if b is None:
                print("AUTO DONE: no free batch left. %d batches this session, %d episodes saved, %d failed, %.1f audio hours, %.0f minutes" % (
                    stats["batches"], stats["ok"], stats["bad"], stats["audio"] / 3600, (time.time() - t_start) / 60), flush=True)
                return
            do_batch(b)
        print("AUTO DONE: time limit reached. %d batches this session, %d episodes saved, %d failed, %.1f audio hours, %.0f minutes" % (
            stats["batches"], stats["ok"], stats["bad"], stats["audio"] / 3600, (time.time() - t_start) / 60), flush=True)
    else:
        for b in range(a.batch, a.batch + a.count):
            if time.time() - t_start > a.max_minutes * 60 or not any(r[0] == str(b) for r in allrows):
                break
            do_batch(b)


if __name__ == "__main__":
    main()
