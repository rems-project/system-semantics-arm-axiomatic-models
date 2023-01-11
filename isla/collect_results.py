import os
import sys
import asyncio
import pathlib
import argparse
import traceback
import subprocess
import datetime as dt

thisdir = pathlib.Path(__file__).parent.absolute()
parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--extraargs", default="")
parser.add_argument("--testlist", default="")
parser.add_argument("--draw-graphs", default=False, action='store_true')
parser.add_argument("--outfile")
parser.add_argument("--max-workers", default=None)
parser.add_argument("--tmpdir", default=None)
args = parser.parse_args()

model = args.model
extraargs = args.extraargs
testlist = args.testlist

def log(fmt, *args, **kwargs):
    print("[{!s}] {}".format(dt.datetime.now(), fmt.format(*args, **kwargs)), flush=True, file=sys.stderr)

async def run_isla_on(worker_id, fname, outfile):
    testname = pathlib.Path(fname).stem
    test_log = (pathlib.Path(tmpdir) / testname).with_suffix(".log")
    test_error_log = (pathlib.Path(tmpdir) / testname).with_suffix(".error.log")

    try:
        argv = [
            "-s",
            "--no-print-directory",
            "TEST={}".format(fname),
            "MODEL={}".format(model),
            "EXTRAARGS=--remove-uninteresting safe --verbose {}".format(extraargs),
            "DRAW_GRAPHS={}".format("1" if args.draw_graphs else "")
        ]
        log("<{worker_id}> running... {} :: {}", fname, argv, worker_id=worker_id)
        with open(test_log, "w") as tmpdirfile:
            p = await asyncio.create_subprocess_exec("make", *argv, stdout=outfile, stderr=tmpdirfile, encoding="utf-8")
            rc = await p.wait()
            log("<{worker_id}> ...finished run {} returns {}", fname, rc, worker_id=worker_id)
    except Exception as e:
        log("<{worker_id}> for {}, error: {!r}", testname, e, worker_id=worker_id)

        with open(test_error_log, "w") as tmperrlogfile:
            traceback.print_exception(*sys.exc_info(), file=tmperrlogfile)
    except BaseException as e:
        log("<{worker_id}> Unknown BaseException: {e!r}", e=e, worker_id=worker_id)
        raise

async def worker(worker_id, q):
    while True:
        job = await q.get()
        if job is None:
            return

        t, f = job
        await run_isla_on(worker_id, t, f)

async def main():
    with open(resultfile, "w") as f:
        channel = asyncio.Queue()
        worker_tasks = [worker(i, channel) for i in range(workers)]

        for t in tests:
            channel.put_nowait((t, f))

        for _ in worker_tasks:
            channel.put_nowait(None)

        await asyncio.gather(*worker_tasks)

tests = [
    str(f)
    for f in (thisdir/"tests").iterdir()
    if f.is_file() and f.suffix == ".toml"
]

if testlist:
    if "," in testlist:
        tests = testlist.split(",")
    else:
        tests = testlist.split(" ")

resultfile = args.outfile
if resultfile is None:
    resultfile = "results_{}.txt".format(model)

workers = args.max_workers
if workers is not None:
    workers = int(workers)

tmpdir = args.tmpdir
if tmpdir is None:
    tmpdir = os.environ["TMPDIR"]
os.makedirs(tmpdir, exist_ok=True)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())