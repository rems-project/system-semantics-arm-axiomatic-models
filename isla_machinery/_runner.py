#!/usr/bin/env python3
""" run litmus tests

example usage:

> ./run.py run path/to/MP+pos.litmus.toml
aarch64_mmu_strong_ETS.cat  MP+pos  Allow

> # or with all the extra info
> ./run.sh path/to/MP+pos.litmus.toml -vv
[...]

> # or for debugging a test
> ./run.sh path/to/MP+pos.litmus.toml -vvv
[...]

> # with a different model
> ./run.sh path/to/MP+pos.litmus.toml --model=../../path/to/another.cat
[...]

> # or run nightly tests
> # this generates a nightly-YYYY-MM-DD.tar.gz tarball with all the results from all the models + test LaTeX sources + diagrams
> ./nightly.sh path/to/@all --models ../../path/to/one.cat ../../path/to/second.cat
[...]
"""

import os
import io
import re
import ast
import csv
import sys
import enum
import shlex
import signal
import tarfile
import asyncio
import pathlib
import argparse
import itertools
import contextlib
import datetime as dt
import collections

HERE = pathlib.Path(__file__).parent.resolve()
REMS_DIR = HERE.parent.parent
HERDTOOLS_DIR = REMS_DIR / "herdtools7-litmus-translator"
ISLA_DIR = REMS_DIR / "isla"

if "TMPDIR" in os.environ:
    TMPDIR = pathlib.Path(os.environ["TMPDIR"])
else:
    TMPDIR = None

# assuming dir structure:
# + rems/
# +--isla/
# +--system-semantics-arm-confidential/
# +--+--isla/
# +--system-semantics-arm-axiomatic-models/
# +--+--models/
DEFAULT_ISLA_AXIOMATIC_PATH = ISLA_DIR / "target" / "release" / "isla-axiomatic"
DEFAULT_SNAPSHOTS_DIR = REMS_DIR / "isla-snapshots"
DEFAULT_CONFIG_PATH = ISLA_DIR / "configs"
DEFAULT_MODEL = REMS_DIR / "system-semantics-arm-axiomatic-models" / "models" / "aarch64_base.cat"
DEFAULT_MODEL_PGTABLE = REMS_DIR / "system-semantics-arm-axiomatic-models" / "models" / "aarch64_mmu_strong_ETS.cat"
DEFAULT_MODEL_IFETCH = REMS_DIR / "system-semantics-arm-axiomatic-models" / "models" / "aarch64_ifetch.cat"
DEFAULT_ISLA_LITMUS = ISLA_DIR / "isla-litmus" / "isla-litmus"
DEFAULT_LITMUS_TRANSLATOR = HERDTOOLS_DIR / "_build" / "install" / "default" / "bin" / "litmus-translator"

DEFAULT_ISLA_VERSION = "arm9.3"

ISLA_CONFIG_VERSIONS = {
    "arm9.3": DEFAULT_CONFIG_PATH / "armv9p3.toml",
    "arm9.3-mmu": DEFAULT_CONFIG_PATH / "armv9p3_mmu_on.toml",
    "arm9.4": DEFAULT_CONFIG_PATH / "armv9p4.toml",
}

ISLA_FOOTPRINT_CONFIG_VERSIONS = {
    "arm9.3": DEFAULT_CONFIG_PATH / "armv9p3.toml",
    "arm9.3-mmu": DEFAULT_CONFIG_PATH / "armv9p3.toml",
    "arm9.4": DEFAULT_CONFIG_PATH / "armv9p4.toml",
}

ISLA_SNAPSHOT_VERSIONS = {
    "arm9.3": DEFAULT_SNAPSHOTS_DIR / "armv9p3.ir",
    "arm9.3-mmu": DEFAULT_SNAPSHOTS_DIR / "armv9p3.ir",
    "arm9.4": DEFAULT_SNAPSHOTS_DIR / "armv9p4.ir",
}

ISLA_DEBUG_PROBES = {
    "arm9.3": [
        "AArch64_TakeReset",
        "__FetchNextInstr",
        "BranchTo",
        "AArch64_TakeException",
        "PC_read",
        "__ExecuteInstr",
    ],
    "arm9.4": [
        "AArch64_TakeReset",
        "__FetchInstr",
        "__DecodeExecute",
        "BranchTo",
        "AArch64_TakeException",
    ]
}

DEFAULT_DOT_DIR = HERE / "dots"
DEFAULT_TEX_DIR = HERE / "tex"

# default args
DEFAULT_OPT = "--remove-uninteresting=safe"

CURRENTLY_RUNNING_ISLA_INSTANCES = set()

class Mode(enum.Enum):
    DATA = enum.auto()
    IFETCH = enum.auto()
    PGTABLE = enum.auto()

DEFAULT_MODE = Mode.DATA

_default_model = {
    Mode.DATA: DEFAULT_MODEL,
    Mode.IFETCH: DEFAULT_MODEL_IFETCH,
    Mode.PGTABLE: DEFAULT_MODEL_PGTABLE,
}

_default_tmpdir = {
    Mode.DATA: HERE / "_tmp",
    Mode.IFETCH: HERE / "_tmp",
    Mode.PGTABLE: HERE / "_tmp_pgtable",
}

async def _run_isla(
    litmus_test: "TestFile",
    *extra,
    outf: io.TextIOBase,
    mode=DEFAULT_MODE,
    isla_path=DEFAULT_ISLA_AXIOMATIC_PATH,
    arch=None,
    config=None,
    footprint_config=None,
    model=DEFAULT_MODEL,
    variants=None,
    opt=DEFAULT_OPT,
    dot_dir: pathlib.Path = None,
    generate_latex_only: bool = False,
    runner_config,
) -> int:
    """run isla-axiomatic on the litmus test

    passes output to a .log file in the TMPDIR for this test+model combination
    if verbose then passes through to stdout too.
    """

    # logf: pathlib.Path = TMPDIR / litmus_test.src_path.name / _model_name(model)
    # logf = logf.with_suffix(".log")

    cmd = []

    if runner_config.gdb and not generate_latex_only:
        cmd.extend([
            "rust-gdb",
            "--args",
            f"{isla_path}",
        ])
    else:
        cmd.append(f"{isla_path}")

    cmd.extend([
        f"--arch={arch}",
        f"--config={config}",
        f"--model={model}",
        f"--footprint-config={footprint_config}",
    ])

    if mode == Mode.PGTABLE:
        cmd.append(f"--armv8-page-tables")
    elif mode == Mode.IFETCH:
        cmd.append(f"--ifetch")

    cmd.append(f"--isla-litmus={runner_config.isla_litmus}")
    cmd.append(f"--litmus-translator={runner_config.litmus_translator}")

    if opt:
        cmd.append(opt)

    if runner_config.verbose >= 2:
        cmd.append("--verbose")

    if dot_dir is not None:
        cmd.append(f"--dot={dot_dir}")

    cmd.append(f"--graph={runner_config.graph}")

    if not runner_config.generate_output_model:
        cmd.append("--no-z3-model")

    if generate_latex_only:
        cmd.append(f"--latex={runner_config.latex}")
        runner_config.latex.mkdir(parents=True, exist_ok=True)

    if runner_config.z3_memory is not None:
        cmd.append(f"--memory={runner_config.z3_memory}")

    if variants is not None and variants != []:
        variants = ",".join(variants)
        cmd.append(f"--variant={variants}")

    # for performance
    cmd.append("--check-sat-using")
    cmd.append("(then dt2bv qe simplify solve-eqs bv)")

    # now give it the actual test path

    # ugh: isla-axiomatic version 0.2.0 may or may not have -t, no way to tell!
    # cmd.append(f"-t")
    cmd.append(f"{litmus_test.src_path}")

    cmd.extend(extra)

    env = {"TMPDIR": str(TMPDIR)}
    stdout = collections.deque(maxlen=10)

    async def _passthrough_stream(stream):
        if stream is None:
            return

        nonlocal stdout
        while True:
            line = await stream.readline()

            if line == b"":
                break

            stdout.append(line.decode())

            outf.write(line)
            if not runner_config.batch or runner_config.verbose:
                print(line.decode(), flush=True)

    if runner_config.verbose:
        env_str = " ".join(f"{e}={v}" for (e,v) in env.items())
        cmd_str = shlex.join(cmd)
        print(f"[{dt.datetime.now()}] Running {cmd} with {env=}:\n{env_str} {cmd_str}")

    # stuff os.environ in there too
    env = {**os.environ, **env}

    if (not runner_config.gdb or generate_latex_only) and runner_config.batch:
        STDOUT = asyncio.subprocess.PIPE
    else:
        STDOUT = None

    proc = await asyncio.create_subprocess_exec(*cmd, stdout=STDOUT, env=env)
    CURRENTLY_RUNNING_ISLA_INSTANCES.add(proc)
    try:
        # deliberately do not catch stderr
        # let that fall through to the user.
        await asyncio.gather(_passthrough_stream(proc.stdout), proc.wait())
    except asyncio.CancelledError:
        try:
            proc.kill()
        except:
            pass
    CURRENTLY_RUNNING_ISLA_INSTANCES.remove(proc)
    return proc.returncode

def _model_name(model_path):
    """given a path to a model.toml file
    extract the name of the model
    """
    return model_path.stem

async def _compile_dot(
    dot_path: pathlib.Path,
    *,
    layout_engine="neato",
    dottype="pdf",
    runner_config,
):
    infile = dot_path
    outfile = dot_path.with_suffix(f".{dottype}")
    cmd = [
        "dot",
        f"-K{layout_engine}",
        "-n",
        f"-T{dottype}",
        f"{infile}",
        "-o",
        f"{outfile}",
    ]
    if runner_config.verbose:
        print("Running", shlex.join(cmd))
    proc = await asyncio.create_subprocess_exec(*cmd)
    await proc.wait()

async def _compile_dots(
    dot_dir: pathlib.Path,
    runner_config,
) -> None:
    for f in dot_dir.iterdir():
        if f.suffix == ".dot":
            await _compile_dot(f, runner_config=runner_config)

class LitmusError(Exception):
    pass

class TestFile:
    """ a .litmus or .litmus.toml or @file
    """
    src_path: pathlib.Path

    def __init__(self, src_path: pathlib.Path):
        self.src_path = src_path

    async def run(self, config, outf) -> int:
        extra = []

        for ea in config.extraargs:
            extra.extend(shlex.split(ea))

        if config.default_isla_args:
            extra.extend(shlex.split(config.default_isla_args))

        # run once to generate .tex
        if config.generate_latex:
            await _run_isla(
                self,
                outf=outf,
                mode=config.mode,
                isla_path=config.isla_axiomatic,
                arch=config.arch,
                config=config.config,
                footprint_config=config.footprint_config,
                opt=config.optimize,
                generate_latex_only=True,
                runner_config=config,
            )

        # now try attach any debugging args
        if config.verbose >= 3:
            # linearize output
            extra.extend(["-T", "1"])
            # debugging symbolic evaluation
            debug_args = "lp"
            if config.verbose >= 4:
                debug_args += "m"

            debug_args += "".join(config.debug_args)

            extra.extend(["-D", debug_args])
            probes = []

            if config.arch_version is not None:
                probes.extend(ISLA_DEBUG_PROBES[config.arch_version])

            probes.extend(config.probes)

            for p in probes:
                extra.extend(["--probe", p])

            if config.verbose >= 4:
                extra.extend(["--probe", "AArch64_TranslateAddress"])
                extra.extend(["--probe", "AArch64_S1Translate"])
                extra.extend(["--probe", "AArch64_S2Translate"])

            # for debugging the model
            extra.extend(["--graph-show-all-reads"])
            extra.extend(["--graph-shows=po,rf,co,rfi,trf,tfr,fr,bob,ctxob,dob,iio,tob,addr,data,ctrl"])

        # run again to get the actual results
        res = await _run_isla(
            self,
            outf=outf,
            mode=config.mode,
            isla_path=config.isla_axiomatic,
            arch=config.arch,
            config=config.config,
            footprint_config=config.footprint_config,
            model=config.model,
            variants=config.variant,
            opt=config.optimize,
            dot_dir=config.dot,
            runner_config=config,
            *extra,
        )

        return res

@contextlib.contextmanager
def change_dir(new_dir):
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        yield
    finally:
        os.chdir(old_dir)

def _make_tarball(args, tmpdir, nightly_tmp_dirname):
    """ Create a tarball out of the `tmpdir`/`nightly_tmp_dirname` directory and stick it in the current dir
    """
    tarname = nightly_tmp_dirname + ".tar.gz"

    if args.verbose:
        print(f"[{dt.datetime.now()}] zipping up tarball {tarname!r}")
    tar = tarfile.open(tarname, "w:gz")
    with change_dir(tmpdir):
        tar.add(nightly_tmp_dirname)
    tar.close()

def _nightly_dirname():
    """ Return the currently nightly dirname
    """
    now = dt.datetime.now().strftime("%Y-%m-%d")
    return "nightly-{}".format(now)

class Runner:
    def __init__(self):
        self._tasks = []
        self._attempting_cancel = False

    def cancel_all(self):
        if not self._attempting_cancel:
            print("Interrupted !", file=sys.stderr)
            print("... stopping jobs gracefully", file=sys.stderr)
            print("[^C again to force quit]", file=sys.stderr)
            self._attempting_cancel = True
            for t in self._tasks:
                t.cancel()
        else:
            print("OK force quitting")
            for proc in CURRENTLY_RUNNING_ISLA_INSTANCES:
                try:
                    proc.kill()
                except:
                    pass

            sys.exit(1)

    def _mk_model_results_filename(self, nightly_tmp_dir, model_name, variants=None):
        fname = model_name
        if variants is not None and variants != []:
            variants = "_".join(variants)
            fname = f"{fname}_{variants}"
        return (nightly_tmp_dir / fname).with_suffix(".txt")

    async def _run(self, test: TestFile, args, log_prefix):
        for model, variants in zip(args.models, args.variants):
            outf_path = (log_prefix / ("_".join(variants))).with_suffix(".log")
            args.model = model
            args.variant = variants
            outf_path.parent.mkdir(exist_ok=True, parents=True)
            with outf_path.open("w") as f:
                await test.run(args, f)

        if args.dot is not None:
            await _compile_dots(dot_dir=args.dot, runner_config=args)

    async def run_tests(self, args) -> None:
        test_file = TestFile(args.test_file)

        log_path: pathlib.Path = TMPDIR / test_file.src_path.name
        log_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # if `make run`
            # then just print summary table
            if args.command == "run":
                log_path_prefix: pathlib.Path = TMPDIR / test_file.src_path.name

                if args.dot and not args.dot.exists():
                    args.dot.mkdir(parents=True, exist_ok=True)

                results = await self._run(test_file, args, log_path_prefix)

            # if nightly build then collect up results into tarball
            elif args.command == "nightly":
                nightly_tmp_dirname = args.out

                nightly_tmp = TMPDIR / nightly_tmp_dirname
                nightly_tmp.mkdir(parents=True)  # if exists, raise error.

                # these don't need to be model specific
                # the tex files are always generated the same
                # and the allowed execution should be the same so long as candidate numbering is deterministic
                nightly_tmp_tex_dir = nightly_tmp / "tex"
                nightly_tmp_tex_dir.mkdir()

                nightly_tmp_dots_dir = nightly_tmp / "dots"
                nightly_tmp_dots_dir.mkdir()

                args.dot = nightly_tmp_dots_dir
                args.latex = nightly_tmp_tex_dir

                # touch all the model results files
                for model, variants in zip(args.models, args.variants):
                    model_name = _model_name(model)
                    model_results_path = self._mk_model_results_filename(nightly_tmp, model_name, variants)
                    model_results_path.write_text("")

                model_results_path = nightly_tmp / model_name
                log_path_prefix: pathlib.Path = TMPDIR / test_file.src_path.name
                results = await self._run(test_file, args, log_path_prefix)

                # async for model, variants, result in results:
                #     model_name = _model_name(model)
                #     model_results_path = self._mk_model_results_filename(nightly_tmp, model_name, variants)
                #     with open(model_results_path, "a", newline="") as f:
                #         writer = csv.writer(f)
                #         writer.writerow([f"{test.name}", f"{result.name}"])

                _make_tarball(args, TMPDIR, nightly_tmp_dirname)
            elif args.command == "tar":
                _make_tarball(args, TMPDIR, args.nightly)

        except asyncio.CancelledError:
            print("! interrupted", file=sys.stderr)

    async def run(self, args) -> int:
        # ensure TMPDIR exists
        TMPDIR.mkdir(parents=True, exist_ok=True)

        # collect all the litmus tests we were given
        run_task = asyncio.create_task(self.run_tests(args))
        self._tasks.extend([run_task])

        # let ^C cancel them
        asyncio.get_event_loop().add_signal_handler(signal.SIGINT, self.cancel_all)

        # let everything run and wait for it to finish
        await run_task
        return 0

def _add_common_args(parser):
    # shared configuration

    # can pass --arch and --config instead to override default --arch-version
    parser.add_argument("--arch-version", metavar="VERSION", default=None, help=f"architecture version to use (default: {DEFAULT_ISLA_VERSION})")
    parser.add_argument("--arch", metavar="PATH", help=f"override architecture snapshot to pass to isla-axiomatic")
    parser.add_argument("--config", metavar="PATH", default=None, help=f"override config to pass to isla-axiomatic")
    parser.add_argument("--footprint-config", metavar="PATH", default=None, help=f"override config to pass to isla-axiomatic to use for footprint analysis")

    parser.add_argument("--isla-axiomatic", metavar="PATH", default=DEFAULT_ISLA_AXIOMATIC_PATH, type=pathlib.Path, help=f"path to isla-axiomatic (default: {DEFAULT_ISLA_AXIOMATIC_PATH})")
    parser.add_argument("--isla-litmus", metavar="PATH", default=DEFAULT_ISLA_LITMUS, type=pathlib.Path, help=f"path to isla-litmus (default: {DEFAULT_ISLA_LITMUS})")
    parser.add_argument("--litmus-translator", metavar="PATH", default=DEFAULT_LITMUS_TRANSLATOR, type=pathlib.Path, help=f"path to litmus-translator (default: {DEFAULT_LITMUS_TRANSLATOR})")

    parser.add_argument("--ifetch", dest="mode", action="store_const", const=Mode.IFETCH, help="Run isla-axiomatic with ifetch mode enabled")
    parser.add_argument("--pgtable", dest="mode", action="store_const", const=Mode.PGTABLE, help="Run isla-axiomatic with translation table walks enabled")

    # enable/disable isla-axiomatic optimizations
    parser.add_argument("--optimize", metavar="ARG", default=DEFAULT_OPT, help=f"command to pass to isla-axiomatic (default: {DEFAULT_OPT!r})")
    parser.add_argument("--no-optimize", dest="optimize", action="store_const", const=None)
    parser.add_argument("--probe", dest="probes", metavar="PROBE", action="append", help=f"additional --probe commands to isla")
    parser.add_argument("--debug-args", metavar="D", nargs="*", default=[], help=f"additional -D commands to isla")
    parser.add_argument("--default-isla-args", default="")
    parser.add_argument("--no-default-isla-args", dest="default-args", action="store_const", const="")
    parser.add_argument("--extraargs", help="extra arguments to pass directly to `isla-axiomatic`", action="append", default=[])
    parser.add_argument("--no-output-model", dest="generate_output_model", action="store_false", help=f"do not generate output model")

    parser.add_argument("--generate-latex", action="store_true", default=True)
    parser.add_argument("--no-generate-latex", dest="generate_latex", action="store_false")
    parser.add_argument("--latex", metavar="PATH", default=DEFAULT_TEX_DIR, type=pathlib.Path, help=f"directory to write the LaTeX source listing to (default: {DEFAULT_TEX_DIR})")

    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--z3-memory", metavar="N", default=None)
    parser.add_argument("--gdb", action="store_true", default=False)

    parser.add_argument("--allow-bad-names", action="store_true", default=False, help="Allow mismatched names between file and [name] attribute")

    parser.add_argument("--tmpdir", metavar="PATH", default=None, type=pathlib.Path, help=f"$TMPDIR override (current: {TMPDIR})")

    parser.add_argument("--batch", action="store_true", default=False, help="Run in batch mode")


def fail(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


def main(argv=None) -> int:
    # it's ok, it's a super-global environment var anyway
    global TMPDIR

    parser = argparse.ArgumentParser(__doc__)

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run litmus tests")
    _add_common_args(run_parser)
    run_parser.add_argument("--model", metavar="PATH", dest="models", nargs="+", type=pathlib.Path, default=None, help=f"model to pass to isla-axiomatic (default: {DEFAULT_MODEL})")
    run_parser.add_argument("--variant", metavar="VARIANT", dest="variants", default=None, action="append")
    run_parser.add_argument("--dot", metavar="PATH", default=DEFAULT_DOT_DIR, type=pathlib.Path, help=f"directory to store generated graphs (default: {DEFAULT_DOT_DIR})")
    run_parser.add_argument("--no-graph", dest="dot", action="store_const", const=None, help=f"do not generate graph")
    run_parser.add_argument("--graph", action="store", default="ascii", choices=["ascii", "dot", "none"])

    # tests passed positionally
    run_parser.add_argument("test_file", metavar="TEST_PATH", type=pathlib.Path, help="path to .litmus or .litmus.toml or @all file to run")

    nightly_parser = subparsers.add_parser("nightly", help="Run a batched nightly run")
    _add_common_args(nightly_parser)

    nightly_parser.add_argument("test_file", metavar="TEST_PATH", default=None, type=pathlib.Path, help="path to single @file with names of all tests to run")
    nightly_parser.add_argument("--models", metavar="MODEL_PATH", dest="models", default=None, type=pathlib.Path, nargs="+")
    nightly_parser.add_argument("--variants", metavar="VARIANT", dest="variants", default=None, nargs="+", help="list of variant names for each model")
    nightly_parser.add_argument("-out", metavar="DIRNAME", default=_nightly_dirname(), help="name of the nightly dir/tarball (default: nightly-YYYY-MM-DD)")

    tar_parser = subparsers.add_parser("tar", help="Make a nightly tarball")
    _add_common_args(tar_parser)
    tar_parser.add_argument("nightly", metavar="DIRNAME", default=_nightly_dirname(), help="name of the nightly dir (default: nightly-YYYY-MM-DD)")

    args = parser.parse_args(argv)

    # collect comma-separated probes
    if args.probes is None:
        args.probes = []
    else:
        args.probes = [p for pr in args.probes for p in pr.split(",")]

    if args.command is None:
        parser.print_usage(sys.stderr)
        fail("error: one of {run,nightly,tar} required")

    # substitute some mode-specific defaults
    if args.mode is None:
        args.mode = Mode.DATA
    if hasattr(args, "model") and args.model is None:
        args.model = _default_model[args.mode]
    if hasattr(args, "models") and args.models is None:
        args.models = [_default_model[args.mode]]
    if not hasattr(args, "variants") or args.variants is None:
        args.variants = [[]]*len(args.models)
    else:
        if len(args.variants) != len(args.models):
            fail("error: length of --variants must match number of --models")
        # parse --variants A B C,D into [[A], [B], [C,D]]
        args.variants = [v.split(",") for v in args.variants]
        # normalise [["None"]] to [[]]
        for mi, vs in enumerate(args.variants):
            if vs == ["None"]:
                args.variants[mi] = []

    # --arch-version and --arch/--config are mutually exclusive
    if args.arch_version is not None and (args.arch is not None or args.config is not None):
        fail("error: --arch and --config are mutually exclusive with --arch-version")
    elif args.arch_version is None and ((args.arch is not None) ^ (args.config is not None)):
        fail("error: both --arch and --config must be supplied, or neither.")

    if "graph" not in args:
        args.graph = "none"

    # if no --arch or --config, then use --arch-version
    if args.arch is None and args.config is None:
        if args.arch_version is None:
            args.arch_version = DEFAULT_ISLA_VERSION

        if args.mode == Mode.PGTABLE:
            args.arch_version += "-mmu"

        args.arch = ISLA_SNAPSHOT_VERSIONS[args.arch_version]
        args.config = ISLA_CONFIG_VERSIONS[args.arch_version]
        args.footprint_config = ISLA_FOOTPRINT_CONFIG_VERSIONS[args.arch_version]

    if TMPDIR is None:
        if args.tmpdir is not None:
            TMPDIR = args.tmpdir
        else:
            TMPDIR = _default_tmpdir[args.mode]

    runner = Runner()
    try:
        return asyncio.run(runner.run(args))
    except LitmusError as le:
        print(le)
        return 1

if __name__ == "__main__":
    sys.exit(main())
