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
DEFAULT_ARCH = REMS_DIR / "isla-snapshots" / "armv9.ir"
DEFAULT_CONFIG = ISLA_DIR / "configs" / "armv9.toml"
DEFAULT_CONFIG_PGTABLE = ISLA_DIR / "configs" / "armv9_mmu_on.toml"
DEFAULT_MODEL = REMS_DIR / "system-semantics-arm-axiomatic-models" / "models" / "aarch64_base.cat"
DEFAULT_MODEL_PGTABLE = REMS_DIR / "system-semantics-arm-axiomatic-models" / "models" / "aarch64_mmu_strong_ETS.cat"
DEFAULT_MODEL_IFETCH = REMS_DIR / "system-semantics-arm-axiomatic-models" / "models" / "aarch64_ifetch.cat"
DEFAULT_FOOTPRINT = ISLA_DIR / "configs" / "armv9.toml"
DEFAULT_ISLA_LITMUS = ISLA_DIR / "isla-litmus" / "isla-litmus"
DEFAULT_LITMUS_TRANSLATOR = HERDTOOLS_DIR / "_build" / "install" / "default" / "bin" / "litmus-translator"

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

_default_config = {
    Mode.DATA: DEFAULT_CONFIG,
    Mode.IFETCH: DEFAULT_CONFIG,
    Mode.PGTABLE: DEFAULT_CONFIG_PGTABLE,
}

_default_tmpdir = {
    Mode.DATA: HERE / "_tmp",
    Mode.IFETCH: HERE / "_tmp",
    Mode.PGTABLE: HERE / "_tmp_pgtable",
}

async def _run_isla(
    litmus_test: "LitmusTest",
    *extra,
    mode=DEFAULT_MODE,
    isla_path=DEFAULT_ISLA_AXIOMATIC_PATH,
    arch=DEFAULT_ARCH,
    config=DEFAULT_CONFIG,
    footprint_config=DEFAULT_FOOTPRINT,
    model=DEFAULT_MODEL,
    variants=None,
    opt=DEFAULT_OPT,
    dot_dir: pathlib.Path = None,
    generate_latex_only: bool = False,
    runner_config,
) -> "Result | None":
    """run isla-axiomatic on the litmus test

    passes output to a .log file in the TMPDIR for this test+model combination
    if verbose then passes through to stdout too.
    """

    logf: pathlib.Path = TMPDIR / litmus_test.name / _model_name(model)
    logf = logf.with_suffix(".log")

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
        logf.parent.mkdir(parents=True, exist_ok=True)
        with logf.open("wb") as f:
            while True:
                line = await stream.readline()

                if line == b"":
                    break

                stdout.append(line.decode())

                f.write(line)
                if runner_config.verbose:
                    print(line.decode())

    if runner_config.verbose:
        env_str = " ".join(f"{e}={v}" for (e,v) in env.items())
        cmd_str = shlex.join(cmd)
        print(f"[{dt.datetime.now()}] Running {cmd} with {env=}:\n{env_str} {cmd_str}")

    # stuff os.environ in there too
    env = {**os.environ, **env}

    if not runner_config.gdb or generate_latex_only:
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
    if generate_latex_only:
        return None
    else:
        return Result.from_isla_stdout_postfix(proc.returncode, stdout)

def _model_name(model_path):
    """given a path to a model.toml file
    extract the name of the model
    """
    return model_path.stem

class Result(enum.Enum):
    Allow = enum.auto()
    Forbid = enum.auto()
    Error = enum.auto()
    IslaCrash = enum.auto()
    IslaEmpty = enum.auto()

    @classmethod
    def from_str(cls, s):
        if s == "allowed":
            return cls.Allow
        elif s == "forbidden":
            return cls.Forbid
        elif s == "error":
            return cls.Error
        else:
            raise ValueError(f"Unknown result state {s!r}")

    @classmethod
    def from_isla_stdout_postfix(cls, returncode, stdout_postfix: "list[str]"):
        if returncode != 0:
            return cls.IslaCrash
        elif stdout_postfix:
            final_line = stdout_postfix[-1]

            name, state, *_ = final_line.split(" ")
            return cls.from_str(state)
        else:
            return cls.IslaEmpty

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
    litmus_test: "LitmusTest",
    dot_dir: pathlib.Path,
    runner_config,
) -> None:
    for f in dot_dir.iterdir():
        m = re.fullmatch(rf"{re.escape(litmus_test.name)}\_allow\_\d+.dot", f.name)
        if m is not None:
            await _compile_dot(f, runner_config=runner_config)

class LitmusError(Exception):
    pass

class LitmusTest:
    name: str
    src_path: pathlib.Path

    def __init__(self, name: str, src_path: pathlib.Path):
        self.name = name
        self.src_path = src_path

    @classmethod
    def from_path(cls, p, allow_bad_names=False):
        """ check the path `p` exists and is the right format before creating a `LitmusTest` object.
        """
        if not p.exists():
            raise LitmusError(f"Test {p} does not exist")

        required_suffixes = [".litmus.toml", ".litmus"]
        for suffix in required_suffixes:
            if p.name[-len(suffix):] == suffix:
                break
        else:
            raise LitmusError(
                f"Test {p} does not appear to be one of: {required_suffixes}"
            )

        test_name_from_filename = p.name[:-len(suffix)]

        # sanity check the name matches the name without the .litmus.toml
        # instead of depending on toml just read that name="..." field
        if suffix == ".litmus.toml":
            content = p.read_text()
            m = re.search(r"name\s*=\s*[\'\"](?P<name>.*)[\'\"]", content)
            if m is None:
                raise LitmusError(f"Could not find [name] for test {p}")

            test_name_from_toml = m["name"]

            if test_name_from_filename != test_name_from_toml:
                msg = f"Test {p} has different name {test_name_from_toml!r} in file"
                if not allow_bad_names:
                    raise LitmusError(f"{msg} (pass --allow-bad-names to continue anyway)")
                else:
                    print(msg, file=sys.stderr)

        return cls(test_name_from_filename, p)

    @classmethod
    def from_collection_path(cls, p, allow_bad_names=False):
        content = p.read_text()
        for line in content.splitlines():
            if line.strip():
                yield from cls.tests_from_test_file(p.parent / line, allow_bad_names=allow_bad_names)

    @classmethod
    def tests_from_test_file(cls, test_path, allow_bad_names=False):
        if test_path.stem[0] == "@":
            yield from cls.from_collection_path(test_path, allow_bad_names=allow_bad_names)
        else:
            try:
                yield cls.from_path(test_path, allow_bad_names=allow_bad_names)
            except LitmusError as e:
                print(e, file=sys.stderr)

    async def run(self, config) -> Result:
        extra = []

        for ea in config.extraargs:
            extra.extend(shlex.split(ea))

        if config.default_isla_args:
            extra.extend(shlex.split(config.default_isla_args))

        # run once to generate .tex
        if config.generate_latex:
            await _run_isla(
                self,
                mode=config.mode,
                isla_path=config.isla_axiomatic,
                arch=config.arch,
                config=config.config,
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
            probes = [
                "__FetchNextInstr",
                "BranchTo",
                "AArch64_TakeException",
                "PC_read",
                "__ExecuteInstr",
                *config.probes,
            ]

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
            mode=config.mode,
            isla_path=config.isla_axiomatic,
            arch=config.arch,
            config=config.config,
            model=config.model,
            variants=config.variant,
            opt=config.optimize,
            dot_dir=config.dot,
            runner_config=config,
            *extra,
        )

        if config.dot is not None:
            await _compile_dots(self, dot_dir=config.dot, runner_config=config)

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

    async def _run_all(self, tests, args):
        for model, variants in zip(args.models, args.variants):
            for t in tests:
                args.model = model
                args.variant = variants
                res = await t.run(args)
                yield (model, variants, t, res)

    async def run_tests(self, args) -> None:
        try:
            # if `make run`
            # then just print summary table
            if args.command == "run":
                if args.dot and not args.dot.exists():
                    args.dot.mkdir(parents=True, exist_ok=True)

                tests = []
                for test_path in args.tests:
                    tests.extend(LitmusTest.tests_from_test_file(test_path, allow_bad_names=args.allow_bad_names))
                results = self._run_all(tests, args)

                async for model, variants, test, result in results:
                    if variants:
                        variants = ",".join(variants)
                        print(f"{model}({variants})\t{test.name},{result.name}")
                    else:
                        print(f"{model}\t{test.name},{result.name}")
            # if nightly build then collect up results into tarball
            elif args.command == "nightly":
                nightly_tmp_dirname = args.out

                # collect tests
                tests = []
                for test_path in args.tests:
                    tests.extend(LitmusTest.tests_from_test_file(test_path, allow_bad_names=args.allow_bad_names))

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

                results = self._run_all(tests, args)

                async for model, variants, test, result in results:
                    model_name = _model_name(model)
                    model_results_path = self._mk_model_results_filename(nightly_tmp, model_name, variants)
                    with open(model_results_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([f"{test.name}", f"{result.name}"])

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
    parser.add_argument("--config", metavar="PATH", default=None, help=f"config to pass to isla-axiomatic (default: {DEFAULT_CONFIG})")
    parser.add_argument("--arch", metavar="PATH", default=DEFAULT_ARCH, help=f"arch to pass to isla-axiomatic (default: {DEFAULT_ARCH})")
    parser.add_argument("--isla-axiomatic", metavar="PATH", default=DEFAULT_ISLA_AXIOMATIC_PATH, type=pathlib.Path, help=f"path to isla-axiomatic (default: {DEFAULT_ISLA_AXIOMATIC_PATH})")
    parser.add_argument("--isla-litmus", metavar="PATH", default=DEFAULT_ISLA_LITMUS, type=pathlib.Path, help=f"path to isla-litmus (default: {DEFAULT_ISLA_LITMUS})")
    parser.add_argument("--litmus-translator", metavar="PATH", default=DEFAULT_LITMUS_TRANSLATOR, type=pathlib.Path, help=f"path to litmus-translator (default: {DEFAULT_LITMUS_TRANSLATOR})")

    parser.add_argument("--ifetch", dest="mode", action="store_const", const=Mode.IFETCH, help="Run isla-axiomatic with ifetch mode enabled")
    parser.add_argument("--pgtable", dest="mode", action="store_const", const=Mode.PGTABLE, help="Run isla-axiomatic with translation table walks enabled")

    # enable/disable isla-axiomatic optimizations
    parser.add_argument("--optimize", metavar="ARG", default=DEFAULT_OPT, help=f"command to pass to isla-axiomatic (default: {DEFAULT_OPT!r})")
    parser.add_argument("--no-optimize", dest="optimize", action="store_const", const=None)
    parser.add_argument("--probes", metavar="PROBES", nargs="*", default=[], help=f"additional --probe commands to isla")
    parser.add_argument("--debug-args", metavar="D", nargs="*", default=[], help=f"additional -D commands to isla")
    parser.add_argument("--default-isla-args", default="--graph-human-readable")
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

    # tests passed positionally
    run_parser.add_argument("tests", metavar="TEST_PATH", type=pathlib.Path, nargs="*", help="paths to .litmus or .litmus.toml files to run")

    nightly_parser = subparsers.add_parser("nightly", help="Run a batched nightly run")
    _add_common_args(nightly_parser)

    nightly_parser.add_argument("tests", metavar="TEST_PATH", default=None, type=pathlib.Path, nargs=1, help="path to single @file with names of all tests to run")
    nightly_parser.add_argument("--models", metavar="MODEL_PATH", dest="models", default=None, type=pathlib.Path, nargs="+")
    nightly_parser.add_argument("--variants", metavar="VARIANT", dest="variants", default=None, nargs="+", help="list of variant names for each model")
    nightly_parser.add_argument("-out", metavar="DIRNAME", default=_nightly_dirname(), help="name of the nightly dir/tarball (default: nightly-YYYY-MM-DD)")

    tar_parser = subparsers.add_parser("tar", help="Make a nightly tarball")
    _add_common_args(tar_parser)
    tar_parser.add_argument("nightly", metavar="DIRNAME", default=_nightly_dirname(), help="name of the nightly dir (default: nightly-YYYY-MM-DD)")

    args = parser.parse_args(argv)

    # collect comma-separated probes
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
    if args.config is None:
        args.config = _default_config[args.mode]
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
