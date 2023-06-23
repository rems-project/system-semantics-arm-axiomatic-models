isla-axiomatic machinery
========================

Some scripts to help drive isla-axiomatic for a variety of Arm models

Examples
--------

Assuming:
- z3 installed and in $PATH
- a directory structure like:
    parent dir/
    |- litmus-tests/
       |- litmus-tests-armv8a-system/
       |- litmus-tests-armv8a-system-vmsa/
    |- rems/
       |- isla/
          |- isla-litmus*
          |- target/release/isla-axiomatic*
       |- isla-snapshots/
       |- system-semantics-arm-axiomatic-models/

*binary files produced by running `make` and `make -C isla-litmus` in isla/

To run a normal 'data' memory test (substitute path with your path to armv8 litmus tests):
$ ./run.sh ../../../litmus-tests/litmus-tests-armv8a-private/tests/non-mixed-size/HAND/MP+dmb.sy+ctrl.litmus

To run an instruction-fetching test:
$ ./run_ifetch.sh ../../../litmus-tests/litmus-tests-armv8a-system/tests/ifetch/HAND/SM.litmus

To run a pagetable test
$ ./run_pgtable.sh ../../../litmus-tests/litmus-tests-armv8a-system-vmsa/tests/pgtable/HAND/CoWTf.inv+po.litmus.toml

Then check out the dots/ and tex/ directories in isla_machinery/ to see generated graphviz diagrams and LaTeX source listings.

Full invocation
---------------

The above examples generate the following isla-axiomatic invocations (should be copy/paste'able from here):

```
TMPDIR=./_tmp ../../isla/target/release/isla-axiomatic \
    --arch=../../isla-snapshots/armv9.ir \
    --config=../../isla/configs/armv9.toml \
    --model=../../system-semantics-arm-axiomatic-models/models/aarch64_base.cat \
    --footprint-config=../../isla/configs/armv9.toml \
    --isla-litmus=../../isla/isla-litmus/isla-litmus \
    --remove-uninteresting=safe \
    --dot=./dots \
    --check-sat-using '(then dt2bv qe simplify solve-eqs bv)' \
    --graph-human-readable \
    ../../../litmus-tests/litmus-tests-armv8a-private/tests/non-mixed-size/HAND/MP+dmb.sy+ctrl.litmus
```

```
TMPDIR=./_tmp ../../isla/target/release/isla-axiomatic \
    --arch=../../isla-snapshots/armv9.ir \
    --config=../../isla/configs/armv9.toml \
    --model=../../system-semantics-arm-axiomatic-models/models/aarch64_base.cat \
    --footprint-config=../../isla/configs/armv9.toml \
    --ifetch \
    --isla-litmus=../../isla/isla-litmus/isla-litmus \
    --remove-uninteresting=safe \
    --dot=./dots \
    --check-sat-using '(then dt2bv qe simplify solve-eqs bv)' \
    --graph-human-readable \
    ../../../litmus-tests/litmus-tests-armv8a-system/tests/ifetch/HAND/SM.litmus
```

```
TMPDIR=./_tmp_pgtable ../../isla/target/release/isla-axiomatic \
    --arch=../../isla-snapshots/armv9.ir \
    --config=../../isla/configs/armv9_mmu_on.toml \
    --model=../../system-semantics-arm-axiomatic-models/models/aarch64_mmu_strong_ETS.cat \
    --footprint-config=../../isla/configs/armv9.toml \
    --armv8-page-tables \
    --isla-litmus=../../isla/isla-litmus/isla-litmus \
    --remove-uninteresting=safe \
    --dot=./dots \
    --check-sat-using '(then dt2bv qe simplify solve-eqs bv)' \
    --graph-human-readable \
    ../../../litmus-tests/litmus-tests-armv8a-system-vmsa/tests/pgtable/HAND/CoWTf.inv+po.litmus.toml
```