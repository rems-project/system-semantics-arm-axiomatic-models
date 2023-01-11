# assume system-semantics-arm-confidential/ repo in same dir as isla/
ISLAROOT = ../../isla
DOTDIR = ./dots
BINDIR = ./bin
TEXDIR = ./tex

SHELL := /bin/bash

DRAW_GRAPHS = 1

ISLABIN = $(ISLAROOT)/target/release

ISLA = $(ISLABIN)/isla
ISLA_AXIOMATIC = $(ISLABIN)/isla-axiomatic

LITMUS_TESTS_DIR = ../../litmus-tests-armv8a-system-vmsa/

MODEL_DIR = ../../systems-isla-tests/models

export TMPDIR = ./_tmp
export PATH := $(ISLABIN):$(ISLAROOT)/isla-litmus/:$(PATH)
export LD_LIBRARY_PATH := $(ISLAROOT):$(LD_LIBRARY_PATH)

MODEL = $(MODEL_DIR)/aarch64_mmu_strong_ETS.cat
ARCH = ../../isla-snapshots/aarch64.ir
CONFIG = $(ISLAROOT)/configs/aarch64_mmu_on.toml
AARCH64_CONFIG = $(ISLAROOT)/configs/aarch64.toml
EXTRAARGS =
ISLAARGS_MMU = \
	--arch=$(ARCH) --config=$(CONFIG) \
	--model=$(MODEL) \
	--armv8-page-tables \
	--footprint-config=$(AARCH64_CONFIG) \
	--check-sat-using "(then dt2bv qe simplify solve-eqs bv)" \
	$(EXTRAARGS)

ISLAARGS_NOMMU = \
	--arch=$(ARCH) --config=$(AARCH64_CONFIG) \
	--model=$(MODEL) \
	--footprint-config=$(AARCH64_CONFIG) \
	$(EXTRAARGS)

ISLAARGS = $(ISLARGS_MMU)

.PHONY: FORCE

define run-isla
	$(ISLA_AXIOMATIC) $(ISLAARGS_MMU) $(if $(DRAW_GRAPHS),--dot $(DOTDIR)/$(basename  $(notdir $(TEST)))/,) -t $1
endef

.PHONY: run
run: check
	@mkdir -p $(DOTDIR)/$(basename $(notdir $(TEST)))/
	$(call run-isla,$(TEST))

NO_CHECK_Z3 =

# `make _tmp/results/CoWTf.inv+po/` will run isla-axiomatic to generate:
# 	results from running on $(MODEL)
#	LaTeX sources of the test listing
#	graphviz .dot output of all executions
#		and run neato to generate .pdfs
$(TMPDIR)/results/%/: $(LITMUS_TESTS_DIR)/tests/pgtable/HAND/%.litmus.toml
	mkdir -p $@
	$(MAKE) -s run TEST=$< EXTRAARGS="--latex=$@"
	$(MAKE) -s run TEST=$< EXTRAARGS="--remove-uninteresting safe" DOTDIR=$@ MODEL=$(MODEL_DIR)/aarch64_mmu_no_axioms.cat
	$(MAKE) -s run TEST=$< EXTRAARGS="--remove-uninteresting safe" DRAW_GRAPHS=0 > $@/result_$(basename $(notdir $(MODEL)))
	for f in $@$*.litmus/*.dot; do neato -Tpdf -n $$f -o $$f.pdf; done

# check for known old z3 versions
# namely the one on legion ...
z3_version = $(shell z3 --version)
z3_check = $(shell z3 --version | grep -e '4\.4\..')
check:
ifeq ($(NO_CHECK_Z3),)
ifneq ($(z3_check),)
	$(info Detected z3 version '$(z3_version)' which was too old,  isla requires 4.8+)
	$(info Run 'make NO_CHECK_Z3=1 [...]' to use anyway)
	$(error failed check)
endif
endif
.PHONY: check

example: TEST=$(LITMUS_TEST_DIR)/tests/pgtable/HAND/CoWinvT+po.litmus.toml
example: EXTRAARGS=--remove-uninteresting safe
example: run dots
	@echo See $(BINDIR)/CoWinvT+po/*.pdf
.PHONY: example

example-with-debug: TEST=$(LITMUS_TEST_DIR)/tests/pgtable/HAND/CoWinvT+po.litmus.toml
example-with-debug: EXTRAARGS=--remove-uninteresting safe --verbose --debug p --probe AArch64_TakeException --probe __fetchA64 --probe AArch64_ExceptionReturn
example-with-debug: run dots
	@echo See $(BINDIR)/CoWinvT+po/*.pdf
.PHONY: example-with-debug


DOTTYPE = pdf
DOTFILES = $(wildcard $(DOTDIR)/**/*.dot)
GRAPHFILES = $(patsubst $(DOTDIR)/%.dot,$(BINDIR)/%.$(DOTTYPE),$(DOTFILES))

.PHONY: dots
dots: $(GRAPHFILES)
LAYOUT_ENGINE = neato

$(BINDIR)/%.$(DOTTYPE): $(DOTDIR)/%.dot
	@mkdir -p $(dir $@)
	dot -K$(LAYOUT_ENGINE) -n  -T$(DOTTYPE) $< -o $@

TESTS = $(wildcard ../../systems-isla-tests/tests/pgtable/*.toml)
PKVM_TESTS = $(wildcard ../../systems-isla-tests/tests/pgtable/pKVM/*.toml)

MAX_WORKERS = 4

results_allow_all.txt:
	python3 ./collect_results.py ../../systems-isla-tests/models/aarch64_mmu_no_axioms.cat --outfile=$@ --draw-graphs --extraargs="--graph-padding='instr-down=1'" --testlist='$(TESTS)' --max-workers=$(MAX_WORKERS) --tmpdir=$(TMPDIR)/all/
.PRECIOUS: results_allow_all.txt

results_strong.txt:
	python3 ./collect_results.py ../../systems-isla-tests/models/aarch64_mmu_strong.cat --outfile=$@ --testlist='$(TESTS)' --max-workers=$(MAX_WORKERS) --tmpdir=$(TMPDIR)/strong/
.PRECIOUS: results_strong.txt

results_ETS.txt:
	python3 ./collect_results.py ../../systems-isla-tests/models/aarch64_mmu_strong_ETS.cat --outfile=$@ --testlist='$(TESTS)' --max-workers=$(MAX_WORKERS) --tmpdir=$(TMPDIR)/ets/
.PRECIOUS: results_ETS.txt

results_pkvm.txt:
	python3 ./collect_results.py ../../systems-isla-tests/models/aarch64_mmu_strong.cat --outfile=$@ --extraargs="--merge-split-stages" --testlist='$(PKVM_TESTS)' --max-workers=$(MAX_WORKERS) --tmpdir=$(TMPDIR)/pkvm/
.PRECIOUS: results_pKVM.txt

results_pkvm_allow.txt:
	python3 ./collect_results.py ../../systems-isla-tests/models/aarch64_mmu_no_axioms.cat --outfile=$@ --extraargs="--merge-split-stages" --testlist='$(PKVM_TESTS)' --max-workers=$(MAX_WORKERS) --tmpdir=$(TMPDIR)/pkvmall/
.PRECIOUS: results_pkvm_allow.txt

tex.txt:
	@mkdir -p $(TMPDIR)/latex/
	python3 ./collect_results.py ../../systems-isla-tests/models/aarch64_mmu_no_axioms.cat --outfile=$@ --extraargs="--latex=$(TMPDIR)/latex" --testlist='$(TESTS)' --max-workers=$(MAX_WORKERS) --tmpdir=$(TMPDIR)/litmustex/
.PRECIOUS: tex.txt

run_all:
	@echo Building LaTeX test source...
	@mkdir -p $(TMPDIR)/latex/
	$(MAKE) tex.txt MAX_WORKERS=$(MAX_WORKERS) TESTS='$(TESTS)'
	@echo Generating diagrams for all tests...
	$(MAKE) results_allow_all.txt MAX_WORKERS=$(MAX_WORKERS) TESTS='$(TESTS)'
	@echo Computing allow/forbid with strong model
	$(MAKE) results_strong.txt MAX_WORKERS=$(MAX_WORKERS) TESTS='$(TESTS)'
	@echo Computing allow/forbid with ETS model
	$(MAKE) results_ETS.txt MAX_WORKERS=$(MAX_WORKERS) TESTS='$(TESTS)'
	@echo Computing pKVM test results
	$(MAKE) results_pkvm.txt MAX_WORKERS=$(MAX_WORKERS) TESTS='$(TESTS)'
	@echo Computing pKVM diagrams
	$(MAKE) results_pkvm.txt MAX_WORKERS=$(MAX_WORKERS) TESTS='$(TESTS)'
	@echo Packaging up results into results/$(curdate)/
	$(MAKE) package_current_results
.PHONY: run_all

curdate = $(shell date '+%Y-%m-%d')
package_current_results:
	mkdir -p results/$(curdate)/
	cp results*.txt results/$(curdate)
	cp -r dots/ results/$(curdate)
	mkdir -p results/$(curdate)/latex/
	cp -f $(TMPDIR)/latex/* results/$(curdate)/latex/
	cd results && tar -czvf $(curdate).tar.gz $(curdate)/

.PHONY: clean_dots
clean_dots:
	rm -rf bin/*
	rm -rf dots/*

.PHONY: clean
clean: clean_dots
	rm -f results_allow_all.txt
	rm -f results_strong.txt
	rm -f results_ETS.txt
	rm -f results_pkvm.txt
	rm -f tex.txt

.PHONY: clean_temp
clean_temp:
	rm -rf $(TMPDIR)/
