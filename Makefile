# Reproduce every number and figure in README.md.
#
#   make test       regression tests (~10 s)
#   make results    run the four MG5-free analysis scripts (~8 min, mostly the pull study)
#   make figures    refresh the plots README.md embeds (figures/), re-running only
#                   the scripts whose inputs changed since their last run
#   make all        test + figures
#   make clean      delete data/processed/ (everything in it is regenerable)
#
#   N_TOYS=50 make figures   quicker, noisier pull study
#
# Works with GNU make 3.81 (macOS) and later. The MadGraph5 closure test is
# not part of it: run src/simulate_mumu_mg5.py by hand, see README section 3.

PYTHON ?= python
N_TOYS ?= 500
OUT := data/processed
STAMPS := $(OUT)/.stamps

RAW := $(wildcard data/raw/*.csv)
CORE := src/compute_cross_section.py src/electroweak.py $(RAW)
TOY := $(CORE) src/simulate_mumu.py

FIGURES := figures/opal_mumu_lineshape.png figures/closure_toy_lineshape.png \
           figures/closure_toy_afb.png figures/pull_study.png \
           figures/mumu_extended_lineshape.png figures/joint_fit.png

.PHONY: help test results figures all clean

help:
	@sed -n 's/^#   //p' Makefile

test:
	$(PYTHON) -m pytest

results: $(STAMPS)/compute $(STAMPS)/toy $(STAMPS)/pull $(STAMPS)/extended

figures: $(FIGURES)

all: test figures

clean:
	rm -rf $(OUT)

# One stamp per script: it is rebuilt when the script or anything it imports
# or reads changes, and every figure it writes is copied after it.
$(STAMPS)/compute: $(CORE)
	$(PYTHON) src/compute_cross_section.py
	@mkdir -p $(STAMPS) && touch $@

$(STAMPS)/toy: $(TOY)
	$(PYTHON) src/simulate_mumu.py
	@mkdir -p $(STAMPS) && touch $@

$(STAMPS)/pull: $(TOY) src/pull_study.py
	$(PYTHON) src/pull_study.py --n-toys $(N_TOYS)
	@mkdir -p $(STAMPS) && touch $@

$(STAMPS)/extended: $(CORE) src/extended_fits.py
	$(PYTHON) src/extended_fits.py
	@mkdir -p $(STAMPS) && touch $@

figures/opal_mumu_lineshape.png: $(STAMPS)/compute
	cp $(OUT)/mumu_cross_section.png $@
figures/closure_toy_lineshape.png: $(STAMPS)/toy
	cp $(OUT)/simulated_mumu_closure.png $@
figures/closure_toy_afb.png: $(STAMPS)/toy
	cp $(OUT)/simulated_mumu_afb.png $@
figures/pull_study.png: $(STAMPS)/pull
	cp $(OUT)/pull_study.png $@
figures/mumu_extended_lineshape.png: $(STAMPS)/extended
	cp $(OUT)/mumu_extended_lineshape.png $@
figures/joint_fit.png: $(STAMPS)/extended
	cp $(OUT)/joint_fit.png $@
