# Reproduce the results of the M87 RIAF + jet paper. Use the conda environment "m87" (environment.yml).
#   make build     compile the jet code (run.x + f2py module JetCode) and the RIAF radiative code
#   make data      rebuild the SED points from the EHT MWL 2017 release and compare them with data/obs_data/
#   make riaf      solve the RIAF (hydrodynamics + spectrum)               -> analysis/runs/riaf_m87/
#   make models    run the jet models: Jet+RIAF, Jet-only, Ro-compatible B and C -> analysis/runs/<model>/
#   make figures   paper figures, chi2 table and derived numbers            -> output/, analysis/runs/
#   make check     compare the results with the paper values (reference/)
#   make all       everything above, in order
#   make clean     remove all outputs (runs and figures); `make distclean` also removes the builds
PY      ?= python
JETDIR   = codes/extended-jet/Code
RIAFDIR  = codes/thermal-riaf
SCRIPTS  = analysis/scripts
export LC_NUMERIC = C

.PHONY: all build data riaf models figures check clean distclean

all: build data riaf models figures check

build:
	$(MAKE) -C $(JETDIR)
	cd $(JETDIR) && $(MAKE) f2py
	cd $(RIAFDIR) && ./build.sh

data:
	$(PY) $(SCRIPTS)/make_obs_data.py

riaf:
	$(PY) $(SCRIPTS)/run_models.py --riaf-only

models:
	$(PY) $(SCRIPTS)/run_models.py jet_riaf jet_only ro_B ro_C

figures:
	$(PY) $(SCRIPTS)/fig_sed.py
	$(PY) $(SCRIPTS)/fig_maps.py
	$(PY) $(SCRIPTS)/fig_scans.py
	$(PY) $(SCRIPTS)/fig_jet_power_illustrative.py
	$(PY) $(SCRIPTS)/chi2_table.py
	$(PY) $(SCRIPTS)/derived_quantities.py --outdir analysis/runs/derived
	$(PY) $(SCRIPTS)/ic_estimate.py
	$(PY) $(SCRIPTS)/ro_variants.py

check:
	$(PY) $(SCRIPTS)/check_results.py

clean:
	rm -rf analysis/runs/* output

distclean: clean
	$(MAKE) -C $(JETDIR) clean
	rm -rf $(RIAFDIR)/radproc/build
