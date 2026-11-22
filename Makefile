.PHONY: edgevisor clean check
.DEFAULT_GOAL := edgevisor
# Each target delegates to the same object directory. Serialize delegations
# so `make -j2 edgevisor check` cannot compile/write the same objects concurrently.
# The delegated make still uses the jobserver for its own parallel build.
.NOTPARALLEL:

edgevisor:
	$(MAKE) -C EdgeVisor edgevisor

clean:
	$(MAKE) -C EdgeVisor clean

check:
	$(MAKE) -C EdgeVisor check
