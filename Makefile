.PHONY: dllama clean check
.DEFAULT_GOAL := dllama
# Each target delegates to the same object directory. Serialize delegations
# so `make -j2 dllama check` cannot compile/write the same objects concurrently.
# The delegated make still uses the jobserver for its own parallel build.
.NOTPARALLEL:

dllama:
	$(MAKE) -C EdgeVisor dllama

clean:
	$(MAKE) -C EdgeVisor clean

check:
	$(MAKE) -C EdgeVisor check
