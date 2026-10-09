.PHONY: dllama clean check
.DEFAULT_GOAL := dllama

dllama:
	$(MAKE) -C EdgeVisor dllama

clean:
	$(MAKE) -C EdgeVisor clean

check:
	$(MAKE) -C EdgeVisor check
