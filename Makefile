.PHONY: dllama clean
.DEFAULT_GOAL := dllama

dllama:
	$(MAKE) -C EdgeVisor dllama

clean:
	$(MAKE) -C EdgeVisor clean
