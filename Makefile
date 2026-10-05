.PHONY: install test fetch composites labels chips train
install: ; pip install -e ".[dev,fetch]"
test:    ; pytest -q
fetch:   ; bcchange fetch-vectors
composites: ; bcchange build-composites
labels:  ; bcchange make-labels
chips:   ; bcchange make-chips
train:   ; bcchange train --epochs 15
data:    fetch composites labels chips
