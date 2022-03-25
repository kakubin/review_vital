all: build

build:
	@docker build --tag review_vital:latest .

run:
	@docker run --rm -it -v $(PWD):/review_vital review_vital:latest bash

.PHONY: all build run
