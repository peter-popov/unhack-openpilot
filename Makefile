dockerimage ?= img_sandbox
dockerfile ?= Dockerfile.cpu
srcdir ?= $(shell pwd)
datadir ?= $(shell pwd)

install:
	@docker build -t $(dockerimage) -f $(dockerfile) .

i: install


update:
	@docker build -t $(dockerimage) -f $(dockerfile) . --pull --no-cache

u: update

requirements.txt: requirements.in
	docker run --volume $(CURDIR):/usr/src/app --rm img_sandbox pip-compile --generate-hashes
	touch requirements.txt

run: install
	@docker run                              \
	  --ipc=host                             \
	  --rm                                   \
	  -v $(srcdir):/usr/src/app/             \
	  -v $(datadir):/data					 \
	  -it $(dockerimage)                     \
	  bash

r: run

notebook: install
	@docker run                              \
	  --ipc=host                             \
	  --rm                                   \
	  -v $(srcdir):/usr/src/app/             \
	  -v $(datadir):/data                    \
	  -p 8888:8888							 \
	  -it $(dockerimage)                     \
	  jupyter notebook --ip=0.0.0.0 --no-browser --allow-root

n: notebook


.PHONY: install i update u notebook n
