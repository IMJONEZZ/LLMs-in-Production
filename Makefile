# Set the path to your scripts folder
SCRIPTS := scripts

.PHONY: clean
clean:
	@$(SCRIPTS)/clean.sh

.PHONY: lint
lint:
	@$(SCRIPTS)/lint.sh

.PHONY: setup
setup:
	@$(SCRIPTS)/setup.sh

.PHONY: test
test:
	@$(SCRIPTS)/test.sh
