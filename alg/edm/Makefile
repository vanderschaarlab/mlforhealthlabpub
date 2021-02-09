EXPERT  = testing.rl
STUDENT = testing.il

.PHONY: expert
expert:
	@python -m $(EXPERT)

.PHONY: one_student
one_student:
	@python -m $(STUDENT)

.PHONY: clean
clean: purge list permit

.PHONY: purge
purge:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo)" 2>/dev/null | xargs rm -rf

.PHONY: list
list:
	@find . -name '*.py' | xargs wc -l

.PHONY: permit
permit:
	@chmod -R 755 ./
