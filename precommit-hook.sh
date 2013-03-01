#!/bin/bash

set -o errexit
set -o nounset

# # To install just do:
# cd .git/hooks/
# ln -s ../../precommit-hook.sh


# Use autopep8 to clean up whitespace etc. (also store and print a diff of
# what it changed)
autopep8 . -d -r --aggressive --ignore=E702 --ignore=E226 > .pep8diff
cat .pep8diff
patch <.pep8diff -p1

echo -e "\n\n\n"

# Run self tests
nosetests --all-modules --processes=8
