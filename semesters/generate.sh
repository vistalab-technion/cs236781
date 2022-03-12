#! /usr/bin/env zsh

##
# To generate files for the old semester in the semesters/ folder:
#
# 1. Edit the `_pages/semesters.md` file and add a new line there for the new
#    (current) semester (which points to `/cs236781`), and make the old one point to a new URL,
#    e.g. `/cs236781/semesters/w22`.
# 2. Edit this script and change the OLD_SEMESTER variable to be the name of the
#    old semester you used in the URL above (e.g. `w22`).
# 3. Run this script.

set -ex
set +o rmstarsilent

OLD_SEMESTER="w22"
docker-compose exec site jekyll build --destination "semesters/${OLD_SEMESTER}" --baseurl "cs236781/semesters/${OLD_SEMESTER}"
rm -rf semesters/${OLD_SEMESTER}/semesters/*
cp _site/semesters/index.html "semesters/${OLD_SEMESTER}/semesters"
