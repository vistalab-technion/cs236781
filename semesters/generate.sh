#! /usr/bin/env zsh

set -ex
set +o rmstarsilent

OLD_SEMESTER="sp21"
docker-compose exec site jekyll build --destination "semesters/${OLD_SEMESTER}" --baseurl "cs236781/semesters/${OLD_SEMESTER}"
rm -rf semesters/${OLD_SEMESTER}/semesters/*
cp _site/semesters/index.html "semesters/${OLD_SEMESTER}/semesters"
