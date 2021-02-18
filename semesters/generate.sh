OLD_SEMESTER="w20_21"
docker-compose exec site jekyll build --destination "semesters/${OLD_SEMESTER}" --baseurl "cs236781/semesters/${OLD_SEMESTER}"
rm -rf "semesters/${OLD_SEMESTER}/semesters/*"
cp _site/semesters/index.html "semesters/${OLD_SEMESTER}/semesters"
