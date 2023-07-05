#!/bin/bash
# If you are editing the .py files only, you can run this script to create .md
# notebooks from the .py files. Becareful, if you changed the md files, this
# will overwrite your changes.

echo "Across all example_* directories, copy all driver_*.py to driver_*.md"
for file in example_*/driver_*.py; do 
    cp -- "$file" "${file%.py}.md"
done

echo "Removing all \"\"\" from driver_*.md"
find example_* -name 'driver_*.md' |xargs perl -pi -e 's/"""//g'
