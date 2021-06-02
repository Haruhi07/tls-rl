#!/bin/sh

echo "Enter commit info:"
read info
git add .
git commit -m "$info"
git push
