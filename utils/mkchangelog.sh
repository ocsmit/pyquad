#!/bin/sh
OUTPUT_FILE="CHANGELOG.md"

exclude=""
#for i in COPYING README.md setup.py data deploy docs tests utils; do
#	exclude="$exclude :(exclude)../$i"
#done
(
echo "# Change log"
echo
git log --pretty="* $hash %d %s %cd" $exclude |
sed '
/(HEAD/{
    s/^\(.*\) (HEAD.*HEAD) \(.* \(\([^ ]\+ \)\{5\}[^ ]\+\)\)$/## HEAD\n\3\n\n\1 \2/
}
/(tag:/{
    s/^\(.*\) (tag: \([^)]*\)) \(.*\) \(\([^ ]\+ \)\{5\}[^ ]\+\)$/\n## \2\n\4\n\n\1 \3 \4/
}
s/\( [^ ]\+\)\{6\}$//
s/_/\\_/g
'
) > $CHANGELOG

res=$(git status --porcelain | grep -c ".\$OUTPUT_FILE$")
if [ "$res" -gt 0 ]; then
  git add $OUTPUT_FILE
  git commit --amend
  echo "Populated Changelog in $OUTPUT_FILE"
fi
