### Rules

```bash
find . -type f -name "*-train.txt.gz" | tr "/" " " | awk '{ print "zcat " $2 "/" $3 "/" $4 " > tmp/kg.tsv ; java -XX:-UseGCOverheadLimit -Xmx4G -jar bin/amie_plus.jar tmp/kg.tsv > " $2 "/" $3 "/rules/" $3 "-rules.txt 2>&1 ; rm -f tmp/kg.tsv"  }' | sort > rules.sh
```
