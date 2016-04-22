### Rules

For mining the rules with AMIE+:

```bash
find . -type f -name "*-train.txt.gz" | tr "/" " " | awk '{ print "mkdir tmp ; zcat " $2 "/" $3 "/" $4 " > tmp/kg.tsv ; mkdir -p " $2 "/" $3 "/rules/ ; java -Xms32g -Xmx128g -jar bin/amie_plus.jar tmp/kg.tsv > " $2 "/" $3 "/rules/" $3 "-rules.txt 2>&1 ; rm -rf tmp"  }' | sort > rules.sh
```

For converting them into json:

```bash
find . -type f -name "*.txt" | grep rules | grep -v yago3 | sed -e 's/\<txt\>//g' | awk '{ print "./bin/amie-to-json.py " $1 "txt > " $1 "json ; gzip -9 " $1 "json" }' | sort
```
