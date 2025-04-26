cat Independent-test_l_hg19_vcf.csv | while read s
do
chr=`echo ${s} | cut -d ',' -f 1 | sed 's/chr//g'`
pos=`echo ${s} | cut -d ',' -f 2`
label=`echo ${s} | cut -d ',' -f 6`
result=`tabix hg19_DVAR.score.gz $chr:$pos-$pos` 
echo -e "$result\t$label" >> DVAR_sr.tsv
done
