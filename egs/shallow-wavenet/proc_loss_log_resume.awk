# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

BEGIN {
    min_idx=0;
    min_val=9999.9;
    min_trn=9999.9
} {
    #if ($2=="average" && $3=="evaluation") {
    if ($6=="average" && $7=="evaluation") {
        split($5,str1,")");
        #split($1,str1,")");
        split(str1[1],str2,":");
        idx=str2[2];
        #printf "%d %lf %lf\n", idx, $10, tmp_trn;
        #printf "%d %lf %s %s %lf %s %s\n", idx, $6, $7, $8, tmp_trn, tmp_trn2, tmp_trn3;
        printf "%d %lf %s %s %lf %s %s\n", idx, $10, $11, $12, tmp_trn, tmp_trn2, tmp_trn3;
        #if ($6<=min_val) {
        if ($10<=min_val) {
            min_idx=idx;
            #min_val=$6;
            min_val=$10;
            #min_val2=$7;
            min_val2=$11;
            #min_val3=$8;
            min_val3=$12;
            min_trn=tmp_trn
            min_trn2=tmp_trn2
            min_trn3=tmp_trn3
        }
    #} else if ($2=="average" && $3=="training") {
    } else if ($6=="average" && $7=="training") {
        #tmp_trn=$6
        tmp_trn=$10
        #tmp_trn2=$7
        tmp_trn2=$11
        #tmp_trn3=$8
        tmp_trn3=$12
    }
} END {
    printf "# min = %d %lf %s %s %lf %s %s \n", min_idx, min_val, min_val2, min_val3, min_trn, min_trn2, min_trn3
}
