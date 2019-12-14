# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

BEGIN {
    min_idx=0;
    min_val=9999.9;
    min_val4=9999.9;
    min_va18=9999.9;
    min_trn=9999.9
} {
    if ($2=="average" && $3=="evaluation") {
        split($1,str1,")");
        split(str1[1],str2,":");
        idx=str2[2];
        printf "%d %lf %s %s %lf dB %s %lf %s %lf %s %s %lf %s %s %lf dB %s %lf %s %lf %s %s\n", \
            idx, $6, $7, $8, $9, $11, $12, $13, $14, $15, $16, \
                tmp_trn, tmp_trn2, tmp_trn3, tmp_trn4, tmp_trn5, tmp_trn6, tmp_trn7, tmp_trn8, tmp_trn9,tmp_trn10;
        if ($6+$9+$14<=min_val+min_val4+min_val8) {
            min_idx=idx;
            min_val=$6;
            min_val2=$7;
            min_val3=$8;
            min_val4=$9;
            min_val5=$11;
            min_val6=$12;
            min_val7=$13;
            min_val8=$14;
            min_val9=$15;
            min_val10=$16;
            min_trn=tmp_trn
            min_trn2=tmp_trn2
            min_trn3=tmp_trn3
            min_trn4=tmp_trn4
            min_trn5=tmp_trn5
            min_trn6=tmp_trn6
            min_trn7=tmp_trn7
            min_trn8=tmp_trn8
            min_trn9=tmp_trn9
            min_trn10=tmp_trn10
        }
    } else if ($2=="average" && $3=="training") {
        tmp_trn=$6
        tmp_trn2=$7
        tmp_trn3=$8
        tmp_trn4=$9
        tmp_trn5=$11
        tmp_trn6=$12
        tmp_trn7=$13
        tmp_trn8=$14
        tmp_trn9=$15
        tmp_trn10=$16
    }
} END {
    printf "# min = %d %lf %s %s %lf dB %s %lf %s %lf %s %s %lf %s %s %lf dB %s %lf %s %lf %s %s\n", \
        min_idx, min_val, min_val2, min_val3, min_val4, min_val5, min_val6, min_val7, min_val8, min_val9, min_val10, \
            min_trn, min_trn2, min_trn3, min_trn4, min_trn5, min_trn6, min_trn7, min_trn8, min_trn9, min_trn10
}

