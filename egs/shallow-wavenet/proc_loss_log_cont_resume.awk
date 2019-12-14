# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

BEGIN {
    min_idx=0;
    min_val=9999.9;
    min_val4=9999.9;
    min_va18=9999.9;
    min_trn=9999.9
} {
    if ($6=="average" && $7=="evaluation") {
        split($5,str1,")");
        split(str1[1],str2,":");
        idx=str2[2];
        printf "%d %lf %s %s %lf dB %s %lf %s %lf %s %s %lf %s %s %lf dB %s %lf %s %lf %s %s\n", \
            idx, $10, $11, $12, $13, $15, $16, $17, $18, $19, $20, \
                tmp_trn, tmp_trn2, tmp_trn3, tmp_trn4, tmp_trn5, tmp_trn6, tmp_trn7, tmp_trn8, tmp_trn9,tmp_trn10;
        if ($10+$13+$18<=min_val+min_val4+min_val8) {
            min_idx=idx;
            min_val=$10;
            min_val2=$11;
            min_val3=$12;
            min_val4=$13;
            min_val5=$15;
            min_val6=$16;
            min_val7=$17;
            min_val8=$18;
            min_val9=$19;
            min_val10=$20;
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
    } else if ($6=="average" && $7=="training") {
        tmp_trn=$10
        tmp_trn2=$11
        tmp_trn3=$12
        tmp_trn4=$13
        tmp_trn5=$15
        tmp_trn6=$16
        tmp_trn7=$17
        tmp_trn8=$18
        tmp_trn9=$19
        tmp_trn10=$20
    }
} END {
    printf "# min = %d %lf %s %s %lf dB %s %lf %s %lf %s %s %lf %s %s %lf dB %s %lf %s %lf %s %s\n", \
        min_idx, min_val, min_val2, min_val3, min_val4, min_val5, min_val6, min_val7, min_val8, min_val9, min_val10, \
            min_trn, min_trn2, min_trn3, min_trn4, min_trn5, min_trn6, min_trn7, min_trn8, min_trn9, min_trn10;
}

