#!/bin/bash

# run with nohup script_name.sh


# null model
if true; then
# for donor in "S1" "S2" "P1" "P2" "Q1" "Q2"; do 
# for daynull in pre0 0 7 15 45; do
#  for model in "3"; do
  #daynull="0"
  
  donor="S2"
  day="15"
  daynull="0"
  python infer_diffexpr_main.py "${donor}_${daynull}_F1_.txt" "${donor}_${daynull}_F2_.txt" "${donor}_${daynull}_F1_.txt" "${donor}_${day}_F2_.txt";
#   python infer_diffexpr_main.py "${donor}_${daynull}_F1_.txt" "${donor}_${daynull}_F2_.txt" "x" "x" $model;
#   python infer_diffexpr_main.py "${donor}_${daynull}_F2_.txt" "${donor}_${daynull}_F1_.txt" "x" "x" $model;
#  done
# done
# done
fi

if false; then
for donor in "S1" "S2" "P1" "P2" "Q1" "Q2"; do 
for daynull in pre0 0 7 15 45; do
 for model in "3"; do
  #daynull="0"
  day="0"
  #python infer_diffexpr_main.py "${donor}_${daynull}_F1_.txt" "${donor}_${daynull}_F2_.txt" "${donor}_${daynull}_F1_.txt" "${donor}_${day}_F1_.txt";
  python infer_diffexpr_main.py "${donor}_${daynull}_F1_.txt" "${donor}_${daynull}_F2_.txt" "x" "x" $model;
#   python infer_diffexpr_main.py "${donor}_${daynull}_F2_.txt" "${donor}_${daynull}_F1_.txt" "x" "x" $model;
 done
done
done
fi

# run grid on cluster
if false; then
  for donor in "P1" "P2"; do # "Q1" "Q2"; do # "S1"
    #donor="Q1" 
    for day in "7" "15" "45"; do
      for ait in {0..19}; do
        #ait=9
        for rep1 in "F1" "F2"; do
          for rep2 in "F1" "F2"; do
            daynull="0"
            #day="0"
            if [ "${rep1}" = "F1" ] && [ "${rep2}" = "F2" ]; then
              echo # skip
            else
              num_jobs=$(ps -u puelma --no-heading | wc -l)
              while [ "$num_jobs" -gt "27" ]; do
                sleep 10
                num_jobs=$(ps -u puelma --no-heading | wc -l)
              done
              echo infer_diffexpr_main.py "${donor}_${daynull}_F1_.txt" "${donor}_${daynull}_F2_.txt" "${donor}_${daynull}_${rep1}_.txt" "${donor}_${day}_${rep2}_.txt" ${ait}
              python infer_diffexpr_main.py "${donor}_${daynull}_F1_.txt" "${donor}_${daynull}_F2_.txt" "${donor}_${daynull}_${rep1}_.txt" "${donor}_${day}_${rep2}_.txt" ${ait} &
              
            fi
          done
        done
      done
   done
  done
fi

# run polsih locally
if false; then
  for donor in "S1" "S2" "P1" "P2" "Q1" "Q2"; do # "S1"
    #donor="Q1"
    for day in "pre0" "0" "7" "15" "45"; do
      #for ait in {0..19}; do
        ait=0
        for rep1 in "F1" "F2"; do
          for rep2 in "F1" "F2"; do
            daynull="0"
            #day="0"
            #if [ "${rep1}" = "F1" ] && [ "${rep2}" = "F2" ]; then
            #  echo # skip
            #else

            output_path='../../output/'
            null_pair=$donor'_0_F1_'$donor'_0_F2'
            diff_pair=$donor'_0_'$rep1'_'$donor'_'$day'_'$rep2
            Ps_type='sym_exp'
            runname='v4_ct_1_mt_2_st_'$Ps_type'_min0_maxinf'
            run_name='diffexpr_pair_'$null_pair'_'$runname
            outpath=$output_path$diff_pair'/'$run_name'/'
            FILE=$outpath'Lsurface.npy'
            if test -f "$FILE"; then
            FILE=$outpath'diffexpr_success.npy'
            # FILE=$outpath'Lsurface'$ait'.npy'
            if test ! -f "$FILE"; then
                #num_jobs=$(ps -u puelma --no-heading | wc -l)
                #while [ "$num_jobs" -gt "27" ]; do
                #    sleep 10
                #    num_jobs=$(ps -u puelma --no-heading | wc -l)
                #done
                echo infer_diffexpr_main.py "${donor}_${daynull}_F1_.txt" "${donor}_${daynull}_F2_.txt" "${donor}_${daynull}_${rep1}_.txt" "${donor}_${day}_${rep2}_.txt" ${ait}
                python infer_diffexpr_main.py "${donor}_${daynull}_F1_.txt" "${donor}_${daynull}_F2_.txt" "${donor}_${daynull}_${rep1}_.txt" "${donor}_${day}_${rep2}_.txt" ${ait} 
            fi
            #fi
            fi
          done
        done
      #done
   done
  done
fi

if false; then
  donor="S2"
  daynull="0"
  day="15"
  nohup python infer_diffexpr_main.py "${donor}_${daynull}_F1_.txt" "${donor}_${daynull}_F2_.txt" "${donor}_${daynull}_F1_.txt" "${donor}_${day}_F2_.txt"  &
fi
