#!/bin/bash

# null model
if false; then
for donor in "S1" "S2" "P1" "P2" "Q1" "Q2"; do 
  for daynull in pre0 0 7 15 45; do
  #daynull="0"
  python infer_diffexpr_main_shift.py "${donor}_${daynull}_F1_.txt" "${donor}_${daynull}_F2_.txt" "${donor}_${daynull}_F1_.txt" "${donor}_${daynull}_F1_.txt";
  # nohup python infer_diffexpr_main.py "${donor}_${daynull}_F1_.txt" "${donor}_${daynull}_F2_.txt" "${donor}_${daynull}_F1_.txt" "${donor}_${day}_F1_.txt" &
  done
done
fi

# diffexpr model
if true; then
  donor="S2" # "S1" "P1" "P2" "Q1" "Q2"; do 
  #for ait in {0..19}; do
  ait="0"
  daynull="0"
  day="15"
    python infer_diffexpr_main_shift.py "${donor}_${daynull}_F1_.txt" "${donor}_${daynull}_F2_.txt" "${donor}_${daynull}_F1_.txt" "${donor}_${day}_F2_.txt" ${ait}
  #  print python infer_diffexpr_main_shift.py "${donor}_${daynull}_F1_.txt" "${donor}_${daynull}_F2_.txt" "${donor}_${daynull}_F1_.txt" "${donor}_${day}_F2_.txt" &
  #python infer_diffexpr_main_shift.py "${donor}_${daynull}_F1_.txt" "${donor}_${daynull}_F2_.txt" "${donor}_${daynull}_F1_.txt" "${donor}_${day}_F2_.txt"
 #done
#  done
fi
