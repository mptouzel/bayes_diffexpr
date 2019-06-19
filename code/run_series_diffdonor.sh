#!/bin/bash

# null model
if true; then
for donor in "S1" ;do # "S2" "P1" "P2" "Q1" "Q2"; do 
# for day in pre0 7 15 45; do
  daynull="0"
  day="15"
  python infer_diffexpr_main.py "${donor}_${daynull}_F1_.txt" "${donor}_${daynull}_F2_.txt" "${donor}_${daynull}_F1_.txt" "${donor}_${day}_F1_.txt";
  # nohup python infer_diffexpr_main.py "${donor}_${daynull}_F1_.txt" "${donor}_${daynull}_F2_.txt" "${donor}_${daynull}_F1_.txt" "${donor}_${day}_F1_.txt" &
# done
done
fi
