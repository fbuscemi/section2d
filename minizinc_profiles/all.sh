minizinc -n 0 -o c1.txt c1.mzn 
python3 convert_csv.py --infile c1.txt --outfile c1.csv

minizinc -n 0 -o c2.txt c2.mzn 
python3 convert_csv.py --infile c2.txt --outfile c2.csv

minizinc -n 0 -o i1.txt i1.mzn 
python3 convert_csv.py --infile i1.txt --outfile i1.csv

minizinc -n 0 -o rect.txt rect.mzn 
python3 convert_csv.py --infile rect.txt --outfile rect.csv

minizinc -n 0 -o tube.txt tube.mzn 
python3 convert_csv.py --infile rect.txt --outfile rect.csv

minizinc -n 0 -o z1.txt z1.mzn 
python3 convert_csv.py --infile z1.txt --outfile z1.csv

minizinc -n 0 -o z2.txt z2.mzn 
python3 convert_csv.py --infile z2.txt --outfile z2.csv

minizinc -n 0 -o l2.txt l2.mzn 
python3 convert_csv.py --infile l2.txt --outfile l2.csv
