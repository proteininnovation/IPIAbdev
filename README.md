WELLCOME TO IPI AL-ML Framework!


1. Download IPIAbDev

1. Create a new environment with Python 3.11 or 3.12
 
conda create -n ml python=3.11 -y
# Activate it
conda activate ml

3. python package install
conda install -c bioconda anarci
pip install -r requirements.txt


4.prepare train set

   filename:         your_trainset_name.xlsx 
   required columns: BARCODE,CDR3,HSEQ,LSEQ,sec_filter,psr_filter

5. How to use IPIPred


# Generate embeddings
python predict_developability.py --build-embedding data/test.xlsx --lm all

python predict_developability.py --build-embedding data/test.xlsx --lm ablang

# Xgboost evaluation with k-fold validation
python predict_developability.py --kfold 10 --target sec_filter --lm antiberta2 --model xgboost

# Randomforest evaluation with k-fold validation
python predict_developability.py --kfold 5 --target sec_filter --lm antiberty --model rf

# Train final  model with full dataset
python predict_developability.py --train --target sec_filter --lm antiberta2 --model xgboost

#  Predict instantly
python predict_developability.py --predict data/new_lib.xlsx --target sec_filter --lm antiberta2



# Train PSR with CNN
python predict_developability.py --train --target psr_filter --lm antiberta2 --model cnn

# 10-fold CV with CNN
python predict_developability.py --kfold 10 --target sec_filter --lm antiberta2 --model cnn

# Predict on test set
python predict_developability.py --predict data/test.xlsx --target sec_filter --lm ablang --model cnn

ML architecture amd design: Hoan Nguyen, PhD
Contact: {Hoan.Nguyen, Andre.Teixeira}@proteininnovation.org 
