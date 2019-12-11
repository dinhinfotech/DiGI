**DiGI**

This repository consists of the data and source code which are used to evaluate the performances of the DiGI method proposed in the paper entitled "Heterogeneous Networks Integration for Disease Gene Prioritization with Node Kernels", submitted to Bioinformatics journal. 

DiGI is also implemented as a web tool and it is available at: http://rna.informatik.uni-freiburg.de/DiGI/Input.jsp

**Dependency**

- Python >= 2.7
- scikit-learn >= 0.17.1
- networkx >= 2.2
- scipy >= 1.3.0
- EDeN: https://github.com/fabriziocosta/EDeN.git

**Data**

There are different sub-folders in the Data folder which contain the data used for different experimental setting in the paper. Besides, you can find the test data in the "test_data" sub-folder which is used for a quick test run.

**How to run DiGI**

Under the root directory of this repository, run the main.py with papameters' values. Following is an example how to run our model using test data:

> **_python main.py \  
--adj_folder data/test_data/adj_matrices \  
--train_genes_file data/test_data/train_genes \  
--train_labels_file data/test_data/train_labels \  
--all_genes_file data/test_data/all_genes \  
--list_D 10 15 \  
--list_C 5 \  
--list_d 1 2 \    
--list_r 1 2 \  
--use_lou_**
