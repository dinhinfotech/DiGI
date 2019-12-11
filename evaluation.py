import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import util
from sklearn.svm import SVC


class Validation:
    def __init__(self,
                 kernels=None,
                 all_genes=None, 
                 training_genes=None,
                 training_labels=None,
                 n_folds=5):
                     
        self.kernels = kernels
        self.all_genes = all_genes
        self.training_genes = training_genes
        self.training_labels = training_labels
        self.n_folds = n_folds

    def select_parameters(self, training_genes=None, training_labels=None):
        """Model selection"""
        
        list_c = [10e-4, 10e-3, 10e-2, 10e-1, 1, 10e+1, 10e+2, 10e+3, 10e+4]
        
        dict_gene_idx = {}
        for idx, gene in enumerate(self.all_genes):
            dict_gene_idx[gene]=idx
            
        dict_paras_auc = {}
        
        for kernel_idx in range(len(self.kernels)):
            for c_idx in range(len(list_c)):
                dict_paras_auc[(kernel_idx, c_idx)] = 0

        skf = StratifiedKFold(n_splits=3, shuffle=False)
        for train_index, test_index in skf.split(np.zeros(len(training_labels)), training_labels):
            training_genes_left = [training_genes[idx] for idx in train_index]
            training_indices = [dict_gene_idx[gene] for gene in training_genes_left]
            training_labels_left = [training_labels[idx] for idx in train_index]
            test_genes_left = [training_genes[idx] for idx in test_index]
            test_indices = [dict_gene_idx[gene] for gene in test_genes_left]
            test_labels_left = [training_labels[idx] for idx in test_index]
            unknown_genes = []
            unknown_genes.extend(test_genes_left)
            for gene in self.all_genes:
                if gene not in training_genes:
                    unknown_genes.append(gene)
            unknown_indices = [dict_gene_idx[gene] for gene in unknown_genes]
        
            for kernel_idx, kernel in enumerate(self.kernels):
                training_kernel = util.extract_submatrix(training_indices,training_indices,kernel)
                unknown_kernel = util.extract_submatrix(unknown_indices,training_indices,kernel)
                
                for c_idx, c in enumerate(list_c):                        
                    clf = SVC(C=c, kernel='precomputed')
                    clf.fit(training_kernel, training_labels_left)
                    
                    scores = clf.decision_function(unknown_kernel)
                    
                    qscores = []
                    
                    for s in scores[:len(test_indices)]:
                        qscore = float(sum([int(s >= value) for value in scores]))/len(scores)
                        qscores.append(qscore)
                    fpr, tpr, thresholds = metrics.roc_curve(test_labels_left, qscores, pos_label=1)
                    auc = metrics.auc(fpr, tpr)
                    
                    dict_paras_auc[(kernel_idx, c_idx)] += auc
                
        return max(dict_paras_auc, key=dict_paras_auc.get)        
                
    def validate_kfolds(self):
        
        list_c = [10e-4, 10e-3, 10e-2, 10e-1, 1, 10e+1, 10e+2, 10e+3, 10e+4]
        aucs = []
        
        dict_gene_idx = {}
        for idx, gene in enumerate(self.all_genes):
            dict_gene_idx[gene]=idx
            
        dict_paras_auc = {}
        
        for kernel_idx in range(len(self.kernels)):
            for c_idx in range(len(list_c)):
                dict_paras_auc[(kernel_idx, c_idx)] = 0

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=False)
        for train_index, test_index in skf.split(np.zeros(len(self.training_labels)), self.training_labels):
            training_genes_left = [self.training_genes[idx] for idx in train_index]
            training_indices = [dict_gene_idx[gene] for gene in training_genes_left]
            training_labels_left = [self.training_labels[idx] for idx in train_index]
            
            test_genes_left = [self.training_genes[idx] for idx in test_index]
            test_indices = [dict_gene_idx[gene] for gene in test_genes_left]
            test_labels_left = [self.training_labels[idx] for idx in test_index]
            unknown_genes = []
            unknown_genes.extend(test_genes_left)
            for gene in self.all_genes:
                if gene not in self.training_genes:
                    unknown_genes.append(gene)
            unknown_indices = [dict_gene_idx[gene] for gene in unknown_genes]
            
            (kernel_idx, c_idx) = self.select_parameters(training_genes=training_genes_left, training_labels=training_labels_left)
            
            training_kernel = util.extract_submatrix(training_indices, training_indices, self.kernels[kernel_idx])
            unknown_kernel = util.extract_submatrix(unknown_indices, training_indices, self.kernels[kernel_idx])

            clf = SVC(C=list_c[c_idx], kernel='precomputed')
            clf.fit(training_kernel, training_labels_left)
            
            scores = clf.decision_function(unknown_kernel)
            
            qscores = []
            
            for s in scores[:len(test_indices)]:
                qscore = float(sum([int(s >= value) for value in scores]))/len(scores)
                qscores.append(qscore)
            fpr, tpr, thresholds = metrics.roc_curve(test_labels_left, qscores, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            
            aucs.append(auc)
                
        return aucs

    def validate_leave_one_out(self):

        list_c = [10e-4, 10e-3, 10e-2, 10e-1, 1, 10e+1, 10e+2, 10e+3, 10e+4]

        dict_gene_idx = {}
        for idx, gene in enumerate(self.all_genes):
            dict_gene_idx[gene] = idx

        dict_paras_auc = {}

        for kernel_idx in range(len(self.kernels)):
            for c_idx in range(len(list_c)):
                dict_paras_auc[(kernel_idx, c_idx)] = 0

        all_qscores = []
        for train_g_idx, train_g in enumerate(self.training_genes):
            print('processing gene ', train_g_idx)
            training_genes_left = self.training_genes[:]
            del training_genes_left[train_g_idx]

            training_indices = [dict_gene_idx[gene] for gene in training_genes_left]
            training_labels_left = self.training_labels[:]
            del training_labels_left[train_g_idx]

            unknown_genes = [train_g]
            for gene in self.all_genes:
                if gene not in self.training_genes:
                    unknown_genes.append(gene)
            unknown_indices = [dict_gene_idx[gene] for gene in unknown_genes]

            (kernel_idx, c_idx) = self.select_parameters(training_genes=training_genes_left,
                                                         training_labels=training_labels_left)

            training_kernel = util.extract_submatrix(training_indices, training_indices, self.kernels[kernel_idx])
            unknown_kernel = util.extract_submatrix(unknown_indices, training_indices, self.kernels[kernel_idx])

            clf = SVC(C=list_c[c_idx], kernel='precomputed')
            clf.fit(training_kernel, training_labels_left)

            scores = clf.decision_function(unknown_kernel)
            qscore = float(sum([int(scores[0] >= value) for value in scores])) / len(scores)
            all_qscores.append(qscore)

        fpr, tpr, thresholds = metrics.roc_curve(self.training_labels, all_qscores, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        return auc
