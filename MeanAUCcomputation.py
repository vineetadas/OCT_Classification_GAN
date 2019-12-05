
import numpy as np
import scipy.io
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc
import pickle
from scipy.io import loadmat
import pickle
from sklearn.metrics import roc_curve, auc
from scipy import interp
from matplotlib import rc, rcParams
num_classes =3
labels_name={'AMD':0,'DME':1,'NORMAL':2}
rc('font', weight='bold')
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)


tprs_disc = []
aucs_disc = []
mean_fpr_disc = np.linspace(0, 1, 100)

################# Fold1

y_orig_fold1_500 = loadmat('results/y_orig_fold1_500.mat', struct_as_record=False, squeeze_me=True)
y_orig_fold1_500 = y_orig_fold1_500['y_orig_fold1_500']


y_pred_fold1_500 = loadmat('results/y_pred_fold1_500.mat', struct_as_record=False, squeeze_me=True)
y_pred_fold1_500 = y_pred_fold1_500['y_pred_fold1_500']



predictions_disc =  np_utils.to_categorical(y_pred_fold1_500,3)



y_test_disc=  np_utils.to_categorical(y_orig_fold1_500,3)


fpr_disc, tpr_disc, _ = roc_curve(y_test_disc.ravel(), predictions_disc.ravel())

roc_auc_disc = auc(fpr_disc, tpr_disc)
tprs_disc.append(interp(mean_fpr_disc, fpr_disc, tpr_disc))
tprs_disc[-1][0] = 0.0
roc_auc_disc = auc(fpr_disc, tpr_disc)
aucs_disc.append(roc_auc_disc)


#plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

################# Fold2

y_orig_fold2_500 = loadmat('results/y_orig_fold2_500.mat', struct_as_record=False, squeeze_me=True)
y_orig_fold2_500 = y_orig_fold2_500['y_orig_fold2_500']


y_pred_fold2_500 = loadmat('results/y_pred_fold2_500.mat', struct_as_record=False, squeeze_me=True)
y_pred_fold2_500 = y_pred_fold2_500['y_pred_fold2_500']


predictions_disc =  np_utils.to_categorical(y_pred_fold2_500,3)


y_test_disc=  np_utils.to_categorical(y_orig_fold2_500,3)


fpr_disc, tpr_disc, _ = roc_curve(y_test_disc.ravel(), predictions_disc.ravel())

roc_auc_disc = auc(fpr_disc, tpr_disc)
tprs_disc.append(interp(mean_fpr_disc, fpr_disc, tpr_disc))
tprs_disc[-1][0] = 0.0
roc_auc_disc = auc(fpr_disc, tpr_disc)
aucs_disc.append(roc_auc_disc)


#plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

################# Fold3

y_orig_fold3_500 = loadmat('results/y_orig_fold3_500.mat', struct_as_record=False, squeeze_me=True)
y_orig_fold3_500 = y_orig_fold3_500['y_orig_fold3_500']


y_pred_fold3_500 = loadmat('results/y_pred_fold3_500.mat', struct_as_record=False, squeeze_me=True)
y_pred_fold3_500 = y_pred_fold3_500['y_pred_fold3_500']



predictions_disc =np_utils.to_categorical( y_pred_fold3_500,3)


y_test_disc=  np_utils.to_categorical(y_orig_fold3_500,3)


fpr_disc, tpr_disc, _ = roc_curve(y_test_disc.ravel(), predictions_disc.ravel())

roc_auc_disc = auc(fpr_disc, tpr_disc)
tprs_disc.append(interp(mean_fpr_disc, fpr_disc, tpr_disc))
tprs_disc[-1][0] = 0.0
roc_auc_disc = auc(fpr_disc, tpr_disc)
aucs_disc.append(roc_auc_disc)


################# Fold4

y_orig_fold4_500 = loadmat('results/y_orig_fold4_500.mat', struct_as_record=False, squeeze_me=True)
y_orig_fold4_500 = y_orig_fold4_500['y_orig_fold4_500']


y_pred_fold4_500 = loadmat('results/y_pred_fold4_500.mat', struct_as_record=False, squeeze_me=True)
y_pred_fold4_500 = y_pred_fold4_500['y_pred_fold4_500']


predictions_disc = np_utils.to_categorical(y_pred_fold4_500,3)

y_test_disc=  np_utils.to_categorical(y_orig_fold4_500,3)


fpr_disc, tpr_disc, _ = roc_curve(y_test_disc.ravel(), predictions_disc.ravel())

roc_auc_disc = auc(fpr_disc, tpr_disc)
tprs_disc.append(interp(mean_fpr_disc, fpr_disc, tpr_disc))
tprs_disc[-1][0] = 0.0
roc_auc_disc = auc(fpr_disc, tpr_disc)
aucs_disc.append(roc_auc_disc)


#plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)


################# Fold5

y_orig_fold5_500 = loadmat('results/y_orig_fold5_500.mat', struct_as_record=False, squeeze_me=True)
y_orig_fold5_500 = y_orig_fold5_500['y_orig_fold5_500']


y_pred_fold5_500 = loadmat('results/y_pred_fold5_500.mat', struct_as_record=False, squeeze_me=True)
y_pred_fold5_500 = y_pred_fold5_500['y_pred_fold5_500']


predictions_disc =  np_utils.to_categorical(y_pred_fold5_500,3)



y_test_disc=  np_utils.to_categorical(y_orig_fold5_500,3)


fpr_disc, tpr_disc, _ = roc_curve(y_test_disc.ravel(), predictions_disc.ravel())

roc_auc_disc = auc(fpr_disc, tpr_disc)
tprs_disc.append(interp(mean_fpr_disc, fpr_disc, tpr_disc))
tprs_disc[-1][0] = 0.0
roc_auc_disc = auc(fpr_disc, tpr_disc)
aucs_disc.append(roc_auc_disc)


#plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
######################### Mean computation

mean_tpr_disc = np.mean(tprs_disc, axis=0)
mean_tpr_disc[-1] = 1.0
mean_auc_disc = auc(mean_fpr_disc, mean_tpr_disc)
std_auc_disc = np.std(aucs_disc)

print('Mean AUC', mean_auc_disc)
print('STD AUC',std_auc_disc)
 
