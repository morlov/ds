from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import pylab

# Finds best params for model with grid search cv and evaluates on test set
def select_model(df, target, model, params, estimate=False):
    cv = GridSearchCV(model, params, cv=KFold(10), verbose=1, scoring="accuracy", n_jobs=-1, refit=False)
    cv = cv.fit(df, target)
    report_cols = ["params", "rank_test_score", "mean_test_score", "mean_train_score", "std_test_score", "std_train_score"]
    report_df = pd.DataFrame(cv.cv_results_ )[report_cols].sort_values(by="rank_test_score")
    par = report_df["params"].iloc[0]
    print(par)
    display(report_df.head())
    
def report_roc(y, prob):
    fpr, tpr, _ = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)
    
    # Plot ROC curve
    pylab.clf()
    pylab.plot(fpr, tpr, label='ROC AUC = %0.2f' % roc_auc)
    pylab.plot([0, 1], [0, 1], 'k--')
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('False Positive Rate')
    pylab.ylabel('True Positive Rate')
    pylab.title('Receiver operating characteristic')
    pylab.legend(loc="lower right")
    pylab.grid(True)
    pylab.show()
    
def report_precision_recall(y, prob):
    precision, recall, _ = precision_recall_curve(y, prob)
    area = average_precision_score(y, prob)
    print("Area under the precision-recall curve : %f" %  area)
    
    # Plot ROC curve
    pylab.clf()
    pylab.plot(recall, precision, label='Recall precision')
    pylab.plot([0, 1], [0, 1], 'k--')
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.title('Recall precision curve')
    pylab.legend(loc="lower right")
    pylab.grid(True)
    pylab.show()
