from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle

class ROC:
    class Curve:
        def __init__(self, predictions, expected, color, label):
            self.predictions = predictions
            self.expected = expected
            self.color = color
            self.label = label

        def plot(self):
            fpr, tpr, _ = roc_curve(self.predictions, self.expected)
            area_under_curve = auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                color=self.color,
                label='{} ROC curve (area = {:0.2f})'.format(self.label, area_under_curve),
            )
    
    def __init__(self):
        self.curves = []
        self.colors = cycle(['darkorange', 'darkgreen', 'cornflowerblue', 'aqua'])
        
    def add_curve(self, **kwargs):
        self.curves.append(self.Curve(color=next(self.colors), **kwargs))
        
    def show(self):
        plt.figure(figsize=(12, 12))
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        
        for curve in self.curves:
            curve.plot()

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc='lower right')
        plt.show()
