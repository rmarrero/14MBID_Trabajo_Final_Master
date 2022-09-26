#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob, os
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn.metrics as metrics


# In[2]:


def make_classification_report(model, y_true, x_test, title=''):
    
    
    # Generate predictions
    y_preds = model.predict(x_test)
    print('__________________________________________________________________')
    print(f'Reporte para el Clasificador: \n\t{title}')
    print('__________________________________________________________________')
    print('\n')
    
    # Generate report
    report = metrics.classification_report(y_true, y_preds,)
    report_dict = metrics.classification_report(y_true, y_preds,output_dict=True)
    
    # Add the title to the report dictionary
    report_dict['title'] = title
    print(report)
    print('__________________________________________________________________')
    
    return report_dict


# In[3]:


def plot_confusion_matrix(model, X, y, title=''):

    # Plot the matrix with labels    
    fig = metrics.plot_confusion_matrix(model, X, y, normalize='true', 
                                        cmap='Greens') 

    # Remove grid lines
    plt.grid(False)
    
    # Set title
    plt.title(f'Matriz de Confusión para el Clasificador {title}', fontdict={'fontsize':17})
    plt.show()
    print('__________________________________________________________________')
    return fig


# In[4]:


def plot_roc_curve(model, xtest, ytest, title=''):

    # Creating the plot
    fig, ax = plt.subplots(figsize=(8,6), ncols=1)
    roc_plot = metrics.plot_roc_curve(model, xtest, ytest, ax=ax)

    # Setting the title of the plot
    ax.set_title(f'Area bajo la curva para el Clasificador {title}', 
                 fontdict={'fontsize':17})

    # Setting a legend for the plot
    ax.legend()
    plt.show();
    
    return fig


# In[5]:


def plot_top_features(model, xtrain, title=''):

    # Turn the feature importances into a series 
    importances = pd.Series(model.feature_importances_, index=xtrain.columns)
    
    # Plot the top most important features
    importances.nlargest(20).sort_values().plot(kind='barh')
    plt.title(f'Variables Significativas para {title}', fontdict={'fontsize':17})
    plt.xlabel('Importancia')
    return importances.sort_values(ascending=False)


# In[6]:


def evaluate_model(model, xtrain, xtest, ytest, tree=False, title=''):
    
    make_classification_report(model, ytest, xtest, title=title)
    plot_confusion_matrix(model, xtest, ytest, title=title)
    plot_roc_curve(model, xtest, ytest, title=title)
    
    # Feature importance can only be run on tree based models
    if tree:
        plot_top_features(model, xtrain, title=title)

