#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob, os
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn.metrics as metrics


# In[2]:


import numpy as np
import cv2
import os
import json


# In[3]:


def val_counts(df):
    '''
    Explicar función
    '''

    for col in df.columns:
        print(f'{col} value counts', '\n')
        display(df[col].value_counts(dropna=False))
        print('************************************')


# In[4]:


def tension_alta(x):
    if (x['G25a_1'] == 1) & (x['G25c_1'] ==1):
        return 1
    elif (x['G25a_1'] == 1) & (x['G25c_1'] ==2):
        return 3
    elif (x['G25a_1'] == 1) & (x['G25c_1'] ==8):
        return 3
    elif (x['G25a_1'] == 1) & (x['G25c_1'] ==9):
        return 3
    elif (x['G25a_1'] == 2):
        return 2
    elif (x['G25a_1'] == 8):
        return 2
    elif (x['G25a_1'] == 9):
        return 2


# In[5]:


def infarto(x):
    if (x['G25a_2'] == 1) & (x['G25c_2'] ==1):
        return 1
    elif (x['G25a_2'] == 1) & (x['G25c_2'] ==2):
        return 3
    elif (x['G25a_2'] == 1) & (x['G25c_2'] ==8):
        return 3
    elif (x['G25a_2'] == 1) & (x['G25c_2'] ==9):
        return 3
    elif (x['G25a_2'] == 2):
        return 2
    elif (x['G25a_2'] == 8):
        return 2
    elif (x['G25a_2'] == 9):
        return 2


# In[6]:


def angina(x):
    if (x['G25a_3'] == 1) & (x['G25c_3'] ==1):
        return 1
    elif (x['G25a_3'] == 1) & (x['G25c_3'] ==2):
        return 3
    elif (x['G25a_3'] == 1) & (x['G25c_3'] ==8):
        return 3
    elif (x['G25a_3'] == 1) & (x['G25c_3'] ==9):
        return 3
    elif (x['G25a_3'] == 2):
        return 2
    elif (x['G25a_3'] == 8):
        return 2
    elif (x['G25a_3'] == 9):
        return 2


# In[7]:


def otros_corazon(x):
    if (x['G25a_4'] == 1) & (x['G25c_4'] ==1):
        return 1
    elif (x['G25a_4'] == 1) & (x['G25c_4'] ==2):
        return 3
    elif (x['G25a_4'] == 1) & (x['G25c_4'] ==8):
        return 3
    elif (x['G25a_4'] == 1) & (x['G25c_4'] ==9):
        return 3
    elif (x['G25a_4'] == 2):
        return 2
    elif (x['G25a_4'] == 8):
        return 2
    elif (x['G25a_4'] == 9):
        return 2


# In[8]:


def varices(x):
    if (x['G25a_5'] == 1) & (x['G25c_5'] ==1):
        return 1
    elif (x['G25a_5'] == 1) & (x['G25c_5'] ==2):
        return 3
    elif (x['G25a_5'] == 1) & (x['G25c_5'] ==8):
        return 3
    elif (x['G25a_5'] == 1) & (x['G25c_5'] ==9):
        return 3
    elif (x['G25a_5'] == 2):
        return 2
    elif (x['G25a_5'] == 8):
        return 2
    elif (x['G25a_5'] == 9):
        return 2


# In[9]:


def artrosis(x):
    if (x['G25a_6'] == 1) & (x['G25c_6'] ==1):
        return 1
    elif (x['G25a_6'] == 1) & (x['G25c_6'] ==2):
        return 3
    elif (x['G25a_6'] == 1) & (x['G25c_6'] ==8):
        return 3
    elif (x['G25a_6'] == 1) & (x['G25c_6'] ==9):
        return 3
    elif (x['G25a_6'] == 2):
        return 2
    elif (x['G25a_6'] == 8):
        return 2
    elif (x['G25a_6'] == 9):
        return 2


# In[10]:


def cervical(x):
    if (x['G25a_7'] == 1) & (x['G25c_7'] ==1):
        return 1
    elif (x['G25a_7'] == 1) & (x['G25c_7'] ==2):
        return 3
    elif (x['G25a_7'] == 1) & (x['G25c_7'] ==8):
        return 3
    elif (x['G25a_7'] == 1) & (x['G25c_7'] ==9):
        return 3
    elif (x['G25a_7'] == 2):
        return 2
    elif (x['G25a_7'] == 8):
        return 2
    elif (x['G25a_7'] == 9):
        return 2


# In[11]:


def lumbar(x):
    if (x['G25a_8'] == 1) & (x['G25c_8'] ==1):
        return 1
    elif (x['G25a_8'] == 1) & (x['G25c_8'] ==2):
        return 3
    elif (x['G25a_8'] == 1) & (x['G25c_8'] ==8):
        return 3
    elif (x['G25a_8'] == 1) & (x['G25c_8'] ==9):
        return 3
    elif (x['G25a_8'] == 2):
        return 2
    elif (x['G25a_8'] == 8):
        return 2
    elif (x['G25a_8'] == 9):
        return 2


# In[12]:


def alergia(x):
    if (x['G25a_9'] == 1) & (x['G25c_9'] ==1):
        return 1
    elif (x['G25a_9'] == 1) & (x['G25c_9'] ==2):
        return 3
    elif (x['G25a_9'] == 1) & (x['G25c_9'] ==8):
        return 3
    elif (x['G25a_9'] == 1) & (x['G25c_9'] ==9):
        return 3
    elif (x['G25a_9'] == 2):
        return 2
    elif (x['G25a_9'] == 8):
        return 2
    elif (x['G25a_9'] == 9):
        return 2


# In[13]:


def asma(x):
    if (x['G25a_10'] == 1) & (x['G25c_10'] ==1):
        return 1
    elif (x['G25a_10'] == 1) & (x['G25c_10'] ==2):
        return 3
    elif (x['G25a_10'] == 1) & (x['G25c_10'] ==8):
        return 3
    elif (x['G25a_10'] == 1) & (x['G25c_10'] ==9):
        return 3
    elif (x['G25a_10'] == 2):
        return 2
    elif (x['G25a_10'] == 8):
        return 2
    elif (x['G25a_10'] == 9):
        return 2


# In[14]:


def bronquitis(x):
    if (x['G25a_11'] == 1) & (x['G25c_11'] ==1):
        return 1
    elif (x['G25a_11'] == 1) & (x['G25c_11'] ==2):
        return 3
    elif (x['G25a_11'] == 1) & (x['G25c_11'] ==8):
        return 3
    elif (x['G25a_11'] == 1) & (x['G25c_11'] ==9):
        return 3
    elif (x['G25a_11'] == 2):
        return 2
    elif (x['G25a_11'] == 8):
        return 2
    elif (x['G25a_11'] == 9):
        return 2


# In[15]:


def diabetes(x):
    if (x['G25a_12'] == 1) & (x['G25c_12'] ==1):
        return 1
    elif (x['G25a_12'] == 1) & (x['G25c_12'] ==2):
        return 3
    elif (x['G25a_12'] == 1) & (x['G25c_12'] ==8):
        return 3
    elif (x['G25a_12'] == 1) & (x['G25c_12'] ==9):
        return 3
    elif (x['G25a_12'] == 2):
        return 2
    elif (x['G25a_12'] == 8):
        return 2
    elif (x['G25a_12'] == 9):
        return 2


# In[16]:


def ulcera(x):
    if (x['G25a_13'] == 1) & (x['G25c_13'] ==1):
        return 1
    elif (x['G25a_13'] == 1) & (x['G25c_13'] ==2):
        return 3
    elif (x['G25a_13'] == 1) & (x['G25c_13'] ==8):
        return 3
    elif (x['G25a_13'] == 1) & (x['G25c_13'] ==9):
        return 3
    elif (x['G25a_13'] == 2):
        return 2
    elif (x['G25a_13'] == 8):
        return 2
    elif (x['G25a_13'] == 9):
        return 2


# In[17]:


def urinaria(x):
    if (x['G25a_14'] == 1) & (x['G25c_14'] ==1):
        return 1
    elif (x['G25a_14'] == 1) & (x['G25c_14'] ==2):
        return 3
    elif (x['G25a_14'] == 1) & (x['G25c_14'] ==8):
        return 3
    elif (x['G25a_14'] == 1) & (x['G25c_14'] ==9):
        return 3
    elif (x['G25a_14'] == 2):
        return 2
    elif (x['G25a_14'] == 8):
        return 2
    elif (x['G25a_14'] == 9):
        return 2


# In[18]:


def colesterol(x):
    if (x['G25a_15'] == 1) & (x['G25c_15'] ==1):
        return 1
    elif (x['G25a_15'] == 1) & (x['G25c_15'] ==2):
        return 3
    elif (x['G25a_15'] == 1) & (x['G25c_15'] ==8):
        return 3
    elif (x['G25a_15'] == 1) & (x['G25c_15'] ==9):
        return 3
    elif (x['G25a_15'] == 2):
        return 2
    elif (x['G25a_15'] == 8):
        return 2
    elif (x['G25a_15'] == 9):
        return 2


# In[19]:


def cataratas(x):
    if (x['G25a_16'] == 1) & (x['G25c_16'] ==1):
        return 1
    elif (x['G25a_16'] == 1) & (x['G25c_16'] ==2):
        return 3
    elif (x['G25a_16'] == 1) & (x['G25c_16'] ==8):
        return 3
    elif (x['G25a_16'] == 1) & (x['G25c_16'] ==9):
        return 3
    elif (x['G25a_16'] == 2):
        return 2
    elif (x['G25a_16'] == 8):
        return 2
    elif (x['G25a_16'] == 9):
        return 2


# In[20]:


def dermatitis(x):
    if (x['G25a_17'] == 1) & (x['G25a_17'] ==1):
        return 1
    elif (x['G25a_17'] == 1) & (x['G25a_17'] ==2):
        return 3
    elif (x['G25a_17'] == 1) & (x['G25a_17'] ==8):
        return 3
    elif (x['G25a_17'] == 1) & (x['G25a_17'] ==9):
        return 3
    elif (x['G25a_17'] == 2):
        return 2
    elif (x['G25a_17'] == 8):
        return 2
    elif (x['G25a_17'] == 9):
        return 2


# In[21]:


def estrenimiento(x):
    if (x['G25a_18'] == 1) & (x['G25c_18'] ==1):
        return 1
    elif (x['G25a_18'] == 1) & (x['G25c_18'] ==2):
        return 3
    elif (x['G25a_18'] == 1) & (x['G25c_18'] ==8):
        return 3
    elif (x['G25a_18'] == 1) & (x['G25c_18'] ==9):
        return 3
    elif (x['G25a_18'] == 2):
        return 2
    elif (x['G25a_18'] == 8):
        return 2
    elif (x['G25a_18'] == 9):
        return 2


# In[22]:


def cirrosis(x):
    if (x['G25a_19'] == 1) & (x['G25c_19'] ==1):
        return 1
    elif (x['G25a_19'] == 1) & (x['G25c_19'] ==2):
        return 3
    elif (x['G25a_19'] == 1) & (x['G25c_19'] ==8):
        return 3
    elif (x['G25a_19'] == 1) & (x['G25c_19'] ==9):
        return 3
    elif (x['G25a_19'] == 2):
        return 2
    elif (x['G25a_19'] == 8):
        return 2
    elif (x['G25a_19'] == 9):
        return 2


# In[23]:


def otros_mental(x):
    if (x['G25a_22'] == 1) & (x['G25c_22'] ==1):
        return 1
    elif (x['G25a_22'] == 1) & (x['G25c_22'] ==2):
        return 3
    elif (x['G25a_22'] == 1) & (x['G25c_22'] ==8):
        return 3
    elif (x['G25a_22'] == 1) & (x['G25c_22'] ==9):
        return 3
    elif (x['G25a_22'] == 2):
        return 2
    elif (x['G25a_22'] == 8):
        return 2
    elif (x['G25a_22'] == 9):
        return 2


# In[24]:


def ictus(x):
    if (x['G25a_23'] == 1) & (x['G25c_23'] ==1):
        return 1
    elif (x['G25a_23'] == 1) & (x['G25c_23'] ==2):
        return 3
    elif (x['G25a_23'] == 1) & (x['G25c_23'] ==8):
        return 3
    elif (x['G25a_23'] == 1) & (x['G25c_23'] ==9):
        return 3
    elif (x['G25a_23'] == 2):
        return 2
    elif (x['G25a_23'] == 8):
        return 2
    elif (x['G25a_23'] == 9):
        return 2


# In[25]:


def migrana(x):
    if (x['G25a_24'] == 1) & (x['G25c_24'] ==1):
        return 1
    elif (x['G25a_24'] == 1) & (x['G25c_24'] ==2):
        return 3
    elif (x['G25a_24'] == 1) & (x['G25c_24'] ==8):
        return 3
    elif (x['G25a_24'] == 1) & (x['G25c_24'] ==9):
        return 3
    elif (x['G25a_24'] == 2):
        return 2
    elif (x['G25a_24'] == 8):
        return 2
    elif (x['G25a_24'] == 9):
        return 2


# In[26]:


def hemorroides(x):
    if (x['G25a_25'] == 1) & (x['G25c_25'] ==1):
        return 1
    elif (x['G25a_25'] == 1) & (x['G25c_25'] ==2):
        return 3
    elif (x['G25a_25'] == 1) & (x['G25c_25'] ==8):
        return 3
    elif (x['G25a_25'] == 1) & (x['G25c_25'] ==9):
        return 3
    elif (x['G25a_25'] == 2):
        return 2
    elif (x['G25a_25'] == 8):
        return 2
    elif (x['G25a_25'] == 9):
        return 2


# In[27]:


def tumores(x):
    if (x['G25a_26'] == 1) & (x['G25c_26'] ==1):
        return 1
    elif (x['G25a_26'] == 1) & (x['G25c_26'] ==2):
        return 3
    elif (x['G25a_26'] == 1) & (x['G25c_26'] ==8):
        return 3
    elif (x['G25a_26'] == 1) & (x['G25c_26'] ==9):
        return 3
    elif (x['G25a_26'] == 2):
        return 2
    elif (x['G25a_26'] == 8):
        return 2
    elif (x['G25a_26'] == 9):
        return 2


# In[28]:


def osteoporosis(x):
    if (x['G25a_27'] == 1) & (x['G25c_27'] ==1):
        return 1
    elif (x['G25a_27'] == 1) & (x['G25c_27'] ==2):
        return 3
    elif (x['G25a_27'] == 1) & (x['G25c_27'] ==8):
        return 3
    elif (x['G25a_27'] == 1) & (x['G25c_27'] ==9):
        return 3
    elif (x['G25a_27'] == 2):
        return 2
    elif (x['G25a_27'] == 8):
        return 2
    elif (x['G25a_27'] == 9):
        return 2


# In[29]:


def tiroides(x):
    if (x['G25a_28'] == 1) & (x['G25c_28'] ==1):
        return 1
    elif (x['G25a_28'] == 1) & (x['G25c_28'] ==2):
        return 3
    elif (x['G25a_28'] == 1) & (x['G25c_28'] ==8):
        return 3
    elif (x['G25a_28'] == 1) & (x['G25c_28'] ==9):
        return 3
    elif (x['G25a_28'] == 2):
        return 2
    elif (x['G25a_28'] == 8):
        return 2
    elif (x['G25a_28'] == 9):
        return 2


# In[30]:


def rinon(x):
    if (x['G25a_29'] == 1) & (x['G25c_29'] ==1):
        return 1
    elif (x['G25a_29'] == 1) & (x['G25c_29'] ==2):
        return 3
    elif (x['G25a_29'] == 1) & (x['G25c_29'] ==8):
        return 3
    elif (x['G25a_29'] == 1) & (x['G25c_29'] ==9):
        return 3
    elif (x['G25a_29'] == 2):
        return 2
    elif (x['G25a_29'] == 8):
        return 2
    elif (x['G25a_29'] == 9):
        return 2


# In[31]:


def prostata(x):
    if (x['G25a_30'] == 1) & (x['G25c_30'] ==1):
        return 1
    elif (x['G25a_30'] == 1) & (x['G25c_30'] ==2):
        return 3
    elif (x['G25a_30'] == 1) & (x['G25c_30'] ==8):
        return 3
    elif (x['G25a_30'] == 1) & (x['G25c_30'] ==9):
        return 3
    elif (x['G25a_30'] == 2):
        return 2
    elif (x['G25a_30'] == 8):
        return 2
    elif (x['G25a_30'] == 9):
        return 2


# In[32]:


def meonopausia(x):
    if (x['G25a_31'] == 1) & (x['G25c_31'] ==1):
        return 1
    elif (x['G25a_31'] == 1) & (x['G25c_31'] ==2):
        return 3
    elif (x['G25a_31'] == 1) & (x['G25c_31'] ==8):
        return 3
    elif (x['G25a_31'] == 1) & (x['G25c_31'] ==9):
        return 3
    elif (x['G25a_31'] == 2):
        return 2
    elif (x['G25a_31'] == 8):
        return 2
    elif (x['G25a_31'] == 9):
        return 2


# In[33]:


def lesiones(x):
    if (x['G25a_32'] == 1) & (x['G25c_32'] ==1):
        return 1
    elif (x['G25a_32'] == 1) & (x['G25c_32'] ==2):
        return 3
    elif (x['G25a_32'] == 1) & (x['G25c_32'] ==8):
        return 3
    elif (x['G25a_32'] == 1) & (x['G25c_32'] ==9):
        return 3
    elif (x['G25a_32'] == 2):
        return 2
    elif (x['G25a_32'] == 8):
        return 2
    elif (x['G25a_32'] == 9):
        return 2


# In[34]:


def act_diaria(x):
    if (x['L39_1'] == 2) | (x['L39_1'] ==3):
        return 1
    elif (x['L39_3'] == 2) | (x['L39_3'] ==3):
        return 1
    elif (x['L39_4'] == 2) | (x['L39_4'] ==3):
        return 1
    elif (x['L39_5'] == 2) | (x['L39_5'] ==3):
        return 1
    else:
        return 2


# In[35]:


def act_hogar(x):
    if (x['L42_1'] == 2) | (x['L42_1'] ==3):
        return 1
    elif (x['L42_2'] == 2) | (x['L42_2'] ==3):
        return 1
    elif (x['L42_3'] == 2) | (x['L42_3'] ==3):
        return 1
    elif (x['L42_4'] == 2) | (x['L42_4'] ==3):
        return 1
    elif (x['L42_5'] == 2) | (x['L42_5'] ==3):
        return 1
    elif (x['L42_6'] == 2) | (x['L42_6'] ==3):
        return 1
    elif (x['L42_7'] == 2) | (x['L42_7'] ==3):
        return 1
    else:
        return 2

