#!/usr/bin/env python
# coding: utf-8

# In[29]:


import tkinter as tk
from tkinter import filedialog
import pandas as pd
import pickle
import pandas as pd
import numpy as np
import esm
import torch
import os


def app_function(sequence_list):
    # 在这里添加预处理数据的代码，根据您的需求进行数据处理
    peptide_sequence_list = []
    for seq in sequence_list:
        format_seq = [seq, seq]
        tuple_sequence = tuple(format_seq)
        peptide_sequence_list.append(tuple_sequence)
    #load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    
    #load data
    data = peptide_sequence_list
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    
    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=True)
    token_representations = results["representations"][6]
    
    # Generate per-sequence representations via averaging
    sequence_representations = []
    for i, token_len in enumerate(batch_lens):
        each_seq_rep = token_representations[i, 1:token_len - 1].mean(0).tolist()  
        sequence_representations.append(each_seq_rep)

    embedding_results = pd.DataFrame(sequence_representations)
    
    # 拼接ESMSVM.pkl文件的完整路径
    model_path = os.path.join(r'C:\Users\RH\A_try_change\ESM320_clean\app_development', 'ESMSVM.pkl')
    
    # 加载训练好的模型
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)

    # 使用加载的模型进行预测
    predictions = loaded_model.predict(embedding_results)
    decision_values = loaded_model.decision_function(embedding_results)
    probabilities = 1 / (1 + np.exp(-decision_values))
    

    # 处理预测结果和概率
    result = []
    for i in range(len(predictions)):
        if predictions[i] == 0:
            result.append(['non_activate', 'N'])
        elif predictions[i] == 1:
            result.append(['activate', probabilities[i]])

    # 清空文本框并插入预测结果
    output_text.delete(1.0, tk.END)
    for res in result:
        output_text.insert(tk.END, f"Prediction: {res[0]}\tProbability: {res[1]}\n")


def open_excel_file():
    # 打开文件对话框，选择Excel文件
    file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])

    # 在文本框中显示选择的文件路径
    excel_path_text.delete(0, tk.END)
    excel_path_text.insert(tk.END, file_path)


def predict_sequences():
    # 获取Excel文件路径
    excel_path = excel_path_text.get()

    # 读取Excel文件
    dataset = pd.read_excel(excel_path, na_filter=False)
    sequence_list = dataset['Sequence'].tolist()

    # 执行预测功能
    app_function(sequence_list)


def predict_single_sequence():
    # 获取输入文本框中的序列
    sequence = input_text.get()

    # 执行预测功能
    app_function([sequence])


# 创建主窗口
window = tk.Tk()

# Excel文件路径文本框
excel_path_text = tk.Entry(window)
excel_path_text.pack()

# 打开文件按钮
open_file_button = tk.Button(window, text="Open File", command=open_excel_file)
open_file_button.pack()

# 预测Excel文件按钮
predict_file_button = tk.Button(window, text="Predict Excel", command=predict_sequences)
predict_file_button.pack()

# 输入文本框
input_text = tk.Entry(window)
input_text.pack()

# 预测单独序列按钮
predict_button = tk.Button(window, text="Predict Sequence", command=predict_single_sequence)
predict_button.pack()

# 输出文本框
output_text = tk.Text(window)
output_text.pack()

# 运行主窗口
window.mainloop()


# In[ ]:




