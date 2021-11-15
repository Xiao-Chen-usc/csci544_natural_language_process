#!/usr/bin/env python
# coding: utf-8

# task1

# In[1]:


import pandas as pd
import numpy as np
import collections


# In[2]:


data = pd.read_csv('train',sep='\t',names = ['index','word','tag'])


# In[3]:


ttt = data.word.tolist()


# In[4]:


ttt_dict = collections.defaultdict(int)
for i in ttt:
    ttt_dict[i] = ttt_dict[i]+1


# In[5]:


temp_list = []
for w,o in ttt_dict.items():
    temp = [w,int(o)]
    temp_list.append(temp)




# In[7]:


def occrence(e):
    return e[1]
temp_list.sort(reverse=True, key = occrence)


# In[8]:




# In[10]:


big_list = []
low_frequency = set()


# In[11]:


i = 1
unknown = ['<unk>',0,0]
for w,o in temp_list:
    if o >= 2:
        temp = [w,i,o]
        big_list.append(temp)
        i +=1
    else:
        low_frequency.add(w)
        unknown[2] += o
    


# In[12]:


big_list.insert(0,unknown)




# #### What is the selected threshold for unknown words replacement?
# 
# #### Answer:threshold for unknown words replacement is 4
# 
# #### What is the total size of your vocabulary
# 
# #### Answer:the size of my vocabulary is 13751
# 
# #### unknown number is 42044

# In[15]:


with open('vocab.txt','w') as f:
    for i in big_list:
        a = str(i[0]) + '\t' + str(i[1])+'\t'+str(i[2])+'\n'
        f.write(a)




data_np = data.to_numpy()


# In[19]:


#creat a new list
new_list = []
for i in data_np:
    if i[1] in low_frequency:
        new_list.append([i[0],'<unk>',i[2]])
    else:
        new_list.append([i[0],i[1],i[2]])





# convert the list into sentence
sentence_list = []

for i in range(len(new_list)):
    if new_list[i][0] == 1:
        temp = []
        temp.append(new_list[i])
    else:
        temp.append(new_list[i])
    if ((i+1) < len(new_list)) and new_list[i+1][0] == 1:
        sentence_list.append(temp)
    
        




t_dict = collections.defaultdict(int)
e_dict = collections.defaultdict(int)
tag_dict = collections.defaultdict(int)


# In[24]:


for sentence in sentence_list:
    for i in range(len(sentence)):
        e_dict[sentence[i][2],sentence[i][1]] +=1
        tag_dict[sentence[i][2]] +=1 
        if sentence[i][0] == 1:
            t_dict[('<s>',sentence[i][2])] += 1
        else:
            t_dict[(sentence[i-1][2],sentence[i][2])] += 1
        


# In[25]:


emission_dict = {}


# In[26]:



for key,value in e_dict.items():
    emission_dict[key] = value/tag_dict[key[0]]


# In[27]:


transitary_dict = {}


# In[31]:


sentence_num = len(sentence_list)


# In[32]:


for key,value in t_dict.items():
    if key[0] == '<s>':
        transitary_dict [key] = value/sentence_num
    else:
        transitary_dict [key] = value / tag_dict[key[0]]


# In[33]:


transitary_dict 


# In[34]:


sentence_num


# In[35]:


json_transitary_dict = {}
for key,value in transitary_dict.items():
    json_transitary_dict[str(key)] = value


# In[36]:


json_transitary_dict


# In[37]:


json_emission_dict = {}
for key,value in emission_dict.items():
    json_emission_dict[str(key)] = value



# In[39]:


import json


# In[40]:


final_json = {"transition":json_transitary_dict,"emission":json_emission_dict}



out_file = open("hmm.json", "w")
json.dump(final_json, out_file, indent = 6)
  
out_file.close()


# ## hmm greedy

# In[43]:


#use sentence_list to appliment hmm greedy


# In[44]:


word_tag = collections.defaultdict(set)


# In[45]:


for sentence in sentence_list:
    for i in sentence:
        word_tag[i[1]].add(i[2])


# In[46]:


word_tag = dict(word_tag)




# In[49]:



def greedy_hmm(sentence):
    res = []
    for i in range(len(sentence)):
        target_word = sentence[i][1]
        probility_tag = []
        if target_word in word_tag:
            tag_list = list(word_tag[target_word])
        else:
            target_word = '<unk>'
            tag_list = list(word_tag['<unk>'])
        if sentence[i][0] == 1:
            for tag in tag_list:
                if ('<s>',tag) in transitary_dict:
                    t = transitary_dict[('<s>',tag)]
                else: t = 0
                    
                if (tag,target_word) in emission_dict:
                    e = emission_dict[(tag,target_word)]
                else:
                    e = 0
                probility = t*e
                probility_tag.append(probility)

        else:
            for tag in tag_list:
                if (res[i-1],tag) in transitary_dict:
                    t = transitary_dict[(res[i-1],tag)]
                else: t = 0
                    
                if (tag,target_word) in emission_dict:
                    e = emission_dict[(tag,target_word)]
                else:
                    e = 0
                probility = t*e
                probility_tag.append(probility)
        i_tag = tag_list[probility_tag.index(max(probility_tag))]
        res.append(i_tag)
    return res
        


# In[50]:


correct = 0
total = 0
for sentence in sentence_list:
    res_tag = greedy_hmm(sentence)
    for i in range(len(res_tag)):
        total += 1
        if res_tag[i] == sentence[i][2]:
            correct += 1


# In[51]:


correct/total


# ### test greedy hmm

# In[52]:


test = pd.read_csv('dev',sep='\t',names = ['index','word','tag'])


# In[53]:


test_np = test.to_numpy()


# In[54]:


#creat a test list
new_test = []
for i in test_np:
    if i[1] in low_frequency:
        new_test.append([i[0],'<unk>',i[2]])
    else:
        new_test.append([i[0],i[1],i[2]])




# convert the list into sentence
test_sentence_list = []

for i in range(len(new_test)):
    if new_test[i][0] == 1:
        temp = []
        temp.append(new_test[i])
    else:
        temp.append(new_test[i])
    if ((i+1) < len(new_test)) and new_test[i+1][0] == 1:
        test_sentence_list.append(temp)
    





correct = 0
total = 0
for sentence in test_sentence_list:
    res_tag = greedy_hmm(sentence)
    for i in range(len(res_tag)):
        total += 1
        if res_tag[i] == sentence[i][2]:
            correct += 1


# In[59]:



accuracy_greedy_hmm = correct / total 


# In[60]:


accuracy_greedy_hmm


# ### produce greedy.out

# In[61]:


out = pd.read_csv('test',sep='\t',names = ['index','word'])


# In[62]:


out_np = out.to_numpy()


# In[63]:


#creat a test list
new_out = []
for i in out_np:
    if i[1] in low_frequency:
        new_out.append([i[0],'<unk>'])
    else:
        new_out.append([i[0],i[1]])


# In[64]:


# convert the list into sentence
out_sentence_list = []

for i in range(len(new_out)):
    if new_out[i][0] == 1:
        temp = []
        temp.append(new_out[i])
    else:
        temp.append(new_out[i])
    if ((i+1) < len(new_out)) and new_out[i+1][0] == 1:
        out_sentence_list.append(temp)
    if i == len(new_out)-1:
        out_sentence_list.append(temp)
            
    


# In[65]:


import copy

w_out_sentence_list = copy.deepcopy(out_sentence_list)

w_out_sentence_list

out_res = []
for sentence in out_sentence_list:
    res_tag = greedy_hmm(sentence)
    out_res.append(res_tag)

out_res

for i in range(len(out_res)):
    for i1 in range(len(out_res[i])):
        w_out_sentence_list[i][i1].append(out_res[i][i1])

w_out_sentence_list


# In[66]:


with open('greedy_out.txt','w') as f:
    for sentence in range(len(w_out_sentence_list)):
        if sentence != 0:
            f.write('\n')
        for (i, w, t) in w_out_sentence_list[sentence]:
            f.write(str(i))
            f.write('\t')
            f.write(str(w))
            f.write('\t')
            f.write(str(t))
            f.write('\n')


# In[67]:


sentence_list[77][0:3]


# ## viterbi

# In[68]:



def viterbi_hmm(sentence):
    res = []
    for i in range(len(sentence)):
        target_word = sentence[i][1]
        probility_tag = {}
        if target_word in word_tag:
            tag_list = list(word_tag[target_word])
            
        else:
            target_word = '<unk>'
            tag_list = list(word_tag['<unk>'])
        if sentence[i][0] == 1:
            for tag in tag_list:
                if ('<s>',tag) in transitary_dict:
                    t = transitary_dict[('<s>',tag)]
                else: t = 0
                    
                if (tag,target_word) in emission_dict:
                    e = emission_dict[(tag,target_word)]
                else:
                    e = 0
                probility = t*e
                probility_tag[tag] = ('<s>',probility)

        else:
            for tag in tag_list:
                previous_tag_list = []
                for previous_tag in res[i-1]:
                    if (previous_tag,tag) in transitary_dict:
                        t = transitary_dict[(previous_tag,tag)]
                    else: t = 0

                    if (tag,target_word) in emission_dict:
                        e = emission_dict[(tag,target_word)]
                    else:
                        e = 0
                    probility = t*e*res[-1][previous_tag][1]
                    previous_tag_list.append((previous_tag,probility))
                previous_tag_list = sorted(previous_tag_list,key = lambda x:x[1],reverse = True)  
                probility_tag[tag] = previous_tag_list[0]
        res.append(probility_tag)
        
    return res
        


# In[69]:


hhh = viterbi_hmm(sentence_list[77])


# In[ ]:


hhh


# In[ ]:


def backtrace(table):
    tag_backtrace = []
    length = len(table)
    i = length -1
    end_col = table[i]
    end_tag = max(end_col, key=lambda key: end_col[key][1])
    tag_backtrace.append(end_tag)
    if i!=0:
        previous_tag = end_col[end_tag][0]
    i -= 1
    while i >= 0:
        tag_backtrace.append(previous_tag)
        previous_tag_col = table[i][previous_tag]
        i -= 1
        if i>= 0:
            previous_tag = previous_tag_col[0]
    tag_backtrace = list(reversed(tag_backtrace))
    return tag_backtrace
        
        


# In[ ]:


np.array(sentence_list[77])[:,-1]




# In[ ]:


table = viterbi_hmm(sentence_list[0])





correct = 0
total = 0
for sentence in test_sentence_list:
    table = viterbi_hmm(sentence)
    res_tag = backtrace(table)
    for i in range(len(res_tag)):
        total += 1
        if res_tag[i] == sentence[i][2]:
            correct += 1


# In[ ]:


hmm_viterbi_accuracy = correct/total


# In[ ]:


viterbi_hmm(sentence_list[77][0:2])


# hmm_viterbi_accuracy

# In[ ]:


hmm_viterbi_accuracy


# In[ ]:


w_out_sentence_list = copy.deepcopy(out_sentence_list)

w_out_sentence_list

out_res = []
for sentence in out_sentence_list:
    table = viterbi_hmm(sentence)
    res_tag = backtrace(table)
    out_res.append(res_tag)


for i in range(len(out_res)):
    for i1 in range(len(out_res[i])):
        w_out_sentence_list[i][i1].append(out_res[i][i1])


# In[ ]:


with open('viterbi_out.txt','w') as f:
    for sentence in range(len(w_out_sentence_list)):
        if sentence != 0:
            f.write('\n')
        for (i, w, t) in w_out_sentence_list[sentence]:
            f.write(str(i))
            f.write('\t')
            f.write(str(w))
            f.write('\t')
            f.write(str(t))
            f.write('\n')


# In[ ]:


print('The accuracy for greedy hmm is '+str(accuracy_greedy_hmm)+'\n The accuracy for viterbi is '+str(hmm_viterbi_accuracy))


# In[ ]:




