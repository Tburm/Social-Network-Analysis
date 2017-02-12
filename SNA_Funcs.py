import networkx as nx
#import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import os
import pylab


#G = nx.Graph()
#
#N = [1,2,3,4,5]
#E = [(1,2), (1,3), (1,4), (2,5), (3,5)]
#
#G.add_nodes_from(N)
#G.add_edges_from(E)
#
#nx.draw(G)

# Function to parse an egonet file into a graph object
# Input is the location of an egonet file from the Kaggle data
def ParseEgonet(file_name):
    f = open(file_name)

    # Split file into list for each node
    # First item is the name of the node, second item is the list of connections to that node
    net = []
    for i in f:
        k = i.split(":")
        k[1] = k[1].split(" ")
        k[1] = k[1][1:len(k[1])-1]
        
        k[0] = int(k[0])
        k[1] = [int(x) for x in k[1]]
        net.append(k)
    
    # Create list of all nodes from full list
    Nodes = []    
    for i in net:
        Nodes.append(i[0])
        
    # Create list of tuples
    # Each tuple contains a node a connection
    Edges = []
    for j in net:
        edge_add = []    
        node = j[0]
        conn = j[1]
        for k in conn:
            new_e = (node,k)
            edge_add.append(new_e)
        Edges.extend(edge_add)
    
    # Store the information in a graph object from networkx
    G = nx.Graph()
    G.add_nodes_from(Nodes)
    G.add_edges_from(Edges)
    
    return(G, Nodes, Edges)

# Create a list of all feature names
def ParseFeatures(file_name):
    f = open(file_name)
    feat_names = []
    for i in f:
        i = i.split("\n")
        feat_names.append(i[0])
    return(feat_names)

# Create a list of dictionaries that contain features for each node
# The index within the list is equal to the id num of the node
def ParseFeatureList(file_name):
    f = open(file_name)
    node_Feat = []
    for j in f:
        j = j.split("\n")
        j = j[0]
        
        j = j.split(" ")
        
        feats = []
        for k in j[1:len(j)]:
            att_sp = k.split(";")
            
            att_len = len(att_sp[len(att_sp)-1])
            att_val = int(att_sp[len(att_sp)-1])
            
            att_name = k[0:len(k)-att_len-1]
            
            feats.append((att_name, att_val))
        
        node_Feat.append(dict(feats))
    return(node_Feat)

# Parse a circle file into a dictionary
# Each key is a Node, and the value is a list of their circles
def ParseCircles(file_name): 
    f = open(file_name)
    
    net = []
    cnt = 1
    for i in f:
        k = i.split(":")
        k[1] = k[1].split(" ")
        k[1] = k[1][1:len(k[1])-1]
            
        k[0] = int(k[0][6:])
        k[1] = [int(x) for x in k[1]]
        net.append(k)
        cnt += 1

    members = dict()
    for j in net:
        for k in j[1]:
            if k in members:
                members[k].append(j[0])
            else:
                members[k] = [j[0]]
                
    return(members)

def ForestTest(Circles, Cent):
    # Trying random forest
    cent_nodes = []    
    for x in Cent:
        cent_nodes.extend(Cent[x].keys())
    cent_nodes = list(set(cent_nodes))
        
    circ_nodes = [val for val in Circles.keys() if val in Circles.keys() and val in cent_nodes]
    random.shuffle(circ_nodes)
    
    # Filter the list of features to the defined set
    keep = [2,4,6,8,10,12,14,15,17,20,21,23,24,27,29,33,34,36,39,41,43,47,49,52,55]
    #keep = [12,14,17,20,21,24,27,34,39,43]
    
    filt_Feat = [Features[feat] for feat in keep]  
    
    svc_input = np.zeros([len(circ_nodes), len(filt_Feat)], float)
    
    input_df = pd.DataFrame(svc_input, columns = filt_Feat)
    
    for feat in filt_Feat:
        f_list = []
        for node in circ_nodes:
            if feat in FeatureList[node]:
                f_list.append(FeatureList[node][feat])
            else:
                f_list.append('-1')
        input_df[feat] = f_list
    
    node_Class = [str(Circles[node][0]) for node in input_df['id']]
    input_df['class'] = node_Class
    
    train = [1] * int(len(circ_nodes) * .8) + [0] *  (len(circ_nodes) - int(len(circ_nodes) * .8))
    input_df['train'] = train
    
    
    #    Cent_dict = nx.eigenvector_centrality(Network)
    #    cent_input = np.zeros([len(Nodes), 2], float)
    #    cnt = 0
    #    for i in Cent_dict:
    #        cent_input[cnt] = [i,Cent_dict[i]]
    #        cnt += 1
    #    
    #    new_cent = preprocessing.scale(list(cent_input[:,1]))
    #    cent_input[:,1] = new_cent
    #    Scaled_cent = dict()
    #    for i in cent_input:
    #        Scaled_cent[int(i[0])] = i[1] 
      
    for c in Cent.keys():
        cent_input = []
        for node in input_df['id']:
            try:
                cent_input.append(Cent[c][node])
            except KeyError:
                cent_input.append(0)
        input_df[c] = cent_input

    for col in filt_Feat:
        input_df[col] = input_df[col].astype(str)
        
    feature_In = [feat for feat in filt_Feat if feat != 'id'] + Cent.keys()
    
    train, test = input_df[input_df['train']==True], input_df[input_df['train']==False]
    
    clf = RandomForestClassifier(n_jobs = 2, n_estimators = 200)
    clf.fit(train[feature_In], train['class'])
    
    pred_list = list(clf.predict(test[feature_In]))
    preds = []
    cnt = 0
    for i in test['id']:
        data = [i, list(test['class'])[cnt], pred_list[cnt]]
        preds.append(data)
        cnt += 1
        
    C_cnt = 0
    F_cnt = 0
    for i in preds:
        if i[2] == i[1]:
            C_cnt += 1
        else:
            F_cnt += 1
    Acc = C_cnt/(C_cnt + F_cnt)
    
    var_imp = list(clf.feature_importances_)
    imp_feat = []
    for feat in range(len(feature_In)):
        imp_feat.append((var_imp[feat], feature_In[feat]))
    
    return(clf, imp_feat, Acc, preds)
    #return(C_cnt, F_cnt, C_cnt/(C_cnt + F_cnt))

def MultipleCircles(CircFiles):
    allCirc = dict()

    for fname in CircFiles:
        print fname
        X_Circ = ParseCircles(fname)
        allCirc = dict(allCirc.items() + X_Circ.items())
    
    return(allCirc)

def MultipleCent(NetFiles, c_funs):
    allCent = dict()
    names = [x[0] for x in c_funs]
    for name in names:
        allCent[name] = dict()

    for fname in NetFiles:
        print fname
        Network, Nodes , Edges =  ParseEgonet(fname)
        
        for x in c_funs:
            fun = x[1]
            name = x[0]
            print name
            
            try:
                Cent_dict = fun(Network)
                cent_input = np.zeros([len(Nodes), 2], float)
                cnt = 0
                for i in Cent_dict:
                    cent_input[cnt] = [i,Cent_dict[i]]
                    cnt += 1
                
                new_cent = preprocessing.scale(list(cent_input[:,1]))
                cent_input[:,1] = new_cent
                Scaled_cent = dict()
                for i in cent_input:
                    Scaled_cent[int(i[0])] = i[1]
                
                allCent[name] = dict(allCent[name].items() + Scaled_cent.items())
            
            except Exception:
                for i in Nodes:
                    Scaled_cent = dict()
                    Scaled_cent[i] = 0
                allCent[name] = dict(allCent[name].items() + Scaled_cent.items())
            
    return(allCent)
    
    
    
    
    