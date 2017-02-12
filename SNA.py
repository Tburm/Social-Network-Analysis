execfile("U:/Kaggle/SNA/SNA_Funcs.py")

n_path = 'U:/Kaggle/SNA/egonets/'
n_listing = os.listdir(n_path)
n_files = ['U:/Kaggle/SNA/egonets/' + str(fname) for fname in n_listing]

c_path = 'U:/Kaggle/SNA/Training/'
c_listing = os.listdir(c_path)
c_files = ['U:/Kaggle/SNA/Training/' + str(fname) for fname in c_listing]

c_funs = [['Eig',nx.eigenvector_centrality], ['Betweenness', nx.betweenness_centrality],
          ['Load', nx.load_centrality], ['Degree', nx.degree_centrality],
          ['Closeness', nx.closeness_centrality]]

MCirc = MultipleCircles(c_files)
MCent = MultipleCent(n_files, c_funs)
Features = ParseFeatures("U:/Kaggle/SNA/featureList.txt")
FeatureList = ParseFeatureList("U:/Kaggle/SNA/features.txt")


clf, importance, Acc, Prediction = ForestTest(MCirc, MCent)

importance = sorted(importance, reverse=False)

imp_lab = [x[1] for x in importance]
imp_val = [x[0] for x in importance]

numTests = len(imp_lab)
testNames = imp_lab
scores = imp_val

fig, ax1 = plt.subplots(figsize=(9, 7))
plt.subplots_adjust(left=0.115, right=0.88)
pos = np.arange(numTests)+0.5    # Center bars on the Y-axis ticks
rects = ax1.barh(pos, scores, align='center', height=0.5, color='m')

ax1.axis([0, .5, 0, len(imp_lab)])
pylab.yticks(pos, testNames)

plt.show()


totNodes = [val for val in MCirc.keys() if val in MCent.keys()]

X_Net, Nodes, Edges = ParseEgonet("U:/Kaggle/SNA/egonets/239.egonet")
X_Circ = ParseCircles("U:/Kaggle/SNA/Training/239.circles")

nx.draw(X_Net)

c_list = []
for i in Nodes:
    if i in X_Circ:
        if len(X_Circ[i]) == 1:
            c_list.extend(X_Circ[i])
        else:
            c_list.append(1)
    else:
        c_list.append(0)


nx.draw(X_Net, node_color = c_list)


conn_list = nx.connected_components(X_Net)
conn_dict = dict()
for i in range(len(conn_list)):
    for j in conn_list[i]:
        conn_dict[j] = i + 1

c_list = []
for i in Nodes:
    if i in conn_dict:
        c_list.append(conn_dict[i])

nx.draw(X_Net, node_color = c_list)

# Graph based on standardized centrality
Cent_dict = nx.eigenvector_centrality(X_Net)
Cent_list = [[],[]]
for i in Nodes:
    Cent_list[0].append(i)
    Cent_list[1].append(Cent_dict[i])

St_cent = dict()
for node in range(len(Cent_list[0])):
        St_cent[Cent_list[0][node]] = Cent_list[1][node]

c_list = []
for i in Nodes:
    if i in St_cent:
        c_list.append(St_cent[i])

nx.draw(X_Net, node_color = c_list)

# Standardize the centrality of each set of connected components separately
for i in conn_list:
    to_st = []
    for node in i:
        to_st.append(Cent_dict[node])
    to_st = list(preprocessing.scale(to_st))
        
    for node in range(len(i)):
        Cent_dict[i[node]] = to_st[node]

c_list = []
for i in Nodes:
    if i in Cent_dict:
        c_list.append(Cent_dict[i])

nx.draw(X_Net, node_color = c_list)



matches = dict()
for i in Edges:
    if (i[0],i[1]) in matches:
        next
    else:
        dict1 = FeatureList[i[0]]
        dict2 = FeatureList[i[1]]
        
        cnt = 0
        for j in Features:
            if j in dict1 and j in dict2:
                if dict1[j] == dict2[j]:
                    cnt += 1
        
        matches[(i[0],i[1])] = cnt


hist(matches.values(), bins = 16)

highMatch = dict()
for i in matches:
    if matches[i] >= 5:
        highMatch[i] = matches[i]
        
highEdge = highMatch.keys()
highNode = []
for i in highEdge:
    highNode.extend(i)
highNode = list(set(highNode))

G = nx.Graph()
G.add_nodes_from(highNode)
G.add_edges_from(highEdge)
nx.draw(G)

#Display the features and numbers within the list to filter by hypotheses
#cnt = 0
#for i in Features:
#    print "Feature " + str(cnt) + ": " + i
#    cnt += 1

circ_nodes = X_Circ.keys()
random.shuffle(circ_nodes)
train = [1] * int(len(circ_nodes) * .8) + [0] *  (len(circ_nodes) - int(len(circ_nodes) * .8))

# Filter the list of features to the defined set
keep = [2,4,6,8,10,12,14,15,17,20,21,23,24,27,29,33,34,36,39,41,43,47,49,52,55]
#keep = [12,14,17,20,21,24,27,34,39,43]
filt_Feat = []
for i in keep:
    filt_Feat.append(Features[i])

svc_input = np.zeros([len(circ_nodes), len(filt_Feat)], float)



input_df['id']

for feat in filt_Feat:
    f_list = []
    for node in circ_nodes:
        if feat in FeatureList[node]:
            f_list.append(FeatureList[node][feat])
        else:
            f_list.append('-1')
    input_df[feat] = f_list


node_Class = []
for node in train_Nodes:
    node_Class.append(str(X_Circ[node][0]))

svc_input = np.zeros([len(train_Nodes), len(keep)], float)

Cent_dict = nx.eigenvector_centrality(X_Net)
Cent_Color = []
for i in Nodes:
    Cent_Color.append(Cent_dict[i])

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

cnt = 0
for node in train_Nodes:
    data = []
    for feat in filt_Feat:
        if feat == 'id':
            continue
        elif feat in FeatureList[node]:
            dataAdd = FeatureList[node][feat]
            data.append(dataAdd)
        else:
            data.append(-1)
    data.append(Scaled_cent[node]*10)
        
    svc_input[cnt] = data
    cnt += 1


svc = svm.SVC(kernel='rbf')
svc.fit(svc_input, node_Class)

pred_list = []
for node in test_Nodes:
    actual = X_Circ[node]
    
    data = []
    for feat in filt_Feat:
        if feat == 'id':
            continue
        elif feat in FeatureList[node]:
            dataAdd = FeatureList[node][feat]
            data.append(str(dataAdd))
        else:
            data.append(-1)
    data.append(Scaled_cent[node])  
    
    pred = svc.predict(data)[0]
    pred_list.append([node, actual, pred])

C_cnt = 0
F_cnt = 0
for i in pred_list:
    if int(i[2]) in i[1]:
        C_cnt += 1
    else:
        F_cnt += 1
C_cnt
F_cnt


nx.draw(X_Net, node_color = Cent_Color)


cnt = 0
svc_input = np.zeros([len(train_Nodes), 1], float)
for node in train_Nodes:
    data = [Scaled_cent[node]]
        
    svc_input[cnt] = data
    cnt += 1

pred_list = []
for node in test_Nodes:
    actual = X_Circ[node]
    
    data = [Scaled_cent[node]] 
    
    pred = svc.predict(data)[0]
    pred_list.append([node, actual, pred])


C, F, Acc = ForestTest(X_Circ)
C, F, Acc

    
import pyper
from pyper import *

output = runR('a=3')




