import shutil
import ogb.utils as utils
import random
import os 
import gzip
from ogb.graphproppred import PygGraphPropPredDataset

# all = open('Pipeline/data/ogbg_molesol/mapping/esol.csv', 'r').read().split("\n")[1:]
# smiles = []
path = "Pipeline/data/QM9/QM9_pyg/raw/"




# exit()
# ds = PygGraphPropPredDataset(name = "ogbg-molesol")
# print(ds[0].y)
# print(len(ds))
# for i in all:
#     temp = i.split(",")
#     if temp[-1][-1] == "\"" or temp[-1][-2:] == " \"":
#         for j in range(len(temp)):
#             if temp[j][0] != "\"":
#                 continue
#             smiles.append(temp[1:j][0])    
#             break
#     else: smiles.append(temp[-2])
                
smiles = open("Data/QM9smiles.txt", "r").read().split("\n")
# smiles =[i.split(",")[-2] for i in open('Pipeline/data/ogbg_molesol/mapping/esol.csv', 'r').read().split("\n")]

# print(smiles[:1000])

# print(utils.mol.smiles2graph(smiles[1000]))
print("smiles: ",len(smiles))
# res = []
edge_feat = []
edge = []
graph_label = []
node_feat = []
num_edge_list = []
num_node_list = []
total = 0
for i in range(len(smiles)-1):
    temp  = utils.mol.smiles2graph(smiles[i])
    edge_feat.append(temp["edge_feat"])
    
    temp_res = []
    for j in range(0,len(temp["edge_index"][0]),2):
        temp_res.append([temp["edge_index"][0][j], temp["edge_index"][1][j]])
    edge.append(temp_res)
    graph_label.append(1)
    node_feat.append(temp['node_feat'])
    num_edge_list.append(len(temp['edge_index'][0])/2)
    num_node_list.append(temp['num_nodes'])   

# f= open("file.txt", "w")
# ori = open("Pipeline/data/ogbg_molesol/raw/edge-feat.csv", "r").read().split("\n")[:-1]
# print("len: ",len(ori))
res = []
total_counter = 0
ori_counter = 0
for i in range(len(edge_feat)):
    for j in range(len(edge_feat[i])-1):
        # temp = ori[ori_counter].split(",")
        if j%2 == 0:
            ori_counter += 1
            # if int(temp[0]) != edge_feat[i][j][0] or int(temp[1]) != edge_feat[i][j][1] or int(temp[2]) != edge_feat[i][j][2]:
            #     print("error: ",i,j)
            res.append(edge_feat[i][j])
            # f.write(str(edge_feat[i][j][0])+","+str(edge_feat[i][j][1])+","+str(edge_feat[i][j][2]) + "\n")
        total_counter += 1

# edge_feat = res
# f = open(path+"edge-feat.csv", "w")
# for i in edge_feat:
#     for j in i:
#         f.write(str(j)+",")    
#     f.write("\n")
# with open(path+"edge-feat.csv", 'rb') as f_in, gzip.open(path+"edge-feat.csv.gz", 'wb') as f_out:
#     shutil.copyfileobj(f_in, f_out)
# os.remove(path+"edge-feat.csv")

# f = open(path+"edge.csv", "w")
# for i in edge:
#     for j in i:
#         f.write(str(j[0])+","+str(j[1])+ "\n")
# with open(path+"edge.csv", 'rb') as f_in, gzip.open(path+"edge.csv.gz", 'wb') as f_out:
#     shutil.copyfileobj(f_in, f_out)
# os.remove(path+"edge.csv")

# f = open(path+"num-edge-list.csv", "w")
# for i in num_edge_list:
#     f.write(str(int(i)) + "\n")
# with open(path+"num-edge-list.csv", 'rb') as f_in, gzip.open(path+"num-edge-list.csv.gz", 'wb') as f_out:
#     shutil.copyfileobj(f_in, f_out)
# os.remove(path+"num-edge-list.csv")


# f = open(path+"num-node-list.csv", "w")
# for i in num_node_list:
#     f.write(str(int(i)) + "\n")

# with open(path+"num-node-list.csv", 'rb') as f_in, gzip.open(path+"num-node-list.csv.gz", 'wb') as f_out:
#     shutil.copyfileobj(f_in, f_out)

# os.remove(path+"num-node-list.csv")

# f = open(path+"node-feat.csv", "w")

# for i in node_feat:
#     for j in i:
#         res = ""
#         for k in j:
#             res += str(k)+","
#         res = res[:-1]
#         res += "\n"
#         f.write(res)

# with open(path+"node-feat.csv", 'rb') as f_in, gzip.open(path+"node-feat.csv.gz", 'wb') as f_out:
#     shutil.copyfileobj(f_in, f_out)
# os.remove(path+"node-feat.csv")



# f = open(path+"graph-label.csv", "w")
# for i in range(1,133886):
#     f.write(str(i) + "\n")
# with open(path+"graph-label.csv", 'rb') as f_in, gzip.open(path+"graph-label.csv.gz", 'wb') as f_out:
#     shutil.copyfileobj(f_in, f_out)
# os.remove(path+"graph-label.csv")





ds = PygGraphPropPredDataset(name = "QM9", root = "Pipeline/data/QM9")
print(ds)