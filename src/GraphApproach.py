import cv2
import numpy as np
import matplotlib.pyplot as plt
import igraph
from igraph import *
from Utils import *
from matplotlib.font_manager import weight_dict
from igraph.drawing import edge
from dis import dis
from scipy.spatial import distance
from numpy.lib.function_base import angle


class GraphApproach:
    
    '''
    @image a RGB image
    '''

    def getEdges(self, image):
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imageGray, (3, 3), 0)
        
        binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        blur[np.where(binary == 255)] = 255  # aply threshold to blur
        edges = cv2.Canny(blur, 0, 255)
        
        edges = np.matrix(edges, np.int32)
    
        # plt.imshow(edges, cmap='hot')
        # plt.show()
        
        return edges
    
    def createGraph(self, edges, minRadius=5, maxRadius=10):
        
        height, width = edges.shape
        
        vertexLabel = 0
        for h in range(0, height):
            for w in range(0, width):
                if edges[h, w] == 255:
                    edges[h, w] = vertexLabel
                    vertexLabel += 1
                else:
                    edges[h, w] = -1
        
        # plt.imshow(edges, cmap='hot')
        # plt.show() 
        # print(VertexLabel)  
        
        verticesNo = vertexLabel             
        
        vsIndex = [None] * verticesNo  # to store the correesponding image index
        tupleList = []  # store edges and weights
               
        # all pixels 
        for h in range(0, height):
            for w in range(0 , width):
                                
                if edges[h, w] == -1:  # is not an edge
                                                          
                    vLabelOrg = None
                    for hM in range(h - maxRadius, h + maxRadius):
                        for wM in range(w - maxRadius, w + maxRadius):
                                                   
                            try:  # avoid to access out of dimentions 
                                if edges[hM, wM] >= 0:  # is an edge
                                    vsIndex[edges[hM, wM]] = (hM, wM)
                                     
                                    if vLabelOrg is None:
                                        vLabelOrg = edges[hM, wM]
                                        hOrg, wOrg = hM, wM
                                        continue   
                                    
                                    vLabelDest = edges[hM, wM];
                                                                                             
                                    weight = self.laplaceWeight((hOrg, wOrg), (hM, wM), (h, w), 1, minRadius, maxRadius)
                                    # print(weight)
                                    
                                    tupleList.append((vLabelOrg, vLabelDest, weight))                                    
                                    
                            except:
                                pass        
        
        graph = Graph.TupleList(edges=tupleList, directed=False, weights=True)
               
        graph.vs['imageIndex'] = vsIndex
                             
        graph.simplify(multiple=True, loops=True, combine_edges="max")
        graph.vs.select(_degree=0).delete()
        
        print('Creating graph [OK]')
                
        # membership = graph.community_fastgreedy(weights=graph.es["weight"]).as_clustering().membership
                       
        membership = graph.community_multilevel(weights=graph.es["weight"]).membership 
        # membership = graph.community_label_propagation(weights=graph.es["weights"])   
        
        print('Clutering [OK]')
                
        self.membershipToImage(graph, membership, height, width)
                            
    def membershipToImage(self, graph, membership, height, width):
        
        clusters = np.zeros((height, width), np.int32)  
        
        for i in range(0, len(membership)):
            h, w = graph.vs['imageIndex'][i]
            clusters[h, w] = membership[i]
        
        plt.imshow(clusters, cmap='jet')
        plt.show()
    
    def laplaceWeight(self, p1, p2, pC, b, minRadius, maxRadius):
                  
                
        dist1ToCent = distance.sqeuclidean(p1, pC)
        dist2ToCent = distance.sqeuclidean(p2, pC)
        dist1to2 = distance.sqeuclidean(p1, p2)
        
        diffToCent = np.abs(dist1ToCent - dist2ToCent) 
        
        relation = diffToCent/dist1to2   if dist1to2 > 0 else diffToCent / 0.001
        
        # angle = np.degrees(np.arctan2(p2[1] - pC[1], p2[0] - pC[0]) - np.arctan2(p1[1] - pC[1], p1[0] - pC[0]))                
               
        # relation = distDiff / angle if angle > 0 else distDiff / 0.0001
                      
        return (1 / (2 * b)) * np.exp(-1 * (relation / b))
    
    def cosine(self, vector1, vector2):
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        cosine = -1.0

        if (norm1 != 0.0) and (norm2 != 0.0):
            cosine = np.dot(vector1, vector2) / (norm1 * norm2)
            if cosine > 1.0:
                cosine = 1.0
            elif cosine < -1.0:
                cosine = -1.0
        elif (norm1 == 0.0) and (norm2 == 0.0):
            cosine = 1.0

        return cosine
    
    def jaccard(self, vector1, vector2):
        
        assert len(vector1) == len(vector2), "vectors must be of equal sizes"
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        if norm1 != 0.0 and norm2 != 0.0:
            # dot = np.dot(norm1, norm2)
            # coefficient = (dot / ((norm1 * norm1) + (norm2 * norm2) - dot))
            # return 1.0 / (1.0 + coefficient)
            dot = np.dot(vector1, vector2)
            coefficient = (dot / ((norm1 * norm1) + (norm2 * norm2) - dot))
            return (coefficient)
        elif norm1 == 0.0 and norm2 == 0.0:
            return 1.0
        else:
            return 0.0
    
    def process(self, rgbImage):
        edges = self.getEdges(rgbImage)
        self.createGraph(edges)  
        print('OK')  
    
