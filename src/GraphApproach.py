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
        tupleList = [] # store edges and weights
               
        # all pixels non-edge pixels 
        for h in range(0, height):
            for w in range(0 , width):
                
                if edges[h, w] >= 0:  # is edge
                    
                    vLabelOrg = edges[h, w]
                    vsIndex[vLabelOrg] = (h, w)
                    
                    for hM in range(h - maxRadius, h + maxRadius):
                        for wM in range(w - maxRadius, w + maxRadius):
                            vLabelDest = edges[hM, wM] 
                            try:  # avoid to access out of dimentions 
                                if vLabelDest >= 0 and vLabelOrg != vLabelDest:  # is edge and is not the same pixel
                                                         
                                    weight = self.laplaceWeight((h, w), (hM, wM), 1, minRadius, maxRadius)
                                    
                                    tupleList.append((vLabelOrg, vLabelDest, weight))                                    
                                    
                            except:
                                pass        
            
        
        graph = Graph.TupleList(edges=tupleList, directed=False, weights=True)
               
        graph.vs['imageIndex'] = vsIndex
                             
        graph.simplify(multiple=True, loops=True, combine_edges="max")
        graph.vs.select(_degree=0).delete()
        
        print('Creating graph [OK]')
                
        #membership = graph.community_fastgreedy(weights=graph.es["weight"]).as_clustering().membership
                       
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
    
    def laplaceWeight(self, vector1, vector2, b, minRadius, maxRadius):
                  
        # dist = np.linalg.norm(np.array(vector1)-np.array(vector2))
        dist = distance.euclidean(vector1, vector2)
               
        # return 1 if minRadius<= dist <= maxRadius else 0
              
        return (1 / (2 * b)) * np.exp(-1 * (dist / b))
    
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
    
