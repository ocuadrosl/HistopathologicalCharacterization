import cv2
import numpy as np
import matplotlib.pyplot as plt
import igraph
from igraph import *
from Utils import *


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
    
        # plt.imshow(edges, cmap='hot')
        # plt.show()
        
        return edges
    
    def createGraph(self, edges, maskSize=30):
        
        height, width = edges.shape
        
        maskSizeCenter = maskSize // 2
        
        graphEdges = []
        graphWeights = []
        vertexIds = []
        
        # edges
        
        for h in range(0, height):
            cH = h + maskSizeCenter  # mask corner
            for w in range(0, width):
                
                vertexId = (h * width) + w
                vertexIds.append((h, w))
                
                if edges[h, w] == 255:  # is edge?
                    
                    # print(vertexId, h,w)
                    # mask
                    cW = w + maskSizeCenter  # mask corner
                    d1 = cartesianToPolar(h, w, cH, cW)[0]
                    
                    for hM in range(h - maskSizeCenter, h + maskSizeCenter):
                        for wM in range(w - maskSizeCenter, w + maskSizeCenter):
                            
                            try:
                                if edges[hM, wM] == 255 and (hM, wM) != (h, w):  # is edge
                                    
                                    graphEdges.append((vertexId, (hM * width) + wM))
                                                                        
                                    d2 = cartesianToPolar(hM, wM, cH, cW)[0]
                                   
                                    weight = self.laplaceWeight(d1, d2, 5) 
                                    #print(weight)
                                    graphWeights.append(weight)
                                                              
                            except:
                                pass        
                
        graph = Graph(edges=graphEdges)
        
        graph.vs["ids"] = vertexIds
        graph.es["weights"] = graphWeights
                     
        graph.simplify(multiple=True, loops=True, combine_edges="max")
        graph.vs.select(_degree=0).delete()
        print('Creating graph [OK]')
                              
        # igraph.plot(graph, vertex_size=3, layout = graph.layout('kk'))
        
        membership = graph.community_fastgreedy(weights=graph.es["weights"]).as_clustering().membership
        print('Clutering [OK]')
                
        # membership = graph.community_multilevel(weights=graph.es["weights"]).membership 
        # membership = graph.community_label_propagation(weights=graph.es["weights"])   
                
        self.membershipToImage(graph, membership, height, width)
        
                            
    def membershipToImage(self, graph, membership, height, width):
        
        clusters = np.zeros((height, width), np.int32)  
        
        for i in range(0, len(membership)):
            h, w = graph.vs['ids'][i]
            clusters[h, w] = membership[i]
        
        plt.imshow(clusters, cmap='jet')
        plt.show()
    
    def laplaceWeight(self, vector1, vector2, b):
        
        return (1 / (2 * b)) * np.exp(-1 * (np.abs(vector1 - vector2) / b))
    
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
    
