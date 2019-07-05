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
    @return edges and angles
    '''

    def getGradients(self, image):
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
              
        blur = cv2.GaussianBlur(imageGray, (3, 3), 0)
              
        binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # gx, gy  = np.gradient(blur)
        
        gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0) 
        gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
               
        angles = np.arctan2(gy, gx) * (180 / np.pi)
        
        # h,w = imageGray.shape
        # x, y = np.mgrid[0:h, 0:w]
        # plt.quiver(x, y, gx, gy, units='x', pivot='tip', width=0.09,   scale=1 / 0.01)
          
        # plt.imshow(angles, cmap='hot')
        # plt.show()
        
        blur[np.where(binary == 255)] = 255  # aply threshold to blur
        
        edges = cv2.Canny(blur, 0, 255)
        
        edges = np.matrix(edges, dtype=np.int64)
    
        #plt.imshow(edges, cmap='hot')
        #plt.show()
        
        return edges, angles
    
    def createGraph(self, edges, angles , minRadius=1, maxRadius=5):
                
        height, width = edges.shape
        
        vsIndex = []
        vertexLabel = 0
        
        for h in range(0, height):
            for w in range(0, width):
                if edges[h, w] == 255:
                    edges[h, w] = vertexLabel
                    vsIndex.append((h, w)) 
                    vertexLabel += 1
                    
                else:
                    edges[h, w] = -1
        
        
        #print(vertexLabel)
        #plt.imshow(edges, cmap='hot')
        #plt.show() 
          
        
        # verticesNo = vertexLabel             
        
        # vsIndex = [None] * verticesNo  # to store the correesponding image index
        tupleList = []  # store edges and weights
               
        # all pixels 
        for h in range(0, height):
            for w in range(0 , width):
                               
                if edges[h, w] >= 0:  # is an edge
                                                        
                    vLabelOrg = edges[h, w]
                    # vsIndex[vLabelOrg] = (h, w)
                    
                    for hD in range(h - maxRadius, h + maxRadius):
                        for wD in range(w - maxRadius, w + maxRadius):
                                                   
                            try:  # avoid to access out of dimentions 
                                
                                vLabelDest = edges[hD, wD]
                                if vLabelDest >= 0 and vLabelDest != vLabelOrg:  # is an edge
                                                                                                                                                                 
                                    weight = self.laplaceWeight((h, w), (hD, wD), angles[h, w], angles[hD, wD], 1)
                                    # print(weight)
                                    tupleList.append([vLabelOrg, vLabelDest, weight])                                    
                                    
                            except:
                                pass        
        
        graph = Graph.TupleList(edges=tupleList, directed=False, weights=True)
               
        graph.vs['imageIndex'] = vsIndex
                             
        graph.simplify(multiple=True, loops=True, combine_edges="max")
        graph.vs.select(_degree=0).delete()
        
        print('Creating graph [OK]')
                
                
        membership = graph.community_fastgreedy(weights=graph.es["weight"]).as_clustering().membership
        
                             
        # membership = graph.community_multilevel(weights=graph.es["weight"]).membership 
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
    
    def laplaceWeight(self, pOrg, pDest, angleOrg , angleDest, b):
                           
        fOrg = pOrg[0], pOrg[1]  # , angleOrg#, distOrgToCent
        fDest = pDest[0], pDest[1]  # , angleDest#, distDestToCent   
        
        # print(distance.sqeuclidean(fOrg,fDest))
        
        diff = distance.euclidean(pOrg, pDest)
       
        # print(diff, (1 / (2 * b)) * np.exp(-1 * ( diff / b)))
        
        return np.exp(-1 * diff)
                            
        #return (1 / (2 * b)) * np.exp(-1 * (diff / b))
    
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
        edges, angles = self.getGradients(rgbImage)
        self.createGraph(edges, angles)  
        print('OK')  
    
