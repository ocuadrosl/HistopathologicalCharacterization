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
from numpy.core.numeric import indices


class GraphApproach:
    
    '''
    @image a RGB image
    @return edges and angles
    '''

    def computeGradients(self, image):
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
          
        #plt.imshow(angles, cmap='hot')
        #plt.show()
        
        blur[np.where(binary == 255)] = 255  # aply threshold to blur
        
        edges = cv2.Canny(blur, 0, 255)
        
        edges = np.matrix(edges, dtype=np.int64)
    
        # plt.imshow(edges, cmap='hot')
        # plt.show()
        
        edgeAngle = np.zeros(edges.shape, np.float32) 
        
        edgeAngle[np.where(edges == 255)] = angles[np.where(edges == 255)]
        
        #plt.imshow(edgeAngle, cmap='hot')
        #plt.show()
        
        return edges, edgeAngle
    
    def createGraph(self, edges, angles, maxRadius=10):
                
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
        
        # print(vertexLabel)
        # plt.imshow(edges, cmap='hot')
        # plt.show() 
               
        tupleList = []  # store edges and weights
        
        B = np.zeros((height, width), np.int32) 
                       
        # all pixels 
        for h in range(0 + maxRadius, height - maxRadius):
            for w in range(0 + maxRadius , width - maxRadius):
                               
                if edges[h, w] >= 0:  # is an edge
                                                        
                    vLabelOrg = edges[h, w]
                    angOrg = angles[h, w]
                                       
                    mask = edges[h - maxRadius : h + maxRadius, w - maxRadius: w + maxRadius]
                    angMask = angles[h - maxRadius : h + maxRadius, w - maxRadius: w + maxRadius]
                    # plt.imshow(mask, cmap='jet')
                    # plt.show()
                                        
                    # b = self.computeB(mask)
                    b = self.parameterB(angMask)
                                       
                    for hD in range(0, maxRadius * 2):
                        for wD in range(0, maxRadius * 2):
                                                                       
                            # try:  # avoid to access out of dimentions 
                            vLabelDest = mask[hD, wD]
                            angTarg = angMask[hD, wD]
                            
                            if (vLabelDest >= 0) and (vLabelDest != vLabelOrg):  # is an edge
                                                                                                                                                 
                                # weight = self.laplaceWeight((maxRadius // 2, maxRadius // 2), (hD, wD), edges[h, w], edges[hD, wD], b)
                                # weight = self.angularWeight(angOrg, angTarg, b)
                                weight = self.weight(angOrg, angTarg, [maxRadius // 2, maxRadius // 2], [hD, wD], b)
                                # print (angOrg, angTarg, weight)
                                
                                tupleList.append([(h, w), vsIndex[vLabelDest], weight])
                                
                            # except:
                                # continue        
                
        graph = Graph.TupleList(edges=tupleList, directed=False, weights=True)
                             
        graph.simplify(multiple=True, loops=True, combine_edges="max")
        # graph.vs.select(_degree=0).delete()
        
        print('Creating graph [OK]')
                             
        membership = graph.community_fastgreedy(weights=graph.es["weight"]).as_clustering().membership
        
        # membership = graph.community_edge_betweenness(weights=graph.es["weight"], directed=False).as_clustering().membership
        # membership = graph.community_leading_eigenvector(weights=graph.es["weight"]).membership
               
        # plot(graph.community_fastgreedy(weights=graph.es["weight"]).as_clustering())
                             
        # membership = graph.community_multilevel(weights=graph.es["weight"]).membership 
        # membership = graph.community_label_propagation(weights=graph.es["weight"]).membership   
        
        print('Clutering [OK]')
                
        self.membershipToImage(graph, membership, height, width)
        
    def computeB(self, mask):
                       
        indicesH, indicesW = np.nonzero(mask >= 0)
                       
        medianH = np.median(indicesH)
        medianW = np.median(indicesW)
        
        # bH = np.sum(np.abs(np.array(indicesH) - medianH)) / len(indicesH)
        # bW = np.sum(np.abs(np.array(indicesW) - medianW)) / len(indicesW)
        
        b = np.sum(np.abs(np.array(indicesH) - medianH) + np.abs(np.array(indicesW) - medianW)) / len(indicesH)
        
        # print(b)
        
        return b
        # return np.max((bH, bW))
    
    def weight(self, angOrg, angTarg, pOrg, pTarg, b):
        
        pOrg.append(angOrg)
        pTarg.append(angTarg)
        
        dist = distance.sqeuclidean(pOrg, pTarg)
        
        return (1 / (2 * b)) * np.exp(-1 * (dist / b))
    
    def angularWeight(self, angOrg, angTarg, b=1):
        """
        computes weight using the gradient angle
        Args:
        angOrg: angle of Origing 
        angTarg: angle of Target
        """
              
        return (1 / (2 * b)) * np.exp(-1 * (np.abs(angOrg - angTarg) / b))
        
    def parameterB(self, mask):
         
        median = np.median(mask)
        
        b = np.sum(np.abs(mask - median)) / mask.size 
        
        return  b if b > 0 else 0.00001
                            
    def membershipToImage(self, graph, membership, height, width):
        
        clusters = np.zeros((height, width), np.int32) 
                        
        for i in range(0, len(membership)):
            h, w = graph.vs['name'][i]
            clusters[h, w] = membership[i]
        
        plt.imshow(clusters, cmap='jet')
        plt.show()
    
    def laplaceWeight(self, pOrg, pDest, angleOrg , angleDest, b):
                           
        # fOrg = pOrg[0], pOrg[1]  # , angleOrg#, distOrgToCent
        # fDest = pDest[0], pDest[1]  # , angleDest#, distDestToCent   
        
        # print(distance.sqeuclidean(fOrg,fDest))
        
        dist = distance.euclidean(pOrg, pDest)
        
        # print(pOrg,pDest, b ,(1 / (2 * b)) * np.exp(-1 * (2*dist / b)))
                            
        return (1 / (2 * b)) * np.exp(-1 * (dist / b))
    
    def process(self, rgbImage):
        edges, angles = self.computeGradients(rgbImage)
        self.createGraph(edges, angles)  
        print('OK')  
    
