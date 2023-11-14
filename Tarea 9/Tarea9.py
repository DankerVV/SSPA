#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 12:14:18 2022

@author: carlossalcidoa
"""

import cv2
import numpy as np
from math import sqrt


#cone sta funcion reviso si un punto ya esta en alguna de mis listas
def isInTheList(elemento,arreglo):
    for i in arreglo:
        if np.array_equal(i,elemento):
            return True            
    return False  
    

while True:
    #para cargar el mapa
    mapa=cv2.imread('mapa3.png')
    #pasamos la imagen a escala de grises
    gray = cv2.cvtColor(mapa,cv2.COLOR_BGR2GRAY)
    #muestro la imagen en escala de grises
    cv2.imshow('mapa',gray)
    #obtengo un binarizacion en blaco todos lo pixeles cuyo valor en sea entre 254 y 255
    ret,th1 = cv2.threshold(gray,254,255,cv2.THRESH_BINARY)
    #hago un kernel de 11x11 de unos. Los Kernels se acostumbra hacerse de tamaño no par y cuadrados
    #para que se den una idea algo asi:
    """
    1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1
    """
    kernel = np.ones((11,11), np.uint8) 
    #aplico un filtro de dilatacion. Este filtro hace que los puntos los puntos blancos se expandan 
    #probocando que algunos puntitos negros desaparecan #le pueden hacer un cv.imshow para que vean el resultado
    #cv2.imshow('aaa', th1)
    
    th1 = cv2.dilate(th1,kernel,1)
    kernel = np.ones((11,11), np.uint8)
    
    
    #Despues aplico uno de erosion que hace lo opuesto al de dilatacion
    th1 = cv2.erode(th1,kernel,1)
    #aplico un flitro gausiando de 5x5  para suavisar los bordes 
    th1 = cv2.GaussianBlur(th1,(5,5),cv2.BORDER_DEFAULT) 
    #muestro como queda mi mapa
    cv2.imshow('thres',th1)
    #Aplico la deteccion de Esquinas de Harris. para mas informacion consulten https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
    dst = cv2.cornerHarris(th1,2,3,0.05)
    ret, dst = cv2.threshold(dst,0.04*dst.max(),255,0)
    dst = np.uint8(dst)
    ret,th2 = cv2.threshold(th1,235,255,cv2.THRESH_BINARY)
    th2 = cv2.dilate(th2,kernel,1)
    #aqui devuelvo la imagen binarizada a tres canales
    th2 = cv2.cvtColor(th2,cv2.COLOR_GRAY2BGR)
    
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst,30, cv2.CV_32S)
    centroids = np.delete(centroids, 0, 0)#PARA ELIMINAR LOS PUNTOS DENTRO DE LOS ESPACIOS EN NEGRO
    vertices=np.int0(centroids)
    
    aux1=vertices
    aux2=vertices
    verticesConectados=[]
    aristas=[]
    
    #aqui voy a buscar cuales son las esquinas que estan conectadas
    cont =0
    for h in range(len(aux1)):
        i=aux1[h]
        
        for k in range(h,len(aux2)):
            j=aux2[k]
            if not (i==j).all(): 
                cont+=1
                print(cont)
                
                print(i,end='')
                print(j)
                
                promx = (i[0] + j[0]) / 2
                #int(promx)
                
                promy = (i[1] + j[1]) /2
                #int(promy)
                
                #prom = promx, promy
                #print(cont)
                
                x1 = int((i[0] + promx)/2)
                y1 = int((i[1] + promy)/2)
                x2 = int((i[0] + x1)/2)
                y2 = int((i[1] + y1)/2)
                
                x3 = int((j[0] + x2)/2)
                y3 = int((j[1] + y2)/2)
                x4 = int((j[0] + x3)/2)
                y4 = int((j[1] + y3)/2)
                
                
                m = th2[int(promy)][int(promx)]
                m1 = th2[y1][x1]
                m2 = th2[y2][x2]
                m3 = th2[y3][x3]
                m4 = th2[y4][x4]
                
                #Verificar si son blancos
                if((m[0]>250 and m[1]>250 and m[2]>250) and (m1[0]>250 and 
                    m1[1]>250 and m1[2]>250) and (m2[0]>250 and m2[1]>250 and 
                    m3[2]>250) and (m3[0]>250 and m3[1]>250 and m3[2]>250) and 
                    (m4[0]>250 and m4[1]>250 and m4[2]>250)):
                                                  
                    distancia = int(sqrt((i[0] - j[0])**2 + (i[1] - j[1])**2))
                    aristas.append((i,j,distancia))
                    
                    if not isInTheList(i,verticesConectados):
                        verticesConectados.append(i)
                        
                        if not isInTheList(j,verticesConectados):
                            verticesConectados.append(j)
                
                            
                """
                print (str(th2[int(promx), int(promy)])) 
                
                if( (str(th2[int(promx * .25), int(promy * .25)])) == 0,0,0 and
                   (str(th2[int(promx * .5), int(promy * .5)])) == 0,0,0 and
                   (str(th2[int(promx * .75), int(promy * .75)])) == 0,0,0 and
                   (str(th2[int(promx * 1), int(promy * 1)])) == 0,0,0 and
                   (str(th2[int(promx * 1.25), int(promy * 1.25)])) == 0,0,0 and
                   (str(th2[int(promx * 1.75), int(promy * 1.75)])) == 0,0,0):
                    print("aaaa")
                    
                """
                
                
                #hacer una linea verde si entre i y j no hay color negro = True
                #aqui deberian sacar los puntos de intermedios y verificar si i y j estan conectados
                #si estan conectados calcular el costo (la distancia en pixeles entre ellos usan teorema de pitagoras papá) y agregarlos al grafo

    #aqui yo dibujo mis lineas de las aristas de color verde, el uno es el grueso de la linea
    #arista[0]   y arista[1]  tienen la forma de [fila, columna]
    
    for arista in aristas:
        cv2.line(th2, tuple(arista[0]), tuple(arista[1]), (0,255,0), 1)
        
        #aqui pinto los puntos de las esquinas que son circulos de de radio de 5 pixeles, el -1 indica que van rellenados los circulos
        #point tiene la forma [fila, columna]
    for point in vertices:
        cv2.circle(th2,(point[0], point[1]), 5, (255,0,0), -1)    
        cv2.waitKey(1)
    
    cv2.imshow('Cables verdes',th2)


    #Aplicar el algoritmo de Prim
    grafo = []
    listaVisitados = []
    listaVisitados.append(verticesConectados[0])
    while len(listaVisitados)!=len(verticesConectados):
        aristasNew=[]
        for a in listaVisitados:
            for arista in aristas:
                if np.array_equal(a,arista[0]) and (not isInTheList(arista[1],listaVisitados)):
                    aristasNew.append([a,arista[1],arista[2]])
                elif np.array_equal(a,arista[1]) and not isInTheList(arista[0],listaVisitados):
                    aristasNew.append([a,arista[0],arista[2]])
        edge = sorted(aristasNew,key=lambda element:element[2])[0]
        grafo.append(edge)
        
        listaVisitados.append(edge[1])
        
    for arista in grafo:
        cv2.line(mapa, tuple(arista[0]), tuple(arista[1]), (0,0,255), 1)
        
    for point in verticesConectados:
        cv2.circle(mapa,(point[0], point[1]), 5, (255,0,0), -1)
    cv2.imshow('ResultadoFinal',mapa)
    break

#aqui muestro como quedo de chingon el grafo
#cv2.imshow('points',th2)









