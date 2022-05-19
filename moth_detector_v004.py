### NECTRAS 
### Autor: W. Daniel Sequeiros
### http://www.nectras.com
### daniel@nectras.com
### Te: +5493515920003

###############
# VERSIONES
################
# V0003. 
#    Se ajusta la imagen a tamanio correcto.
#    Ratio of window 
# V0004.
#    Salida Array detect_identy


import cv2 #Importamos OPENCV
import numpy as np #multi-dimensional arrays

##################
###FUNCIONES
##################


def find_if_close(cnt1,cnt2): #Encuentra contornos proximos 
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 5 : #Distancia entre bordes al unirlos
                return True
            elif i==row1-1 and j==row2-1:
                return False

 
def detect_identy(x,y,w,h,a): #Identificamos que es
    area=w*h
    ratioa=area/a; ## VEMOS CUANTO OCUPA DEL CUADRADO QUE LO RODEA
    ratio=w/h;
    return_array=[];
    return_array["x"]=x;
    return_array["y"]=x;
    return_array["w"]=w;
    return_array["h"]=h;
    return_array["a"]=a;
    return_array["ratio"]=ratio
    return_array["ratioa"]=ratioa
    #return str(ratioa);
    if 600<a<1600:
        if a>1200:
            if 0.8<ratio>1.2:
                return_array["label"]="Moth"
                return return_array
         
        if ratioa>1.5: # el radio de area no corresponde a una mozca 
            return_array["label"]="Unknow"
            return return_array
        return_array["label"]="Fly"
        return return_array
    if 1600<a<5000:# TAMANIO CORRECTO
        if ratioa<2: #RADIO CORRECTO
            return_array["label"]="Moth"
            return return_array
    if 9000<a<15000:# area de feromona 10236.5
        if a>13000:
            return_array["label"]="Moth"
            return return_array # Encontramos una feromona con bicho al lado.
        return_array["label"]="Phero"
        return return_array
    if 20000<a:
        if ratio>0.5:
            return_array["label"]="Unknow"
            return return_array
        return_array["label"]="Window"
        return return_array
    return_array["label"]="Unknow"
    return return_array

#####################
######EMPEZAMOS
#####################

file='6.jpg' #Ruta a imagen
#file='img/captura.jpg' #Ruta a imagen
img_org = cv2.imread("img/"+file) #Cargamos Imagen

####################
####REDIMENSIONAMOS EN CASO DE IMAGENES CHICAS
###################
height =img_org.shape[0];
if(height<1200):
    scale=1200/height
    #print(scale)
    width = int(img_org.shape[1] * scale )
    height = int(img_org.shape[0] * scale)
    dim = (width, height)
    # resize image
    img_org = cv2.resize(img_org, dim, interpolation = cv2.INTER_AREA)
    #cv2.imshow('Resize',img_resize)   # Mostramos IMG original


#############
##DILATACION
#############
dilatation_size=7
dilatation_type = cv2.MORPH_ELLIPSE #cv.MORPH_RECT  / cv.MORPH_CROSS /cv.MORPH_ELLIPSE  
element = cv2.getStructuringElement(dilatation_type, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
img_dst = cv2.dilate(img_org, element) # Dilatacin  .erode() .dilate()
#cv2.imshow('Dilatacion',img_dst)   # Mostramos IMG original
############
##EROCiON
###########3
ero_size=8
ero_type = cv2.MORPH_RECT #cv.MORPH_RECT  / cv.MORPH_CROSS /cv.MORPH_ELLIPSE  
element = cv2.getStructuringElement(ero_type, (2*ero_size + 1, 2*ero_size+1), (ero_size, ero_size))
img_ero = cv2.erode(img_dst, element)
#cv2.imshow('erocion',img_ero)   # Mostramos IMG original
###########
##GAUSS & Backwithe
###########
imggb = cv2.cvtColor(img_ero, cv2.COLOR_BGR2GRAY) #transformamos a escala de grises
imggb = cv2.GaussianBlur(imggb, (5, 5), 0) #transformamos a escala de grises
#cv2.imshow('GAUSBLUR',imggb)   # Mostramos IMG BLANCO Y NEGRO y GAUSEANO

###########
##DETECTAMOS BORDES
##########
#Detectamos los bordes con Canny
img_bordes=imggb
img_show=img_org.copy()
img_bordes= cv2.Canny(img_bordes, 50,140) #detector de bordes Canny
# Buscamos los contornos
(contornos,_) = cv2.findContours(img_bordes.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
#print("He encontrado {} objetos".format(len(contornos)))

cv2.drawContours(img_show,contornos,-1,(0,0,255), 2)#Dibuja los Contornos
#cv2.imshow('BORDES',img_show)   # Mostramos IMG BLANCO Y NEGRO y GAUSEANO


####################
##JUNTAMOS CONTORNOS
#################### 
img_contornos_join=img_org.copy()
contours=contornos
LENGTH = len(contours)
status = np.zeros((LENGTH,1))


for i,cnt1 in enumerate(contours):
    x = i    
    if i != LENGTH-1:
        for j,cnt2 in enumerate(contours[i+1:]):
            x = x+1
            dist = find_if_close(cnt1,cnt2)
            if dist == True:
                val = min(status[i],status[x])
                status[x] = status[i] = val
            else:
                if status[x]==status[i]:
                    status[x] = i+1

unified = []
maximum = int(status.max())+1
for i in range(maximum):
    pos = np.where(status==i)[0]
    if pos.size != 0:
        cont = np.vstack(contours[i] for i in pos)
        hull = cv2.convexHull(cont)
        unified.append(hull)

cv2.drawContours(img_contornos_join,unified,-1,(0,255,0),2)


##AGREGAMOS CONVEX HULL Y TEXTO
font = cv2.FONT_HERSHEY_SIMPLEX # font  
fontScale = 0.5# fontScale 
color = (0, 0, 255)# Blue color in BGR  
thickness = 1# Line thickness of 2 px 
   
 
for con in unified:
    area=area = cv2.contourArea(con); ## CONTORNO SIN RECTANGULO
    x,y,w,h = cv2.boundingRect(con) ## AGREGAMOS LOS RECTANGULOS
    array_detec=detect_identy(x,y,w,h,area)
    label=str(array_detec["label"]);
    cv2.rectangle(img_contornos_join,(x,y),(x+w,y+h),(0,0,255),2)
    org =(x, y+h+15)    #org      
    cv2.putText(img_contornos_join, label, org, font,fontScale, color, thickness, cv2.LINE_AA)



cv2.imshow('Moth Detector',img_contornos_join)   # Mostramos los contornos juntos
print("He encontrado {} objetos".format(len(unified)))
cv2.imwrite( "result/"+file, img_contornos_join );


#cv2.imshow('Orginal',img_org)   # Mostramos IMG original
cv2.waitKey(0)  #Programa en suspenso.
