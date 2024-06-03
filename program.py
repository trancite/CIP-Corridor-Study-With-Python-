import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

contador_direcciones = 0
contador_esquives = 0
MODO = 0

class Pasillo:
    def __init__(self, H, L, tipos_personas, salidas): #El pasillo se inicializa escogiendo cuantos tipos de personas hay y sus salidas
        self.H = H
        self.L = L
        #mapa de calor
        self.Mapa_calor = np.zeros((H, L))

        #camp vectorial (matriu 3d)
        self.Camp_vect = np.zeros((H, L, 2))
        self.Camp_vect_unitari = np.zeros((H, L, 2))

        self.M = np.zeros((H, L))
        self.E = {} #esquivan
        self.C = {} #capces (pueden avanzar en su sentido)
        self.F = {} #finalizados
        self.I = {} #incapaces
        self.P = {} #pared delante
        self.tipos_personas = tipos_personas
        self.personas = {tipo: set() for tipo in self.tipos_personas} #tipo:personas de ese tipo
        self.salidas = salidas #tipo de persona:salidas
        self.out = {tipo: 0 for tipo in self.tipos_personas}

    def analisis(self):
        self.E, self.C, self.F, self.I, self.P = {}, {}, {}, {}, {}
        for grupos in self.personas.values():
            for i in grupos:
                fila, columna, _ = i
                v = self.M[fila, columna]

                if [fila, columna] in self.salidas.get(v, []):
                    self.F[i] = ["adeu"]
                    continue

                D = self.direcciones(i)
                d = D[0]
                a, b, _ = self.mirar_sig(d, i)
                y, x, _ = self.mirar_sig(d, [a, b, v])

                fuera_de_limites = y > self.H - 1 or y < 0 or x > self.L - 1 or x < 0
                condicion_a_b = self.M[a, b] != 0 and self.M[a, b] != v and self.M[a, b] != -1 #true si la persona de delante es de un tipo diferente
                condicion_y_x = self.M[y, x] != 0 and self.M[y, x] != v and self.M[y, x] != -1 if not fuera_de_limites else False #true si la persona de delante de delante es de un tipo diferente

                if fuera_de_limites:
                    if self.M[a, b] != 0:
                        if self.M[a, b] == v:
                            self.I[i] = D
                        else:
                            self.E[i] = D
                    if self.M[a, b] == 0:
                        self.C[i] = D
                else:
                    if (self.M[y, x] == v and self.M[a, b] == 0):
                        self.C[i] = D
                    elif condicion_a_b or (condicion_y_x and self.M[a, b] == 0):
                        self.E[i] = D
                    elif self.M[a, b] == v or self.M[y, x] == v:
                        self.I[i] = D
                    elif self.M[a, b] == -1:
                        self.P[i] = D
                    else:
                        self.C[i] = D

    def mirar_sig(self, j, k):
        if (j == 0):
            return (k[0] - 1, k[1], 0)
        elif (j == 1):
            return (k[0] + 1, k[1], 1)
        elif (j == 2):
            return (k[0], k[1] + 1, 2)
        elif (j == 3):
            return (k[0], k[1] - 1, 3)

    def llenador2(self, cantidad_gente): #cantidad_gente es un diccionario con  {tipo persona : cantidad de personas de ese tipo}
        for k in cantidad_gente:
            j = 0
            while j < cantidad_gente[k]:
                y, x = random.randint(0, self.H - 1), random.randint(0, self.L - 1)
                if self.M[y, x] == 0:
                    self.M[y, x] = int(k)
                    self.personas[k].add((y, x, -1))
                    j += 1


    def flujo(self, flujo, tipo):
        dic = flujo[tipo]
        for entrada, cantidad in dic.items():
            E = self.salidas[entrada].copy()
            k = 0
            j = 0
            for i in E[:]:
                if self.M[i[0], i[1]] != 0:
                    E.remove(i)
            if len(E) == 0:
                return 0
            
            else:
                j = 0
                while k<cantidad:
                    if len(E) == 0:
                        return 0
                    else:
                        y, x = random.choice(list(E))
                        self.M[y, x] = tipo
                        self.personas[tipo].add((y, x, -1))
                        E.remove([y, x])
                        k += 1

    def entorno(self, X): #analiza la matriz del laberinto
        self.M = X
        self.H, self.L = X.shape
        self.Mapa_calor = np.zeros((self.H, self.L))
        for i in range(self.H):    
            for j in range(self.L):
                if (self.M[i, j] != 0 and self.M[i, j] != -1):
                    self.personas[self.M[i, j]].add((i, j, -1))

    def casilla_cercana(self, k): #k es la tuple (a,b,c) en la que (a,b) = (y,x) y c representa de donde viene
        S = self.salidas[self.M[k[0],k[1]]] #S accede a las posibles salidas del individuo de coordenadas [a,b]
        minimo = float("inf")
        ll = None
        for salida in S:
            d = abs(k[0] - salida[0]) + abs(k[1] - salida[1])
            if d < minimo:
                minimo = d
                ll = salida
        return ll
    
    def detector(self, k, d):
        v = self.M[k[0],k[1]]
        cont = 0
        Ig = [0]
        # norte
        if (d == 0):
            for j in range(-5, 6):
                for i in range(1,5):
                    if (0 <= k[0]-i < self.H and 0 <= k[1]+j < self.L):
                        pr = self.M[k[0]-i,k[1]+j]
                        if (pr == v):
                            cont += 1
                        elif (pr not in Ig):
                            cont -= 1
        elif (d == 1):
            for j in range(-5, 6):
                for i in range(1,5):
                    if (0 <= k[0]+i < self.H and 0 <= k[1]+j < self.L):
                        pr = self.M[k[0]+i,k[1]+j] 
                        if (pr == v):
                            cont += 1
                        elif (pr not in Ig):
                            cont -= 1
        elif (d == 2):
            for j in range(-5, 6):
                for i in range(1, 5):    
                    if (0 <= k[0]+j < self.H and 0 <= k[1]+i < self.L):
                        pr = self.M[k[0]+j,k[1]+i]
                        if (pr == v):
                            cont += 1
                        elif (pr not in Ig):
                            cont -= 1
        elif (d == 3):
            for j in range(-5, 6):
                for i in range(1, 5):
                    if (0 <= k[0]+j < self.H and 0 <= k[1]-i < self.L):    
                        pr = self.M[k[0]+j,k[1]-i]
                        if (pr == v):
                            cont += 1
                        elif (pr not in Ig):
                            cont -= 1
        return cont

    def esquive_m(self, k, D_2):
        A = self.detector(k, D_2[0])
        B = self.detector(k, D_2[1])
        if (A>B):
            return [D_2[0],D_2[1]]
        elif (A<B):
            return [D_2[1],D_2[0]]
        else:
            return D_2
    
    def direcciones(self, k): # 0-norte, 1-sur, 2-este, 3-oeste
        global contador_direcciones
        contador_direcciones += 1
        ll = self.casilla_cercana(k)
        dx = ll[1] - k[1]
        dy = ll[0] - k[0]
        D = []
        D2 = []  
        if (MODO == 0 or MODO == 2):
            def v():
                if dy < 0:
                    D.append(0)
                    D2.append(1)
                elif dy > 0:
                    D.append(1)
                    D2.append(0)
                else:
                    al1 = random.randint(0, 1)
                    D.append(al1)
                    D2.append(1 - al1)
        
        elif (MODO == 1):
            def v():
                if (self.M[k[0],k[1]] == 1):
                    D.append(0)
                    D2.append(1)
                else:
                    D.append(1)
                    D2.append(0)

        def h():
            if dx>0:
                D.append(2)
                D2.append(3)

            elif dx<0:
                D.append(3)
                D2.append(2) 
            else:
                al2 = random.randint(0, 1)
                D.append(2 + al2)
                D2.append(3 - al2)

        acciones = [h, v]
        al3 = random.randint(0, 1)
        if abs(dy) > abs(dx) or (abs(dy) == abs(dx) and al3 == 1):
            acciones.reverse()

        for accion in acciones:
            accion()
        D = D + D2[::-1]
        if (k[0] == 0 and 0 in D):
            D.remove(0)
        elif (k[0] == self.H - 1 and 1 in D):
            D.remove(1)
        if (k[1] == 0 and 3 in D):
            D.remove(3)
        elif(k[1] == self.L - 1 and 2 in D):
            D.remove(2)
        return D    
    
    def avance_s(self, k, d):
        y, x = k[0], k[1]
        if (d == 0):#NORTE
            self.M[y - 1, x] = self.M[y, x]
            self.Mapa_calor[y - 1, x] += 1
            self.Camp_vect[y, x, 0] -= 1
        
        elif(d == 1):#SUR
            self.M[y + 1, x] = self.M[y, x]
            self.Mapa_calor[y + 1, x] += 1
            self.Camp_vect[y, x, 0] += 1 
        
        elif(d == 2):#ESTE
            self.M[y, x + 1] = self.M[y, x]
            self.Mapa_calor[y, x + 1] += 1
            self.Camp_vect[y, x, 1] += 1
        
        elif(d == 3):#OESTE
            self.M[y, x - 1] = self.M[y, x]
            self.Mapa_calor[y, x - 1] += 1
            self.Camp_vect[y, x, 1] -= 1
        
    def tiempo(self, t, flujo, modulo1, modulo2):
        estados = [] #para la animacion, almacena cada estado de la matriz

        for k in range(0, t):
            self.analisis()
            B = set()
            self.borrar_pantalla()   
            print(self.M)
            time.sleep(0.25)
            self.tiempo_F()
            self.tiempo_I()
            B = self.tiempo_P(B)
            B = self.tiempo_C(B)
            B = self.tiempo_E(B)
            if k % 2 == 0:
                self.flujo(flujo, 1)
                self.flujo(flujo, 2)
            for b in B:
                self.M[b[0], b[1]] = 0

            #ANIMACION-----------------------------
            #nos restringimos al caso maximo de 4 tipos de persona, añadir más es cuestión de añadir un par de filas de código más
            estados.append(self.M.copy())            
            def update(k):
                matriz = estados[k]
                ax.clear()
                ax.set_xlim(-1, columnas)
                ax.set_ylim(-1, filas)
                ax.set_aspect('equal') #relacion de aspecto 1:1
                for (j, k), val in np.ndenumerate(matriz):
                    if val == 1:
                        circle = plt.Circle((k, j), 0.3, color='darkturquoise')
                        ax.add_artist(circle)
                    elif val == 2:
                        circle = plt.Circle((k, j), 0.3, color='r')
                        ax.add_artist(circle)
                    elif val == 3:
                        circle = plt.Circle((k, j), 0.3, color='orange')
                        ax.add_artist(circle)
                    elif val == 4:
                        circle = plt.Circle((k, j), 0.3, color='hotpink')
                        ax.add_artist(circle)
                    elif val == -1:
                        square = plt.Rectangle((k-0.5, j-0.5), 0.9, 0.9, color='black')
                        ax.add_artist(square)
                plt.gca().invert_yaxis()

        fig, ax = plt.subplots()
        ani = animation.FuncAnimation(fig, update, frames=t, repeat=True)
        #guardar en gif:
        ani.save("animación.gif", writer='imagemagick', fps=6)
        plt.close(fig)
        print(self.out)

        #CAMP VECTORIAL---------------------------
        for i in range(0, self.H):
            for j in range(0, self.L):
                for l in range(0, 2):
                    if (self.Camp_vect[i, j, 0], self.Camp_vect[i, j, 1]) != (0, 0):
                        self.Camp_vect_unitari[i, j, l] = self.Camp_vect[i, j, l]/math.sqrt(self.Camp_vect[i, j, 0]**2 + self.Camp_vect[i, j, 1]**2)
                    else:
                        self.Camp_vect_unitari[i, j, l] = 0

        plt.figure()
        for i in range(0, self.L):
            for j in range(0, self.H):
                if(self.M[j, i] == -1):
                    rect = plt.Rectangle((i-0.5, j-0.5), 0.9, 0.9, color='black')
                    plt.gca().add_patch(rect)
                elif((self.Camp_vect[j, i, 0], self.Camp_vect[j, i, 1]) == (0, 0)):
                    continue
                else:
                    plt.quiver(i, j, self.Camp_vect_unitari[j, i, 1], self.Camp_vect_unitari[j, i, 0], angles = 'xy', scale_units = 'xy', scale = 1)    
        plt.xlim(-1, self.L)
        plt.ylim(-1, self.H)
        plt.gca().invert_yaxis() #para invertir el eje y, que en matrices y en graficas es al reves
        plt.gca().set_aspect('equal')
        #plt.xlabel('Eje X')
        #plt.ylabel('Eje Y')
        #plt.grid()
        #Afegim linees per marcar les parets
        xlin = [0, self.L-1]
        ylin1 = [-0.5, -0.5]
        ylin2 = [self.H-0.5, self.H-0.5]
        plt.plot(xlin, ylin1, color = 'black')
        plt.plot(xlin, ylin2, color = 'black')
        plt.savefig('camp_vectorial.png')
        #plt.show()

        #MAPA DE CALOR-----------------------------------
        fig, ax = plt.subplots()
        ax.matshow(self.Mapa_calor, cmap=plt.cm.Reds)
        fig.savefig('heatmap.png')

        return self.M
        



        return self.M
    
    
    def tiempo_F(self):
        for k, D in self.F.copy().items():
                v = self.M[k[0], k[1]]
                self.M[k[0], k[1]] = 0
                self.out[v] += 1
                self.personas[v].remove(k)

    def tiempo_I(self):
        for k, D in self.I.copy().items():
                v = self.M[k[0], k[1]]

    def tiempo_P(self, B):
        items_P = random.sample(sorted(self.P.items()), len(self.P))
        for k, D in items_P:
                v = self.M[k[0], k[1]]
                condicion_cumplida = False  

            
                for d in D[1:-1]:
                    a, b = self.mirar_sig(d, k)[:-1]
                    if self.no_rep(k[2], d) and self.M[a, b] == 0:
                        self.personas[v].remove(k)
                        self.personas[v].add(self.mirar_sig(d, k))
                        self.avance_s(k, d)
                        B.add(k)
                        condicion_cumplida = True
                        break  

                if not condicion_cumplida:
                    for d in D[1:-1]:
                        a, b = self.mirar_sig(d, k)[:-1]
                        if self.M[a, b] == 0:
                            self.personas[v].remove(k)
                            self.personas[v].add(self.mirar_sig(d, k))
                            self.avance_s(k, d)
                            B.add(k)
                            break
        return B

    def tiempo_C(self, B):
        items_C = random.sample(sorted(self.C.items()), len(self.C))
        for k, D in items_C:
            v = self.M[k[0], k[1]]
            df = D[0]
            n, m = self.mirar_sig(df, k)[:-1]
            if self.M[n, m] == 0:
                self.personas[v].remove(k)
                self.personas[v].add(self.mirar_sig(df, k))
                self.avance_s(k, df)
                B.add(k)
        return B

    def tiempo_E(self, B):
        items_E = random.sample(sorted(self.E.items()), len(self.E))
        for k, D in items_E:
            v = self.M[k[0], k[1]]
            if (len(D) == 4 and MODO == 2):
                D_2 = self.esquive_m(k, D[1:-1])
                for d in D_2:
                    a, b = self.mirar_sig(d, k)[:-1]
                    if (self.M[a, b] == 0):
                        self.avance_s(k, d)
                        self.personas[v].remove(k)
                        self.personas[v].add(self.mirar_sig(d, k))
                        B.add(k)
                        break
            elif (MODO == 0 or MODO == 1): 
                for d in D[1:-1]:
                    a, b = self.mirar_sig(d, k)[:-1]
                    if (self.M[a, b] == 0):
                        self.avance_s(k, d)
                        self.personas[v].remove(k)
                        self.personas[v].add(self.mirar_sig(d, k))
                        B.add(k)
                        break
        return B
 
    def no_rep(self, c, d):
        return not ((c == 0 and d == 1) or (c == 1 and d == 0) or (c == 2 and d == 3) or (c == 3 and d == 2))
            
    def borrar_pantalla(self):        print("\033[H\033[J", end="")

    def imprimir(self):
        print(self.M)
        print(self.personas)
        print(self.M[1, 2])
        print(self.salidas)

def desviacion_estandar(lista):
    if len(lista) == 0:
        return None
    
    media = sum(lista) / len(lista)
    varianza = sum((x - media) ** 2 for x in lista) / len(lista)
    desviacion = math.sqrt(varianza)
    
    return desviacion

def estadistica(Est):
    # Ejemplo de lista de valores de altura y sus coordenadas en x
    x = []
    for k in range(0,len(Est)):
        x.append(k)

    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(x, Est, marker='o', linestyle='None', color='b', label='Altura')
    plt.xlabel('Iteración')
    plt.ylabel('Personas que han abandonado el pasillo')
    plt.title('Gráfica de la Estadística de Altura')
    plt.legend()
    plt.grid(True)
    plt.show()

def contador(k):
    M = 0
    Est = []
    for _ in range(k):
        X= np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        filas = 9
        columnas = 17
        Flujo = {1: {2: 1}, 2: {1: 1}} #(tipo, entrada, cantidad)
        T = [1, 2]
        S1 = []
        S2 = []
        for necronomicon in range(0, filas-4): #salidas
            S1.append([necronomicon, 0])
        for necronomicon in range(0, filas): #salidas
            S1.append([necronomicon, 0])
            S2.append([necronomicon, columnas - 1])
            
        
        s = {1: S1, 2: S2}
        cantidad_gente = {1: 80, 2: 50}
        passadis = Pasillo(20, 50, T, s)
        passadis.entorno(X)
        passadis.tiempo(1000, Flujo)
        for i in passadis.tipos_personas:
            M += passadis.out[i]
            #Est.append(passadis.out[i])
        Est.append(passadis.out[1]+passadis.out[2])
    print(M/(k), k)
    print(desviacion_estandar(Est))
    return(Est)

if __name__ == "__main__":
    H = [1, 2]
MODO = int(input("¿Qué modo quieres? 0-aleatorio, 1-derecha, 2-guiado"))
filas = int(input("¿Cuántas filas tiene tu pasillo?"))
columnas = int(input("¿Cuántas columnas tiene tu pasillo?"))
cantidad_inicial_tipo_1 = int(input("Cuántos individuos quieres de izquiera a derecha?"))
cantidad_inicial_tipo_2 = int(input("Cuántos individuos quieres de derecha a izquierda?"))
flujo_1 = int(input("¿Cuántos quieres que entren por la izquierda?"))
if (flujo_1 == 0):
    modulo_1 = 1
else:    
    modulo_1 = int(input("Y cada cuánto?"))

flujo_2 = int(input("¿Cuántos quieres que entren por la derecha?"))
if (flujo_2 == 0):
    modulo_2 = 1
else:    
    modulo_2 = int(input("¿Y cada cuánto?"))
intervalos = int(input("¿Cuántos turnos quieres ver?"))

Flujo = {1: {2: flujo_1}, 2: {1: flujo_2}} #(tipo, entrada, cantidad)
T = [1, 2]
S1 = []
S2 = []
for k in range(0, filas): #salidas
    S1.append([k, 0])
    S2.append([k, columnas - 1])
    

s = {1: S1, 2: S2}
cantidad_gente = {1: cantidad_inicial_tipo_1, 2: cantidad_inicial_tipo_2}
corredor = Pasillo(filas, columnas, T, s)
corredor.llenador2(cantidad_gente)
start_time = time.time()
corredor.tiempo(intervalos, Flujo, modulo_1, modulo_2)
end_time = time.time()
elapsed_time = (end_time - start_time)
print(f"El código tardó {elapsed_time} segundos en ejecutarse.")
print(f"Se ha ejecutado direcciones {contador_direcciones} veces.")
