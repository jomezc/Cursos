
# ***********************************
# ******** LISTA ###############
# ***********************************
# LISTA = Conjunto de elementos, puede ser string o cualquier tipo
# empiezan por indice 0....
nombres = ['Juan', 'Fulgencio', 'Eusebio', 'pla']
# Imprimir lista
print(nombres)
# acceder a elementos
print(nombres[0])
# acceder a elementos de forma inversa
print(nombres[-1])
# acceder a rango
print(nombres[0:2])
print(nombres[0:-1])
print(nombres[:3])  # desde comienzo
print(nombres[1:])  # desde el indice hasta el final SI incluye el indicado
nombres[3] = 'lacasitos'  # asignar a elemento

for nombre in nombres:  # convenio singular/plural
    print(nombre)
else:
    print('no existen mas nombres en la lista')

# ******** len = longitud lista
print(len(nombres))
# ******** append añadir un elemento
nombres.append('leopardo')
# ******** insert insertar un elemento en un índice determinado
nombres.insert(1, 'Octal')
# ******** Eliminar un elemento, POR VALOR
nombres.remove('Fulgencio')
print(nombres)
# ******** Eliminar el ultimo valor agregado de la lista
nombres.pop()
# ******** Eliminar el por el índice de la lista
del nombres[1]
print(nombres)
# ******** Eliminar todos los elementos de la lista
nombres.clear()
# ********  eliminar de la memoria la lista
del nombres

# No hacemos copia sino apuntamos al otro sitio
list_1 = ["A", "B", "C"]
list_2 = list_1
list_3 = list_2

del list_1[0]
del list_2[0]

print(list_3)  # ['C']

list_1 = ["A", "B", "C"]
list_2 = list_1
list_3 = list_2

del list_1[0]
del list_2  # borra el apuntador

print(list_3)  # ["B", "C"]

# Intersección
a = [1, 2, 3]
b = [3, 4, 5]

print(bool(set(a) & set(b)))

set1 = {1, 2, 3, 4}
set2 = {2, 4, 6, 8}
set3 = {2, 3, 5, 7}
set4 = set1.intersection(set2, set3)
print(set4)  # {2}

my_list = [10, 1, 8, 3, 5]
length = len(my_list)

# intercambio elementos
for i in range(length // 2):
    my_list[i], my_list[length - i - 1] = my_list[length - i - 1], my_list[i]

print(my_list)

# ++++  bubble sort ++++++++++
my_list = []
swapped = True
num = int(input("How many elements do you want to sort: "))

for i in range(num):
    val = float(input("Enter a list element: "))
    my_list.append(val)

while swapped:
    swapped = False
    for i in range(len(my_list) - 1):
        if my_list[i] > my_list[i + 1]:
            swapped = True
            my_list[i], my_list[i + 1] = my_list[i + 1], my_list[i]

print("\nSorted:")
print(my_list)

# ******** métodos propios de phyton para ordenar y dar la vuelta a una lista
my_list = [8, 10, 6, 2, 4]
my_list.sort()
print(my_list)

lst = [5, 3, 1, 2, 4]
print(lst)

lst.reverse()

# SOLUCIONAR EL ERROR DE LAS LISTAS!!!!!!
# list_2 = list_1 no estás copiando el contenido
# sino la dirección de memoria donde está alojado
# Copying the entire list.
list_1 = [1]
list_2 = list_1[:]
list_1[0] = 2
print(list_2)  # [1]

# Copying some part of the list.
my_list = [10, 8, 6, 4, 2]
new_list = my_list[1:3]
print(new_list)

# ***********************************
# ******** compresión listas *************
# ***********************************
'''
# 1 La comprensión de listas le permite crear nuevas listas a partir de las existentes de una manera concisa y elegante. 
La sintaxis de un la comprensión de la lista se ve de la siguiente manera:
[expresión para el elemento en la lista si es condicional] que en realidad es un equivalente del siguiente código:
for elemento en la lista:
 si es condicional:
        expresión
Aquí hay un ejemplo de comprensión de una lista: el código crea una lista de cinco elementos llena con los primeros 
cinco números naturales elevados a la potencia de 3:'''
cubed = [num ** 3 for num in range(5)]
print(cubed)  # outputs: [0, 1, 8, 27, 64]

# Comprensión de listas es una forma de crear nuevas listas basadas en la lista existente. Ofrece una sintaxis más corta
# siendo más compacta y rápida que las otras funciones y bucles utilizados para crear una lista. Por ejemplo,
py_list = ['a-1', 'b-2', 'c-3', 'a-4']
r = [s for s in py_list if "a" in s]
print(r)  # Producción: ['a-1', 'a-4']

# En el código anterior, la comprensión de listas se utiliza para buscar cadenas que tengan a en la lista py_list. Tenga
# en cuenta que escribir el mismo código utilizando otras funciones o bucles habría llevado más tiempo, ya que se
# requiere más código para su implementación, pero la comprensión de listas resuelve ese problema. También podemos usar
# la comprensión de listas para encontrar cadenas que contengan múltiples valores específicos, es decir, podemos
# encontrar cadenas que contengan “a” y “b” en py_list combinando las dos comprensiones. Por ejemplo,

py_list = ['a-1', 'b-2', 'c-3', 'a-4', 'b-8']
q = ['a', 'b']
r = [s for s in py_list if any(xs in s for xs in q)]
print(r)  # Producción: ['a-1', 'b-2', 'a-4','b-8']

# ++++  Eliminar duplicados y ordenar +++++++++++++
lst = [10, 1, 2, 4, 4, 1, 4, 2, 6, 2, 9, 10]
lst = [lst[el] for el in range(len(lst)) if lst[el] not in lst[0:el]]
lst.sort()
print(lst)  # [1, 2, 4, 6, 9, 10]

# ++++   anidados ++++++++++
EMPTY = "-"
ROOK = "ROOK"
board = []

for i in range(8):
    row = [EMPTY for i in range(8)]
    board.append(row)

board[0][0] = ROOK
board[0][7] = ROOK
board[7][0] = ROOK
board[7][7] = ROOK

print(board)

# ++++ tiempo por hora por mes ++++++++++
temps = [[0.0 for h in range(24)] for d in range(31)]
#
# The matrix is magically updated here.

highest = -100.0

for day in temps:
    for temp in day:
        if temp > highest:
            highest = temp
print("The highest temperature was:", highest)

# ********  tridimensional
# (ejemplo hotel 3 edificios de 15 plantas con 20 habitaciones true o false ocupadas
# El primer índice (0 a 2) selecciona uno de los edificios; el segundo (0 a 14) selecciona el piso,
# el tercero (0 a 19) selecciona el número de habitación. Todas las habitaciones son inicialmente libres.
rooms = [[[False for r in range(20)] for f in range(15)] for t in range(3)]
# Check if there are any vacancies on the 15th floor of the third building:

vacancy = 0
for room_number in range(20):
    if not rooms[2][14][room_number]:
        vacancy += 1

# ******** función filter()
# para obtener una cadena específica en una lista de Python La función filter() filtra el iterable dado con la ayuda de
# una función que comprueba si cada elemento satisface alguna condición o no. Devuelve un iterador que aplica la
# comprobación para cada uno de los elementos del iterable. Por ejemplo,

py_lst = ['a-1', 'b-2', 'c-3', 'a-4']
filter(lambda x: 'a' in x, py_lst)
print(filter(lambda x: 'a' in x, py_lst))  # Producción:<filter object at 0x7fd36c1905e0> Tenga en cuenta que la salida
# anterior es un objeto de tipo filtro-iterador ya que la función filter() devuelve un iterador en lugar de una lista.
# Podemos usar la función list() como se muestra en el código siguiente para obtener una lista.
list(filter(lambda x: 'a' in x, py_lst))  # Producción:['a-1','a-4']
# En el código anterior, hemos utilizado filter() para encontrar una cadena con valores específicos en la lista py_list.


# ***********************************
# ******** Tuplas ###############
# ***********************************

# lista son modificables, insertables, modificables, guarda el orden
# Tupla son inmutables, puedes añadir pero no quitar
# ( PARÉNTESIS EN VEZ DE [
frutas = ('manzana', 'piña', 'melon')  # Si solo un elemento, debe terminar con coma
print(len(frutas))
one_element_tuple_1 = (1, )
one_element_tuple_2 = 1.,
# si quitamos la coma no tendríamos tuplas sino variables normales
tupla_1 = (1, 2, 4, 8)
tupla_2 = 1., .5, .25, .125
t = tuple([1, 2])

print(tupla_1)
print(tupla_2)

# Acceder a un elemento con corchetes, solo para declarar paréntesis
print(frutas[0])
print(frutas[-1])
print(frutas[:1])  # sin incluir ultimo indice
# recorrer elementos
for fruta in frutas:
    print(fruta, end=' ')  # PRINT OJO podemos sustituir el salto de línea print por otra cosa

my_tuple = (1, 10, 100)

t1 = my_tuple + (1000, 10000)  # junta varias tuplas en 1
t2 = my_tuple * 3  # multiplica tuplas, como la lista repite sus valores 3 veces

print(len(t2))
print(t1)
print(t2)
print(10 in my_tuple)  # si se encuentra o no
print(-10 not in my_tuple)

# Una de las propiedades de tupla más útiles es su capacidad para aparecer en el
# lado izquierdo del operador de asignación. Viste este fenómeno hace algún tiempo,
# cuando era necesario encontrar una herramienta elegante para intercambiar los valores de dos variables.
# Echa un vistazo al siguiente fragmento:
var = 123

t1 = (1, )
t2 = (2, )
t3 = (3, var)

t1, t2, t3 = t2, t3, t1

print(t1, t2, t3)

# Muestra tres tuplas interactuando - en efecto, los valores almacenados en ellas "circulan" -
# t1 se convierte en t2, t2 se convierte en t3 y t3 se convierte en t1.
# Pueden ser variables, no solo literales y si están en el lado derecho expresiones

# ********  DESEMPAQUETAR TUPLA
tup = 1, 2, 3
a, b, c = tup

print(a * b * c)

# método count() para tuplas Y LISTAS!
tup = 1, 2, 3, 2, 4, 5, 6, 2, 7, 2, 8, 9
duplicado = tup.count(2)

print(duplicado)
# ********  MODIFICAR TUPLA
#
# no buena praxis, si cogemos tupla es porque q
# frutas[0] = 'Pera' # falla porque es una tupla
frutaslista = list(frutas)  # creamos una lista a raiz de la tupla list()
frutaslista[0] = 'Pera'  # Como es una lista si podemos modificarlo
frutas = tuple(frutaslista)  # convertimos la lista a tupla y asignamos
print('\n', frutas)  # por la sentencia de arriba forzamos el \n

# ++++  Ejercicio tupla y lista ++++++++++
# Dada la siguiente tupla, crear una lista que sólo incluya los números menor que 5 utilizando un ciclo for:
tupla = (13, 1, 8, 3, 2, 5, 8)
lista = []
for t in tupla:
    if t < 5:
        lista.append(t)
print(lista, end=' ')


# ***********************************
# ******** SET ###############
# ***********************************
# Un set no mantiene un orden ni permite elementos duplicados, no es posible modificar elementos
# si es posible añadir o eliminarlos
planetas = {'Marte', 'Jupiter', 'Venus'}
print(planetas)  # no tiene orden
print(len(planetas))  # Longitud
planetas.add('Tierra')  # Añadir Elemento

# ******** Preguntar si elemento pertenece al set
print('Marte' in planetas)
# planetas.add('Tierra') de nuevo falla porque NO permite duplicado

# ******** Eliminar elemento
planetas.remove('Marte')  # Si no estuviera nos daría un key error
planetas.discard('Jupiter')  # Si no estuviera no da ningún error

# ******** Limpiar set
planetas.clear()

# ******** eliminar set
del planetas

# ***********************************
# ******** Diccionarios #############
# ***********************************
#   Colección de datos organizados por clave y valor (key:value)
#   No tenemos índices indicar directamente la llave
# dict (key, value)
diccionario = {
    'IDE': 'Integrated Development Environment',
    'OOP': 'Object Oriented Programming',
    'DBMS': 'Database Management System'
}
print(diccionario)
# largo
print(len(diccionario))

# ******** Recuperar Elemento
print(diccionario['IDE'])
print(diccionario.get('OOP'))  # Otra forma

# ********  cambiar elemento
diccionario['IDE'] = 'INTEGRATED DEVELOPMENT ENVIRONMENT'

# ********  Recorrer Elementos #############
for ter in diccionario:
    print(ter)  # Accedemos a las claves

for ter in diccionario.keys():  # usamos la función específica para recuperar la llave
    print(ter)  # Accedemos a las claves

for val in diccionario.values():  # necesitamos la función values() para recuperar solo el valor
    print(val)  # Accedemos a los valores

for ter, val in diccionario.items():  # necesitamos la función items() para recuperar ambos
    # devuelve una tupla!
    print(ter, val)  # Accedemos a los témrinos y valores

# ******** comprobar existencia
print('IDE' in diccionario)  # Debemos respetar mayúsculas y minúsculas

# ******** Agregar un elemento, si agregamos una existente SOBREESCRIBE
diccionario['PK'] = 'Primary Key'
diccionario.update({"pato": "canario"})

print(diccionario)

# ******** Remover elemento

diccionario.pop('DBMS')
del diccionario['PK']
diccionario.popitem()  # el último ( random en versiones viejas

# ******** limpiar para eliminar todos los elementos, usar la función clear()
diccionario.clear()

# ******** Eliminar el diccionario
del diccionario

# COPIAR Diccionario ( NO ASIGNAR!!) !!!!!!!!!!!!!!!
pol_eng_dictionary = {
    "zamek": "castle",
    "woda": "water",
    "gleba": "soil"
    }

copy_dictionary = pol_eng_dictionary.copy()

my_dictionary = {"A": 1, "B": 2}
copy_my_dictionary = my_dictionary.copy()
my_dictionary.clear()
print(copy_my_dictionary)  # {'A': 1, 'B': 2}

# sorted()
dictionary = {"cat": "chat", "dog": "chien", "horse": "cheval"}

for key in sorted(dictionary.keys()):
    print(key, "->", dictionary[key])

# La función sorted() hará lo mejor que pueda - la salida se verá así:
# cat -> chat
# dog -> chien
# horse -> cheval

# 2 DICCIONARIOS en 1 !
d1 = {'Adam Smith': 'A', 'Judy Paxton': 'B+'}
d2 = {'Mary Louis': 'A', 'Patrick White': 'C'}
d3 = {}

for item in (d1, d2):
    d3.update(item)

print(d3)

# ********  TUPLAS + DICCIONARIOS
# De tupla a diccionario
colors = (("green", "#008000"), ("blue", "#0000FF"))
colors_dictionary = dict(colors)
print(colors_dictionary)

# Las tuplas y los diccionarios pueden trabajar juntos Hemos preparado un ejemplo simple, que muestra cómo las tuplas y
# los diccionarios pueden trabajar juntos. Imaginemos el siguiente problema: necesita un programa para evaluar los
# puntajes promedio de los estudiantes; el programa debe pedir el nombre del estudiante, seguido de su puntaje único;
# los nombres podrán introducirse en cualquier orden; ingresar un nombre vacío finaliza la introducción de los datos
# (nota 1: ingresar una puntuación vacía generará la excepción ValueError, pero no se preocupe por eso ahora, verá cómo
# manejar tales casos cuando hablemos de excepciones en la segunda parte de la serie de cursos Python Essentials)
# A continuación, se debe emitir una lista de todos los nombres, junto con la puntuación media evaluada.

# ++++  Diccionario con media de las notas de los alumnos +++++++++
school_class = {}

while True:
    name = input("Enter the student's name: ")
    if name == '':
        break

    score = int(input("Enter the student's score (0-10): "))
    if score not in range(0, 11):
        break

    if name in school_class:
        school_class[name] += (score,)
    else:
        school_class[name] = (score,)

for name in sorted(school_class.keys()):
    adding = 0
    counter = 0
    for score in school_class[name]:
        adding += score
        counter += 1
    print(name, ":", adding / counter)
