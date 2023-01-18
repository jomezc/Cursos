"""
APUNTES JESÚS GÓMEZ CÁNOVAS SOBRE PYTHON

Su código (en ejecución) se encuentra en la parte superior del mismo;
Python (más precisamente, su entorno de ejecución) se encuentra directamente debajo de él;
la siguiente capa de la pirámide está llena con el sistema operativo (sistema operativo):
el entorno de Python proporciona algunas de sus funcionalidades utilizando los servicios del sistema operativo;
Python, aunque muy potente, no es omnipotente: se ve obligado a usar muchos ayudantes si va a procesar archivos
o comunicarse con dispositivos físicos; La capa inferior es el hardware: el procesador (o procesadores),
las interfaces de red, los dispositivos de interfaz humana (ratones, teclados, etc.)
y toda otra maquinaria necesaria para que la computadora funcione;
el sistema operativo sabe cómo conducirlo y usa muchos trucos para conducir todas las partes a un ritmo constante.
"""

print("Texto")  # Imprimir por pantalla

# ***********************************
# ********  VARIABLES  #############
# ***********************************

# Variables no hay que indicar tipo de variables
miVariable = "hola mundo, desde python"
miVariable2 = 2
print(miVariable)
print(miVariable2)

# podemos reutilizar variables (el código se ejecutaba de arriba a abajo....)
miVariable2 = 10
print(miVariable2)

# cada variable se guarda en una posición de memoria en la memoria principal
x = 10
y = 2
z = x + y
print(x, y, z)

# con la función id podemos ver la dirección de memoria donde tenemos guardada la variable X
# cada vez que ejecutamos el programa al almacenarse en memoria ram va a ir cambiando de dirección en las ejecuciones
print(id(x))

# ***********************************
# ********  TIPOS DE DATOS  ##########
# ***********************************
# ******** ENTERO
x = 10
print(x)
# Con la función type podemos ver el tipo del dato
print(type(x))
# Representación de números en otras bases:
print(0o123)  # Octal: 0octal ,83  0O or 0o prefix [0..7] range
print(0x123)  # Hexadecimal: 0x123 ,291 0x or 0X prefix

# ******** STRING
x = "ju-ja"  # se puede utilizar comillas simples y dobles, pero deben comenzar y terminar igual
print(type(x))
print(11_11_11)  # 11_11_11 Python 3.6 has introduced underscores in numeric literals
# Como las variables pueden apuntar cualquier tipo de dato podemos agregar una pista (no las define)
s: str = 'hola'
print(x)
print(type(x))
s: str = 10  # al ser dinámicas y no definirlo podemos igualmente meter un número y funciona
print(x)
print('I\'m Monty Python')  # \ salvar caracteres para especiales como \n o para este ejemplo
# ******** FLOAT

x = 10.5  # Con ponerle el punto ya le estamos declarando una variable de tipo flotante
x = .5
print(x)
print(type(x))
print(3e3)  # Exponente 3 x 108.
print(3E8)
print(0.0000000000000000000001)
# el resultado de ejecutarlo es 1e-22 python escoge el modo más económico de representación de números
anything = float(input("Enter a number: "))
something = float(anything) ** 2.0
print(anything, "to the power of 2 is", something)

x = float(input("Enter value for x: "))


# ******** BOOL
x = True  # booleano, eso si respetar la mayúscula # False
print(type(x))
# OJO SABER QUE TODOS LOS TIPOS DE DATOS SON ALMACENADOS POR CLASES EN PYTHON
print(True > False)  # True
print(True < False)  # False
print(2 == 2.)  # True

# ***********************************
# ********  CADENAS  ############# secuencias inmutables
# ***********************************
'''
  ASCII (utilizado principalmente para codificar el alfabeto latino y algunos de sus derivados) 
y UNICODE (capaz de codificar prácticamente todos los alfabetos que utilizan los humanos).

revisa el código de la letra minúscula a. Esto es 97. Y ahora encuentra la A mayúscula. Su código es 65. 
Ahora calcula la diferencia entre el código de a y A. Es igual a 32. Ese es el código de un espacio.

Unicode asigna caracteres únicos (no ambiguos) (letras, guiones, ideogramas, etc.) a más de un millón de puntos de 
código. Los primeros 128 puntos de código Unicode son idénticos a ASCII, y los primeros 256 puntos de código Unicode 
son idénticos a la página de códigos ISO/IEC 8859-1 (una página de códigos diseñada para idiomas de Europa occidental).

Hay más de un estándar que describe las técnicas utilizadas para implementar Unicode en computadoras y sistemas 
de almacenamiento informático reales. El más general de ellos es UCS-4.
El nombre proviene de Universal Character Set. UCS-4 usa 32 bits (cuatro bytes) para almacenar cada carácter, 
y el código es solo el número único de los puntos de código Unicode. 
Un archivo que contiene texto codificado en UCS-4 puede comenzar con una BOM (marca de orden de bytes):
es una combinación especial no imprimible de bits que anuncian la codificación utilizada  por el contenido de un archivo 
(por ejemplo, UCS-4 o UTF-8).

UTF-8
Uno de los más utilizados es UTF-8.
El nombre se deriva del formato de transformación Unicode.
El concepto es muy inteligente. UTF-8 usa tantos bits para cada uno de los puntos de código como realmente 
necesita para representarlos.

Por ejemplo:
* todos los caracteres latinos (y todos los caracteres ASCII estándar) ocupan ocho bits;
* los caracteres no latinos ocupan 16 bits;
* Las ideografías CJK (China-Japón-Corea) ocupan 24 bits.

Python 3 es totalmente compatible con Unicode y UTF-8:
* puede usar caracteres codificados Unicode/UTF-8 para nombrar variables y otras entidades;
* puedes usarlos durante todas las entradas y salidas.
Esto significa que Python3 es completamente I18Ned.

2. Un número correspondiente a un carácter en particular se llama punto de código

Los /n se cuentan cómo carácter en las ‘’’ ( para multilínea son los 3, 1 solo 1 línea) o si se introducen, 
los vacíos no. + concatenar ( no conmutativo, el orden importa), * replicar n veces ( se pone número). 
variantes abreviadas de los operadores anteriores también son aplicables para cadenas (+= y *=)

Si desea conocer el valor del punto de código ASCII/UNICODE de un carácter específico, puede utilizar una función 
denominada ord() (como en ordinal).

La función necesita una cadena de un carácter como argumento; el incumplimiento de este requisito genera una excepción 
TypeError y devuelve un número que representa el punto de código del argumento.

por el contrario si conoces el valor del punto de código y quieres conocer el caracter: chr()
'''
char_1 = 'a'
char_11 = 'A'
char_2 = ' '  # space

print(ord(char_1))  # 97
print(ord(char_11))  # 65
print(ord(char_2))  # 32
print(chr(97))  # a
print(chr(65))  # A
print(len("\n\n"))  # 2

print("\"I\'m\"\n\"\"learning\"\"\n\"\"\"Python\"\"\"")
# "I'm"
# ""learning""
# """Python"""

# ******** Cadena (String), concatenar valores, con +
miGrupoFavorito = "ACDC" + " " + "The best rock band"
print("Mi grupo favorito es: " + miGrupoFavorito)  # en PRINT
b = "Mi grupo favorito es:"
print(b + " " + miGrupoFavorito)
# ******** podemos también usar comas que mete automáticamente espacio
print(b, miGrupoFavorito)
n1 = "1"
n2 = "2"
print(n1 + n2)  # Concatenación
n1 = 1
n2 = 2
# ******** Sobrecarga
print(n1 + n2)  # suma
n1 = "1"
n2 = 2
print("Concatenación: ", int(n1) + n2)  # conversión a entero int(), tiene que ser valido
print("Programming", "Essentials", "in", sep="***", end="...")  # Programming***Essentials***in...Python
print("Python")
# En las cadenas de Python, la barra invertida (\) es un carácter especial que anuncia que el siguiente carácter tiene
# un significado diferente, por ejemplo, \n (el carácter de nueva línea) inicia una nueva línea de salida.

# ++++ Ejercicio rectángulo +++++++++++
print("+" + 10 * "-" + "+")
print(("|" + " " * 10 + "|\n") * 5, end="")
print("+" + 10 * "-" + "+")

# ++++ Ejercicio Hipotenusa +++++++++++
leg_a = float(input("Input first leg length: "))  # float a número en punto flotante
leg_b = float(input("Input second leg length: "))
print("Hypotenuse length is " + str((leg_a**2 + leg_b**2) ** .5))  # 2 3 -> 3.605551275463989 str a char
x = int(input("Enter a number: "))  # The user enters 2
print(x * "5")

# como son secuencias se pueden recorrer como las listas (ejemplos en for) tanto por con índices positivos como
# negativos o rangos así como in o not in, al ser inmutables, pues eso NI del, insert o append se puede
the_string = 'silly walk'

for character in the_string:
    print(character, end=' ')  # 's i l l y  w a l k'

print(the_string[-1])  # k
print(the_string[1:3])  # il
print("f" in the_string)  # False

# ******** MÉTODOS DE CADENAS  #############

# ******** min() y max() (CADENAS Y NÚMEROS)
# Encuentra el elemento mínimo de la secuencia pasada como argumento. Hay una condición:
# la secuencia (cadena, lista, no importa) no puede estar vacía, de lo contrario obtendrá una excepción ValueError.
# -max(x) el mayor
print(min("aAbByYzZ"))  # A, es menor ascii
print(max("aAbByYzZ"))  # z, es mayor ascii

t = 'The Knights Who Say "Ni!"'
print('[' + min(t) + ']')  # espacio es el 32
print('[' + max(t) + ']')  # y

# También para números, puede además recibir más de dos parámetros devolverá el mayor o menor
t = [0, 1, 2]
print(min(t))  # 0 Menor ascii de los números
print(max(t))  # 2

number1 = int(input("Enter the first number: "))  # Enter the first number: 9999
number2 = int(input("Enter the second number: "))  # Enter the second number: 10000
number3 = int(input("Enter the third number: "))  # Enter the third number: 23

largest_number = max(number1, number2, number3)
print("The largest number is:", largest_number)  # The largest number is: 10000

# ******** list()
# Toma su argumento (una cadena) y crea una nueva lista que contiene todos los caracteres de la
# cadena, uno por elemento de la lista. Nota: no es estrictamente una función de cadena: list()
# puede crear una nueva lista a partir de muchas otras entidades (por ejemplo, de tuplas y diccionarios)
print(list("abcabc"))  # ['a', 'b', 'c', 'a', 'b', 'c']

# ******** count()
# Cuenta todas las ocurrencias del elemento en la secuencia. La ausencia de tales elementos no causa ningún problema.
print("abcabc".count("b"))  # 2
print('abcabc'.count("d"))  # 0

# ******** center()
# copia de la cadena original, centrándola con espacios, o con un número de ocurrencias del segundo parámetro
print('[' + 'alpha'.center(10) + ']')  # [  alpha   ]
print('[' + 'alpha'.center(10, '*') + ']')  # [**alpha***]

# ******** endswith()
# Verifica si la cadena dada termina con el argumento especificado y devuelve True o False, depende de la verificación.
if "epsilon".endswith("on"):
    print("yes")
else:
    print("no")
# yes

# **** startswith()
# Es un reflejo  de "endswith()": comprueba si una cadena dada comienza con la subcadena especificada.
print("omega".startswith("meg"))  # False
print("omega".startswith("om"))  # True

# ******** index()
# Busca la secuencia desde el principio, para encontrar el primer elemento del valor especificado en su argumento.
# Encuentra el elemento mínimo de la secuencia pasada como argumento. Hay una condición:
# la secuencia (cadena, lista, no importa) no puede estar vacía, es decir tiene que encontrarlo, de lo contrario
# obtendrá una excepción ValueError.
print("aAbByYzZaA".index("b"))  # 2
print("aAbByYzZaA".index("Z"))  # 7
print("aAbByYzZaA".index("A"))  # 1
print("aAbByYzZaA".index("0"))  # ValueError: substring not found

# ******** find()
# Es similar a index(), que ya conoce: busca una subcadena y devuelve el índice de su primera aparición, pero:
# es más seguro: no genera un error para un argumento que contiene una subcadena inexistente (devuelve -1 entonces)
# funciona solo con cadenas; no intente aplicarlo a ninguna otra secuencia.
# El segundo argumento especifica el índice en el que se iniciará la búsqueda (no tiene que caber dentro de la cadena).
# El tercer argumento es el límite superior (no incluido) de la búsqueda

print("Eta".find("ta"))  # 1
print("Eta".find("mma"))  # -1
print('kappa'.find('a', 2))  # 4
print('kappa'.find('a', 1, 4))  # 1 si incluye en el rango el principio
print('kappa'.find('a', 2))  # 4 desde sin hasta
print('kappa'.find('a', 2, 4))  # -1 no inclusive en el final es el rango según este ejemplo k 0, a 1, p 2, p 3, a 4
the_text = """A variation of the ordinary lorem ipsum
text has been used in typesetting since the 1960s"""

fnd = the_text.find('the')
while fnd != -1:
    print(fnd)
    fnd = the_text.find('the', fnd + 1)
    # 15 80

# ******** rfind()
# Hacen casi lo mismo que sus contrapartes (los que no tienen el prefijo r), pero comienzan sus
# búsquedas desde el final de la cadena, no desde el principio (por lo tanto, el prefijo r, de derecho).
print("tau tau tau".rfind("ta"))  # 8
print("tau tau tau".rfind("ta", 9))  # -1 desde el final al 9
print("tau tau tau".rfind("ta", 3, 9))  # 4 desde el 9 al 3

# ******** isalnum()
# Comprueba si la cadena contiene solo dígitos o caracteres alfabéticos (letras) y devuelve True o FalseTambién
t = 'Six lambdas'  # False por el espacio
print(t.isalnum())
t = 'ΑβΓδ'  # True son alfabéticos ...
print(t.isalnum())
t = '20E1'  # True
print(t.isalnum())

# ******** isalpha()
# solo letras:
print("Moooo".isalpha())  # True
print('Mu40'.isalpha())  # False

# ******** isdigit()
# solo dígitos:
print('2018'.isdigit())  # True
print("Year2019".isdigit())  # False

# ******** islower()
# Solo letras minúsculas:
print("Moooo".islower())  # False
print('moooo'.islower())  # True

# ********  isspace()
# Solo espacios:
print(' \n '.isspace())  # True
print(" ".isspace())  # True
print("mooo mooo mooo".isspace())  # False

# ********  isupper()
# solo mayúsculas
print("Moooo".isupper())  # False
print('MOOOO'.isupper())  # True

# ******** capitalize()
# Crea una nueva cadena llena de caracteres, si el primer carácter dentro de la cadena es una letra (nota: el primer
# carácter es un elemento con un índice igual a 0, no solo el primer carácter visible), se convertirá a mayúsculas;
# todas las letras restantes de la cadena se convertirán a minúsculas. La cadena original no cambia de ninguna manera
# La cadena modificada se devuelve como resultado
print("Alpha".capitalize())  # Alpha
print('ALPHA'.capitalize())  # Alpha
print(' Alpha'.capitalize())  # alpha
print('123'.capitalize())  # 123
print("αβγδ".capitalize())  # Αβγδ

# ******** swapcase()
# Crea una nueva cadena intercambiando las mayúsculas y minúsculas de todas las letras dentro
# de la cadena de origen: los caracteres en minúsculas se convierten en mayúsculas y viceversa. El resto de caracteres
# no se tocan
print("I know that I know nothing.".swapcase())  # i KNOW THAT i KNOW NOTHING.

# ******** title()
# realiza una función algo similar: cambia la primera letra de cada palabra a mayúsculas y
# cambia todas las demás a minúsculas
print("I know that I know nothing. Part 1.".title())    # I Know That I Know Nothing. Part 1.

# ******** upper()
# Hace una copia de la cadena de origen, reemplaza todas
# las letras minúsculas con sus equivalentes en mayúsculas y devuelve la cadena como resultado.
print("I know that I know nothing. Part 2.".upper())    # I KNOW THAT I KNOW NOTHING. PART 2.

# ******** lower()
# Hace una copia de una cadena de origen, reemplaza todas las letras mayúsculas con sus
# equivalentes en minúsculas y devuelve la cadena como resultado. Una vez más, la cadena de origen permanece intacta.
print("SiGmA=60".lower())  # sigma=60

# ***** join()
# como sugiere su nombre, el método realiza una unión: espera un argumento como una lista; debe asegurarse de que
# todos los elementos de la lista sean cadenas; de lo contrario, el método generará una excepción TypeError;
# todos los elementos de la lista se unirán en una cadena pero...
# ...la cadena desde la que se ha invocado el método se usa como separador, se coloca entre las cadenas;
# la cadena recién creada se devuelve como resultado.
print(",".join(["omicron", "pi", "rho"]))   # omicron,pi,rho
print("abc".join(["omicron", "pi", "rho"]))  # omicronabcpiabcrho
print("".join(["omicron", "pi", "rho"]))  # omicronpirho

# **** split()
# El método split() hace lo que dice: divide la cadena y crea una lista de todas las subcadenas detectadas.
# El método asume que las subcadenas están delimitadas por espacios en blanco: los espacios no participan en la
# operación y no se copian en la lista resultante. Si la cadena está vacía, la lista resultante también está vacía.
# Si le pones otra cosa pues es el separador
print("phi       chi\npsi".split())  # ['phi', 'chi', 'psi'] ignora el \n
print("phi       chi\npsi".split('\n'))  # ['phi       chi', 'psi']

# **** lstrip()
# Sin parámetros devuelve una cadena recién creada formada a partir de la original eliminando
# todos los espacios en blanco INICIALES.
print("[" + " tau ".lstrip() + "]")  # [tau ] OJO iniciales
# De un parámetro hace lo mismo que su versión sin parámetros,
# pero elimina sólo los caracteres incluidos en su argumento (una cadena), hasta encontrar otro carácter
print("www.cisco.com".lstrip("w."))  # cisco.com
print("pythoninstitute.org".lstrip(".org"))  # pythoninstitute.org

# **** rstrip()
# Lo mismo pero desde el otro extremo:
print("[" + " upsilon ".rstrip() + "]")  # [ upsilon]
print("cisco.com".rstrip(".com"))  # cis

# **** strip()
# Combina los efectos causados por rstrip() y lstrip() - crea una nueva cadena que carece de
# todos los espacios en blanco iniciales y finales (ojo no en medio).
print("[" + "   aleph   ".strip() + "]")  # [aleph]
print(".orgpythoninstitute.org".strip(".org"))  # pythoninstitute

# **** replace()
# De dos parámetros devuelve una copia de la cadena original en la que todas las apariciones
# del primer argumento han sido reemplazadas por el segundo argumento.
# Si el segundo argumento es una cadena vacía, reemplazar en realidad es eliminar la cadena del primer argumento.
# La variante replace() de tres parámetros usa el tercer argumento (un número) para limitar el número de reemplazos.
print("www.netacad.com".replace("netacad.com", "pythoninstitute.org"))  # www.pythoninstitute.org
print("This is it!".replace("is", "are"))  # Thare are it!
print("Apple juice".replace("juice", ""))  # Apple
print("This is it!".replace("is", "are", 1))  # Thare is it!
print("This is it is! is".replace("is", "are", 3))  # Thare are it! # ojo numero de remplazos no de caracteres


# ***********************************
# ********  BOOL  ###############
# ***********************************
# Está garantizado que False == 0 and True == 1,
miVariable = True
miVariable2 = False
miVariable3 = 2 < 3  # la comprobación devuelve un valor true o false
print(miVariable, miVariable2, miVariable3)


# ***********************************
# ********  None  ###############
# ***********************************
'''son solo dos tipos de circunstancias en las que None se puede usar de manera segura: cuando lo asigna a una 
variable (o lo devuelve como resultado de una función), y cuando lo comparas con una variable para diagnosticar su 
estado interno. Como aquí:'''
value = None
if value is None:
    print("Sorry, you don't carry any value")
