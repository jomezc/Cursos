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
print(0.0000000000000000000001)  # el resultado de ejecutarlo es 1e-22 python escoge

# el modo más económico de representación de números
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
# Due to the low priority of the == operator, the question shall be treated as equivalent to this one:
# black_sheep == (2 * white_sheep)

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

por el contrario si conoces el valor del punto de código y quieres concer el caracter: chr()
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
# .max(x) el mayor
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
# Cuenta todas las ocurrencias del elemento dentro de la secuencia. La ausencia de tales elementos
# no causa ningún problema.
print("abcabc".count("b"))  # 2
print('abcabc'.count("d"))  # 0

# ******** center()
# Hace una copia de la cadena original,
# tratando de centrarla dentro de un campo de un ancho específico, o con un número de ocurrencias del segundo parámetro
print('[' + 'alpha'.center(10) + ']')  # [  alpha   ]
print('[' + 'alpha'.center(10, '*') + ']')  # [**alpha***]

# ******** endswith()
# Verifica si la cadena dada termina con el argumento especificado y devuelve True o False,
# según el resultado de la verificación.
if "epsilon".endswith("on"):
    print("yes")
else:
    print("no")
# yes

# **** startswith()
# Es un reflejo especular de "endswith()": comprueba si una cadena dada comienza
# con la subcadena especificada.
# Demonstrating the startswith() method:
print("omega".startswith("meg"))  # False
print("omega".startswith("om"))  # True

# ******** index()
# Busca la secuencia desde el principio, para encontrar el primer elemento del valor especificado en su argumento.
# Encuentra el elemento mínimo de la secuencia pasada como argumento. Hay una condición:
# la secuencia (cadena, lista, no importa) no puede estar vacía, de lo contrario obtendrá una excepción ValueError.
print("aAbByYzZaA".index("b"))  # 2
print("aAbByYzZaA".index("Z"))  # 7
print("aAbByYzZaA".index("A"))  # 1

# ******** find()
# Es similar a index(), que ya conoce: busca una subcadena y devuelve el índice de la primera
# aparición de esta subcadena, pero:
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
# si le pones otra cosa pues es el separador
print("phi       chi\npsi".split())  # ['phi', 'chi', 'psi']
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
#  Si el segundo argumento es una cadena vacía, reemplazar en realidad es eliminar la cadena del primer argumento.
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

# ***********************************
# ********  IF  ###############
# ***********************************
if miVariable2:  # LOS DOS PUNTOS SON NECESARIOS
    print("el resultado fue verdadero")
else:
    print("el resultado fue falso")

x, y, z = 5, 10, 8  # Desempaquetado
x, y, z = z, y, x

print(x > z)
print((y - 5) == x)

# ***********************************
# ********  Input ***************
# ***********************************
'''Función input para la entrada del usuario Input devuelve un string de lo introducido por el usuario,  con lo que 
para realizar operaciones numéricas tenemos que convertir a entero, por ejemplo'''
n = int(input("Escribe el primer numero: "))
m = int(input("Escribe el segundo numero: "))  # ahora si no metemos enteros falla claro
r = n + m
print("El resultado es:", r)

# ++++ Ejercicio
dia = int(input('valora tu día del 1 al 10: '))
print("Mi dia estuvo de:", dia)

# ***********************************
# ******** OPERADORES #########
# ***********************************

# **** SUMA +
# ojo de cadenas concatenación
opA = 3
opB = 2
suma = opA + opB
print('Resultado de la suma: ', suma)
# Literal precedido de 'f' y {}
print(f'resultado de la  suma: {suma}')

# **** RESTA -
# ojo de cadenas NO soportado
resta = opA - opB
print(f'Resultado de la resta: {resta}')
print(-4 - 4)  # -8
print(4. - 8)  # -8.
print(-1.1)  # -1.1

# ****  Multiplicación *
mult = opA * opB
print(f'Resultado de la multiplicación: {mult}')

# **** division normal /
div = opA / opB
print(f'Resultado de la división: {div}')

# ****  division entera ( sin parte fraccionaria) //
division = opA // opB
print(f'resultado division entera: {division}')

# **** modulo %
mod = opA % opB
print(f'Resultado del modulo: {mod}')

# **** exponente **
exp = opA ** opB
print(f'Resultado del exponente: {exp}')

# ++++  Ejercicio Rectángulo ++++++++++++++
alto = int(input('Proporciona el alto: '))
ancho = int(input('proporciona el ancho: '))
area = alto * ancho
perimetro = (alto + ancho) * 2
print(f'Área: {area}')
print(f'Perímetro: {ancho}')

# ***********************************
# ******** Operador asignación
# ***********************************

m = 10
m += 1  # incremento con reasignación,== m +1
print(m)
m -= 2  # decremento con reasignación
print(m)  # m = m -2
m *= 3  # m = m * 3
m /= 3
print(m)

print(9 % 6 % 2)  # 1 empieza por la izquierda
print(2 ** 2 ** 3)  # 256 menos la exponenciación que empieza por la derecha.
# operador unario es un operador con un solo operando, por ejemplo, -1 o +3.
print((-2 / 4), (2 / 4), (2 // 4), (-2 // 4))  # -0.5 0.5 0 -1

# ***********************************
# ******** COMPARACIONES##
# ***********************************
''' Python no es consciente (no puede serlo de ninguna manera) de los problemas lingüísticos sutiles: solo compara los 
valores de los puntos de código, carácter por carácter.Cuando comparas dos cadenas de diferentes longitudes y la más 
corta es idéntica al comienzo de la más larga, la cadena más larga se considera mayor.La comparación de cadenas siempre 
distingue entre mayúsculas y minúsculas (las letras mayúsculas se toman como menores que las minúsculas). No comparar 
string con números, 
string == number --> False;
string != number --> True;
string >= number --> falla.  # OJO'''

a = 4
b = 2
r = (a == b)  # lo del lado derecho se evalúa primero y luego se asigna
print(f'Resultado: {r}')
r = a != b
print(f'Resultado: {r}')
r = a > b  # mayor
r = a < b  # menor
r = a >= b  # mayor o igual
r = a <= b  # menor o igual
print(f'Resultado: {r}')
s1 = '12.8'
i = int(s1)  # value error aquí porque no podemos pasar el string 12.8 a entero a float si podríamos

'''
En general, Python ofrece dos formas diferentes de ordenar listas. El primero se implementa como una función llamada 
sorted(). La función toma un argumento (una lista) y devuelve una nueva lista, llena con los elementos del argumento 
ordenado. La lista original permanece intacta.
El segundo método afecta a la lista en sí: no se crea ninguna lista nueva. La ordenación se realiza in situ mediante 
el método denominado sort().
'''
first_greek = ['omega', 'alpha', 'pi', 'gamma']
first_greek_2 = sorted(first_greek)
print(first_greek)  # ['omega', 'alpha', 'pi', 'gamma']
print(first_greek_2)  # ['alpha', 'gamma', 'omega', 'pi']
# sorted() -> crea nueva lista ordenada; sort() ordena la lista original
second_greek = ['omega', 'alpha', 'pi', 'gamma']
print(second_greek)  # ['omega', 'alpha', 'pi', 'gamma']
second_greek.sort()
print(second_greek)  # ['alpha', 'gamma', 'omega', 'pi']

# ++++  Ejemplo prioridad +++++++++
#  1
# ------
# x + 1
#    ---
#  x + 1
#      --
#  x +  1
#       -
#       x
print("y =", 1/(x+(1/(x+(1/(x+(1/x)))))))

# ++++  conversor millas km ++++++++++

kilometers = 12.25
miles = 7.38

miles_to_kilometers = miles * 1.61
kilometers_to_miles = kilometers / 1.61

print(miles, "miles is", round(miles_to_kilometers, 2), "kilometers")
print(kilometers, "kilometers is", round(kilometers_to_miles, 2), "miles")

# ++++  ecuación sencilla ++++++++++
x = -1
x = float(x)
y = 3*(x**3) - 2*(x**2) + 3*x - 1
print("y =", y)


# ++++  Algoritmo pa_impar ++++++++
n = int(input("Ingresa un numero: "))
if n % 2 == 0:
    print(f'{n} es par')
else:
    print(f'{n} es impar')

# ++++  Algoritmo determinaMayorEdad +++++++++
ADULTO = 18
edad = int(input('Dime tu edad: '))
if edad >= ADULTO:
    print(f'tienes {edad}, por ello eres mayor de edad')
else:
    print(f'tienes {edad} por ello eres un cri@')

# ******** paso de minutos en reloj +++++++
"""
Su tarea es preparar un código simple capaz de evaluar la hora de finalización de un período de tiempo, dada como un 
número de minutos (podría ser arbitrariamente grande). La hora de inicio se da como un par de horas (0..23) y minutos 
(0..59). El resultado tiene que ser impreso en la consola. Por ejemplo, si un evento comienza a las 12:17 y dura 59 
minutos, terminará a las 13:16.
input: 12 17 59 --> output: 13:16
input: 23 58 642 --> Expected output: 10:40
"""
hour = int(input("Starting time (hours): "))
mins = int(input("Starting time (minutes): "))
dura = int(input("Event duration (minutes): "))
print(f'{(((mins+dura)//60)+hour)%24}:{(mins+dura)%60}')


# ***********************************
# ******** Operadores lógicos  ######
# ********       e IF          ######
# ***********************************
# pueden utilizarse para booleanos o expresiones
a = True
b = False
r = a and b
print(r)
# and
# or
r = a or b
print(r)
# not
r = not b
print(r)

'''
Ley de Morgan
- not (p and q) == (not p) or (not q)
- not (p or q) == (not p) and (not q)
Valores lógicos frente a bits individuales
Los operadores lógicos toman sus argumentos como un todo, sin que tenga importancia cuántos bits contengan.
Los operadores solo conocen el valor: cero (cuando se restablecen todos los bits) significa Falso;
no cero (cuando al menos un bit está establecido) significa Verdadero.
El resultado de sus operaciones es uno de estos valores: Falso o Verdadero.
Esto significa que este fragmento asignará el valor True a la variable j si i no es cero, de lo contrario, Falso.'''
a = 10
b = 4
"""
Operadores bit a bit ( SOLO ENTEROS)
Hay cuatro operadores que le permiten manipular bits individuales de datos. Se llaman operadores bit a bit.
Cubren  las operaciones en el contexto lógico y un operador adicional. Este es el operador 
xor (como en o exclusivo), y se denota como ^ (signo de intercalación).

Aquí están todos ellos:

& (ampersand) - conjunción bit a bit
    Su salida es 1 solo si todas las entradas son 1. Aquí está la tabla de verdad:
    0 AND 0 = 0
    0 AND 1 = 0
    1 AND 0 = 0
    1 AND 1 = 1;
"""
print("a & b =", a & b)  # a & b = 0
"""
| (bar) - disyunción bit a bit
    El operador OR también es conocido como disyunción lógica. 
    Da como salida 1 siempre que una o más de sus entradas sean 1. Aquí está la tabla de verdad:
    0 OR 0 = 0
    0 OR 1 = 1
    1 OR 0 = 1
    1 OR 1 = 1;
"""
print("a | b =", a | b)  # a | b = 14
"""
~ (tilde) - negación bit a bit
    lo contrario 0 -> 1 ; 1-> 0;
"""
print("~a =", ~a)  # ~a = -11
""" 
^ (signo de intercalación) - bit a bit exclusivo o (xor)
    El operador XOR tiene como salida un 1 siempre que las entradas no coincidan, lo cual ocurre cuando una de las dos 
    entradas es exclusivamente verdadera. Esto es lo mismo que la suma mod 2. 
     Aquí está la tabla de verdad:
    0 XOR 0 = 0
    0 XOR 1 = 1
    1 XOR 0 = 1
    1 XOR 1 = 0.
"""
print("a ^ b =", a ^ b)  # a ^ b = 14
"""
por ello:
& requiere exactamente dos 1 para proporcionar 1 como resultado;
| requiere al menos un 1 para proporcionar 1 como resultado;
^ requiere exactamente un 1 para proporcionar 1 como resultado.
"""
x = 4
y = 1
a = x & y
b = x | y
c = ~x  # tricky!
d = x ^ 5
e = x >> 2
f = x << 2
print(a, b, c, d, e, f)  # 0 5 -5 1 1 16
"""
**** Operadores de Turno
Estos operadores se utilizan para desplazar los bits de un número hacia la izquierda o hacia la derecha, multiplicando
o dividiendo el número por dos, respectivamente.Se pueden utilizar cuando tenemos que multiplicar o dividir un número 
por dos.
- Desplazamiento a la derecha bit a bit >>: desplaza los bits del número a la derecha y llena 0 en los vacíos a la 
  izquierda (llena 1 en el caso de un número negativo) como resultado. Efecto similar al dividir el número con alguna 
  potencia de dos.
- Desplazamiento bit a bit a la izquierda <<: Desplaza los bits del número a la izquierda y, como resultado, llena 0 en 
  los espacios vacíos a la derecha. Efecto similar al de multiplicar el número con alguna potencia de dos.
"""
# ++++  Ejemplos +++++++++
a = 10  # = 0000 1010 (Binary)
print(a >> 1)  # = 0000 0101 = 5
a = -10  # = 1111 0110 (Binary)
print(a >> 1)  # = 1111 1011 = -5

a = 5  # = 0000 0101 (Binary)
print(a << 1)  # = 0000 1010 = 10
print(a << 2)  # = 0001 0100 = 20
b = -10  # = 1111 0110 (Binary)
print(b << 1)  # = 1110 1100 = -20
print(b << 2)  # = 1101 1000 = -40

# ++++  ejemplo and  +++++++++
v = int(input('proporciona un valor numérico: '))
max = 5
r = (v > 0) and (v <= max)  # mas simple.... 0 < v <= máximo
if r:
    print(f'{v} está dentro de rango')
else:
    print(f'{v} está fuera de rango')

# ++++  ejemplo or +++++++++
vacas = True
diaDescanso = False
if vacas or diaDescanso:
    print('A vivir')
else:
    print('niño rata....')

# ++++  ejemplo ejemplo not ++++++
if not (vacas or diaDescanso):
    print('A vivir')
else:
    print('niño rata....')

# ++++  EJEMPLOS COMPLETOS +++++++++
edad = int(input('Dime tu edad: '))

if (20 <= edad <= 30) or (30 <= edad <= 40):
    print(f'{edad} esta dentro de los 20`s o de los 30`s')
    if edad == 30:
        print(f'{edad} esta justo entre ambos')
    elif 20 <= edad <= 30:
        print(f'{edad} esta dentro de los 20`s')
    else:
        print(f'{edad} esta dentro de los 30`s')
else:
    print(f'{edad} no está dentro de los 20`s o de los 30`s')

# ++++   EJEMPLOSSSS ++++++

n = int(input('Proporciona el numero1: '))
m = int(input('Proporciona el numero2: '))
if n > m:
    print(f'El numero mayor es: {n}')
elif n < m:
    print(f'El numero mayor es: {m}')
else:
    print(f'{n} y {m} son iguales')

# ++++  ejemplo +++++

print('Proporcione los siguientes datos del libro:')
nom = input('Proporciona el nombre: ')
pid = int(input('Proporciona el ID: '))
perras = float(input('Proporciona el precio: '))
env = bool(input('Indica si el envio es gratuito: '))

# ******** PRINT de varias lineas (con formato vamos)

print(f'''
nombre: {nom}
Id: {pid}
¿Envío gratuito?: {env}
''')

# ++++   mas ejercicio +++++++

numero = int(input('proporciona un valor entre 1 y 3: '))
numeroT = ''
if numero == 1:
    numeroT = 'uno'
elif numero == 2:
    numeroT = 'dos'
elif numero == 3:
    numeroT = 'tres'
else:
    numeroT = 'Fuera de rango'
print(f'{numero} - {numeroT}')

# ******** IF TERNARIO ###########
condicion = True
print('verdadero') if condicion else print('falso')

# ++++ Ejercicio IF ++++++++++

edad = int(input('Proporciona tu edad: '))
mensaje = ''
if 0 <= edad < 10:
    mensaje = 'la infancia  es increíble....'
elif 10 <= edad < 20:
    mensaje = 'Muchos cambios y mucho estudio....'
elif 20 <= edad <= 30:
    mensaje = 'Amor y comienza el trabajo...'
else:
    mensaje = 'Etapa de la ida no reconocida....'

print(f'tu edad es: {edad}, {mensaje}')

# ***********************************
# ******** WHILE  #######################
# ***********************************
condicion = True
while condicion:
    print('entro')
    condicion = False
else:  # OPCIONAL
    print('nop')
contador = 0
while contador <= 10:
    print(contador)
    contador += 1
else:  # Ojo en cada iteración
    print(f'fin del contador ultimo numero: {contador}')

c = 5
while c > 0:
    print(c)
    c -= 1

# Ejercicio
n = 0
while n <= 10:
    if n % 3 == 0:
        print(n)
    n += 1

# +++++ ejemplo
# programa que lea una secuencia de números cuenta cuantos son pares (odd) e impares (even), temrina al intorducir 0
odd_numbers = 0
even_numbers = 0

# Read the first number.
number = int(input("Enter a number or type 0 to stop: "))

# 0 terminates execution.
# while number != 0  # EQUIVALENTE! por True != 0:
while number:
    # Check if the number is odd.
    # if number % 2 == 1: # EQUIVALENTE!
    if number % 2:   # EQUIVALENTE! por True != 0
        # Increase the odd_numbers counter.
        odd_numbers += 1
    else:
        # Increase the even_numbers counter.
        even_numbers += 1
    # Read the next number.
    number = int(input("Enter a number or type 0 to stop: "))

# Print results.
print("Odd numbers count:", odd_numbers)
print("Even numbers count:", even_numbers)

# ***********************************
# ********  for  ###################
# ***********************************

# arreglo cadena de caracteres, iterar recorrer cada elemento
cadena = 'hola'
for letra in cadena:
    print(letra)
else:
    print('fin for')

# ********   break
cadena = 'holanda'
for letra in cadena:
    if letra == 'a':
        print(f' letra encontrada: {letra}')
        break  # Rompe el ciclo incluso el else !!!!
else:
    print('fin for')

# ********  continue

# rango de números range()
for i in range(10):
    if i % 2 == 0:
        print(f'Valor: {i}')
        continue  # ** ejecuta la siguiente iteración si cumple
    print(f'{i} no es par')

for i in range(10):
    if i % 3 == 0:
        print(i)

for n in range(2, 10, 2):  # range(start, stop, step)
    print(n)


# ********   enumerate
# En vez de hacer esto:
values = 'ABC'
index = 0
for value in values:
    print(index, value)
    index += 1

# O esto:
for index in range(len(values)):
    value = values[index]
    print(index, value)

# Cuando se utiliza enumerate(), la función devuelve dos variables de bucle:
# -El recuento de la iteración actual
# -El valor del elemento en la iteración actual
for count, value in enumerate(values):
    print(count, value)


# ++++ ejercicio +++++
import time
for i in range(1, 6): # cuenta de 1 inlcuido a 5 ( 6 no incluido)
    print(f" Mississippi {i}")
# Write a for loop that counts to five.
    # Body of the loop - print the loop iteration number and the word "Mississippi".
    # Body of the loop - use:
    time.sleep(1) # suspende la ejecución durante los segundos indicados

print("Ready or not, here I come!")
# Write a print function with the final message.

while True:
    if input("dime: ") == "chupacabra":
        break
print("adios")

# Prompt the user to enter a word
# and assign it to the user_word variable.
user_word = input("dime algo bonito, las vocales no molan: ")
for letter in user_word:
    if letter.upper() in ('A', 'E', 'I', 'O', 'U'):
        continue
    print(letter)
    # Complete the body of the for loop.


# ++++ ej 1 ++++++
"""
Cree un programa con un bucle for y una instrucción break. El programa debe iterar sobre los caracteres en una dirección 
de correo electrónico, salga del bucle cuando llegue al símbolo @ e imprima la parte antes de @ en una línea.
"""
for ch in "john.smith@pythoninstitute.org":
    if ch == "@":
        break
    print(ch, end="")
"""
Cree un programa con un bucle for y una declaración de continuación. El programa debe iterar Use el esqueleto a 
continuación:
"""
for digit in "0165031806510":
    if digit == "0":
        print("x", end="")
        continue
    print(digit, end="")

# ++++  Ejercicio +++++

n = int(input('Proporciona un valor entre 0 y 10: '))
while not (0 <= n <= 10):
    n = input('numero incorrecto, por favor ingresa un numero entre 0 y 10: ')

if 0 <= n < 6:
    print('F')
elif 6 <= n < 7:
    print('D')
elif 7 <= n <= 8:
    print('C')
elif 8 <= n <= 9:
    print('B')
elif 9 <= n <= 10:
    print('A')
else:
    print('No va a entrar aquí')

# ++++  Ejercicio pirámide +++++
"""
Su tarea es escribir un programa que lea la cantidad de bloques que tienen los constructores y muestre la altura de la 
pirámide que se puede construir usando estos bloques. Nota: la altura se mide por la cantidad de capas completamente 
completadas: si los constructores no tienen una cantidad suficiente de bloques y no pueden completar la siguiente capa,
terminan su trabajo de inmediato. Pruebe su código usando los datos que le hemos proporcionado.
Entrada de muestra: 6 --> Salida esperada: La altura de la pirámide: 3
Entrada de muestra: 1000 --> Salida esperada: La altura de la pirámide: 44
"""
blocks = int(input("Enter the number of blocks: "))
height = 0
number = 1
while blocks >= number:
    blocks -= number
    number += 1
    height += 1
print("The height of the pyramid:", height)

# ++++  ejercicio test hypotesis ++++++
# ojo! La cláusula else se ejecuta después de que el ciclo finaliza su ejecución,
# siempre que no haya sido terminado por break, por ejemplo:
"""
tome cualquier número entero no negativo y distinto de cero y asígnele el nombre c0; si es par, evalúe un nuevo c0 como 
c0 ÷ 2; de lo contrario, si es impar, evalúe un nuevo c0 como 3 × c0 + 1;
si c0 ≠ 1, salte al punto 2.
La hipótesis dice que independientemente del valor inicial de c0, siempre irá a 1.
Por supuesto, es una tarea extremadamente compleja usar una computadora para probar la hipótesis de cualquier número 
natural (incluso puede requerir inteligencia artificial), pero puede usar Python para verificar algunos números 
individuales. Tal vez incluso encuentres el que refutaría la hipótesis.
Escriba un programa que lea un número natural y ejecute los pasos anteriores siempre que c0 sea diferente de 1. 
También queremos que cuente los pasos necesarios para lograr el objetivo. Su código también debe generar todos los 
valores intermedios de c0. Pista: la parte más importante del problema es cómo transformar la idea de Collatz en un 
bucle while: esta es la clave del éxito.
Ejemplo de entrada: 15 
salida: 46 23 70 35 106 53 160 80 40 20 10 5 dieciséis 8 4 2 1
pasos = 17
"""
c0 = int(input("numero: "))  # Even || odd
step = 0
while c0>1:
    if c0 % 2 == 0:
        c0 /= 2
        c0=int(c0)
    else:
        c0 = 3* c0 +1
    step +=1
    print(c0)
print(f'steps = {step}')  # par || impar

# ++++  Ejercicio ++++++
''' Su tarea es escribir su propia función, que se comporta casi exactamente como el método split() original, es decir:
debe aceptar exactamente un argumento: una cadena;debe devolver una lista de palabras creadas a partir de la cadena,
dividida en los lugares donde la cadena contiene espacios en blanco; si la cadena está vacía, la función debería
devolver una lista vacía; su nombre debería ser mysplit() Utilice la plantilla en el editor. Prueba tu código
cuidadosamente '''


def mysplit(string):
    try:
        if type(string) != str:
            raise Exception('only accept strings')
        if string.isspace():
            return []
        # return string.split()
        # like is not use split
        string = string.strip()
        l = []
        aux = 0
        for s in range(len(string)):
            if string[s].isspace() or s == (len(string) -1):
                l.append(string[aux:s+1].strip())
                aux=s
        return l
    except Exception as e:
        print('raise an error:', e)


print(mysplit("To be or not to be, that is the question"))
# ['To', 'be', 'or', 'not', 'to', 'be,', 'that', 'is', 'the', 'question']
print(mysplit("To be or not to be,that is the question"))
# ['To', 'be', 'or', 'not', 'to', 'be,that', 'is', 'the', 'question']
print(mysplit("   "))   # []
print(mysplit(" abc "))     # ['abc']
print(mysplit(""))  # []

# ++++  Cifrado Cesar ++++++
text = input("Enter your message: ")  # bLa blaaa
cipher = ''
for char in text:
    if not char.isalpha():
        continue
    char = char.upper()
    code = ord(char) + 1
    if code > ord('Z'):
        code = ord('A')
    cipher += chr(code)
print(cipher)  # CMBCMBBB

# ++++ Descifrar mensaje +++++
cipher = input('Enter your cryptogram: ')  # CMBCMBBB
text = ''
for char in cipher:
    if not char.isalpha():
        continue
    char = char.upper()
    code = ord(char) - 1
    if code < ord('A'):
        code = ord('Z')
    text += chr(code)
print(text)  # BLABLAAA

# ++++  Procesador de numeros ++++++++.
line = input("Enter a line of numbers - separate them with spaces: ")  # 1 2 3 4 5 6
strings = line.split()
total = 0
try:
    for substr in strings:
        total += float(substr)
    print("The total is:", total)
except:
    print(substr, "is not a number.")  # 21.0

# ++++  Cifrado Cesar v2 +++++++
"""
El cifrado César original cambia cada carácter por uno: a se convierte en b, z se convierte en a, y así sucesivamente.
Hagámoslo un poco más difícil y permitamos que el valor desplazado provenga del rango 1..25 inclusive.
Además, deje que el código conserve el caso de las letras (las letras minúsculas seguirán siendo minúsculas)
y todos los caracteres no alfabéticos deben permanecer intactos.

Su tarea es escribir un programa que:
pide al usuario una línea de texto para cifrar;
pide al usuario un valor de cambio (un número entero del rango 1..25nota: debe obligar al usuario a ingresar
un valor de cambio válido (¡no se rinda y no deje que los datos incorrectos lo engañen!)
imprime el texto codificado.
"""
text = input("Inserta el mensaje: ")  # abcxyzABCxyz 123 || The die is cast
ok = False
while not ok:
    try:
        cifr = int(input("Inserta el número de cambio: "))  # 2 || 25
        if not 0 < cifr <= 25:
            raise Exception('El numero debe estar comprendido entre 1 y 25')
        ok = True
    except Exception as e:
        print(e)

cipher = ''

for char in text:
    mod = 25
    if 65 <= ord(char) <= 90 or 97 <= ord(char) <= 122:
        if 65 <= ord(char) <= 90:
            rest = 65
        else:
            rest = 97
        asci = ord(char) - rest
        new_alph_tmp = asci + cifr
        if new_alph_tmp == 25:
            mod += 1
        if new_alph_tmp > 25:
            new_alph = (new_alph_tmp % mod) - 1
        else:
            new_alph = new_alph_tmp % mod
        new_asci = new_alph + rest
        char = chr(new_asci)
        # quitamos el sobrante para jugar con el alfabeto, para
        # después sumarle el dígito acorde el actual jugando con el resto
    cipher += char
print(cipher)  # cdezabCDEzab 123 || Sgd chd hr bzrs


# ++++  Digito de la vida +++++
"""
Algunos dicen que el Dígito de la Vida es un dígito evaluado usando el cumpleaños de alguien. Es simple: solo necesita
sumar todos los dígitos de la fecha. Si el resultado contiene más de un dígito, debe repetir la suma hasta obtener
exactamente un dígito. Por ejemplo:
1 enero 2017 = 2017 01 01
2 + 0 + 1 + 7 + 0 + 1 + 0 + 1 = 12
1 + 2 = 3
3 es el dígito que buscamos y encontramos.

Su tarea es escribir un programa que:
pregunta al usuario su cumpleaños (en el formato AAAAMMDD, AAAADDMM o MMDDAAAA - en realidad, el orden de los dígitos
no importa) emite el dígito de vida para la fecha.
"""
text = input("Inserta Tu cumpleaños: ")  # 19991229 || 20000101
suma = 0
fin = False
while not fin:
    for t in text:
        suma += int(t)
    text = str(suma)
    suma = 0
    fin = True if len(text) == 1 else False
print(f'digito de la vida: {text}')  # 6 || 4

# ++++ Ejercicio CALC IBAN ++++++++++
"""
British: GB72 HBZU 7006 7212 1253 00
French: FR76 30003 03620 00020216907 50
German: DE02100100100152517108
"""

iban = input("Enter IBAN, please: ")
iban = iban.replace(' ','')

if not iban.isalnum():
    print("You have entered invalid characters.")
elif len(iban) < 15:
    print("IBAN entered is too short.")
elif len(iban) > 31:
    print("IBAN entered is too long.")
else:
    iban = (iban[4:] + iban[0:4]).upper()
    iban2 = ''
    for ch in iban:
        if ch.isdigit():
            iban2 += ch
        else:
            iban2 += str(10 + ord(ch) - ord('A'))
    iban = int(iban2)
    if iban % 97 == 1:
        print("IBAN entered is valid.")
    else:
        print("IBAN entered is invalid.")

# ++++  Cadena contenida en Cadena ++++++++

"""
Su tarea es escribir un programa que responda a la siguiente pregunta: ¿los caracteres que componen la primera cadena
están ocultos dentro de la segunda cadena?
si la segunda cadena se da como "vcxzxduybfdsobywuefgas", la respuesta es sí;
si la segunda cadena es "vcxzxdcybfdstbywuefsas", la respuesta es no (ya que no existen las letras "d", "o" o "g",
en este orden)
"""
st1 = 'donor'
st2 = 'Nabucodonosor'
aux = 0
response = 'Yes'
for s in st1:
    aux2 = st2.find(s)
    if aux2 < aux:
        response = 'No'
        break
print(response) #Yes || con donut como st1 --> No

# ++++  Sudoku  ++++++++
"""
Sudoku es un rompecabezas de colocación de números que se juega en un tablero de 9x9. El jugador tiene que llenar el
tablero de una manera muy específica:

Cada fila del tablero debe contener todos los dígitos del 0 al 9 (no importa el orden)
cada columna del tablero debe contener todos los dígitos del 0 al 9 (nuevamente, el orden no importa)
cada una de las nueve "fichas" de 3x3 (las llamaremos "subcuadrados") de la tabla debe contener todos los dígitos del
0 al 9. Si necesita más detalles, puede encontrarlos aquí.

Su tarea es escribir un programa que:

Lee 9 filas del Sudoku, cada una con 9 dígitos (verifique cuidadosamente si los datos ingresados son válidos)
emite Sí si el Sudoku es válido y No en caso contrario.
Pruebe su código usando los datos que le hemos proporcionado.

Datos de prueba
Ejemplo de entrada:
295743861 431865927 876192543 387459216 612387495 549216738 763524189 928671354 154938672 -> Salida de muestra: Sí

Ejemplo de entrada:
195743862 431865927 876192543 387459216 612387495 549216738 763524189 928671354 254938671 -> Salida de muestra: No
"""
tablero9x9 = input('Introduce tu jugada:').split()
# s = [[[[tablero9x9[y+zz][x+z] for x in range(3)] for y in range(3)] for z in range(0, 9, 3)] for zz in range(0, 9, 3)]
comprobacion = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
try:
    for j in range(len(tablero9x9)):
        # comprobamos la fila and comprobamos la columna
        if not (sorted(tablero9x9[j]) == comprobacion
                or sorted(''.join([tablero9x9[0][j], tablero9x9[1][j], tablero9x9[2][j], tablero9x9[3][j],
                                   tablero9x9[4][j], tablero9x9[5][j], tablero9x9[6][j], tablero9x9[7][j],
                                   tablero9x9[8][j]])) == comprobacion):
            raise Exception('No')
    # comprobamos los subcuadrados
    for zz in range(0, 9, 3):
        for z in range(0, 9, 3):
            subcuadrados = ''
            for y in range(3):
                for x in range(3):
                    subcuadrados += tablero9x9[y + zz][x + z]
            if sorted(subcuadrados) != comprobacion:
                raise Exception('No')
except Exception as e:
    print(e)
else:
    print('yes')


# ***********************************
# ******** match ###################
# ***********************************
# solo funciona a partir de la versión 3.10 o posterior


def number_to_string(argument):
    match argument:
        case 0:
            return "zero"
        case 1:
            return "one"
        case 2:
            return "two"
        case default:
            return "something"


number_to_string(0)

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
print(set4)

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
print(list_2)

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
print(r) # Producción: ['a-1', 'a-4']

# En el código anterior, la comprensión de listas se utiliza para buscar cadenas que tengan a en la lista py_list. Tenga
# en cuenta que escribir el mismo código utilizando otras funciones o bucles habría llevado más tiempo, ya que se
# requiere más código para su implementación, pero la comprensión de listas resuelve ese problema. También podemos usar
# la comprensión de listas para encontrar cadenas que contengan múltiples valores específicos, es decir, podemos
# encontrar cadenas que contengan “a” y “b” en py_list combinando las dos comprensiones. Por ejemplo,

py_list = ['a-1', 'b-2', 'c-3', 'a-4', 'b-8']
q = ['a', 'b']
r = [s for s in py_list if any(xs in s for xs in q)]
print(r) # Producción: ['a-1', 'b-2', 'a-4','b-8']

# ++++  Eliminar duplicados y ordenar +++++++++++++
lst = [10, 1, 2, 4, 4, 1, 4, 2, 6, 2, 9, 10]
lst = [lst[l] for l in range(len(lst)) if lst[l] not in lst[0:l]]
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
print(filter(lambda x: 'a' in x, py_lst)) # Producción:<filter object at 0x7fd36c1905e0> Tenga en cuenta que la salida
# anterior es un objeto de tipo filtro-iterador ya que la función filter() devuelve un iterador en lugar de una lista.
# Podemos usar la función list() como se muestra en el código siguiente para obtener una lista.
list(filter(lambda x: 'a' in x, py_lst)) # Producción:['a-1','a-4']
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

t1 = my_tuple + (1000, 10000) # junta varias tuplas en 1
t2 = my_tuple * 3 # multiplica tuplas, como la lista repite sus valores 3 veces

print(len(t2))
print(t1)
print(t2)
print(10 in my_tuple) # si se encuentra o no
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
diccionario.popitem() # el último ( random en versiones viejas

# ******** limpiar para eliminar todos los elementos, usar la función clear()
diccionario.clear()

# ******** Eliminar el diccionario
del diccionario

## COPIAR Diccionario ( NO ASIGNAR!!) !!!!!!!!!!!!!!!
pol_eng_dictionary = {
    "zamek": "castle",
    "woda": "water",
    "gleba": "soil"
    }

copy_dictionary = pol_eng_dictionary.copy()

my_dictionary = {"A": 1, "B": 2}
copy_my_dictionary = my_dictionary.copy()
my_dictionary.clear()
print(copy_my_dictionary) # {'A': 1, 'B': 2}

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


# ***********************************
# ******** Funciones #############
# ***********************************
# definir función def. 1º definir, después llamar (DEBAJO) miFunción también notación de camello La asignación de un
# valor al mensaje de nombre hace que Python olvide su rol anterior. La función denominada anteriomente deja de estar
# disponible si no ponemos llamamos a return, que devuelve la ejecución a donde estaba antes de la llamada a la función
# se ejecuta implícitamente al final, se puede usar para terminar la ejecución de la función a demanda Una variable
# existente fuera de una función tiene un alcance dentro de los cuerpos de las funciones, excluyendo aquellos de ellos
# que definen una variable del mismo nombre. También significa que el alcance de una variable existente fuera de una
# función se admite solo cuando se obtiene su valor (lectura). La asignación de un valor obliga a la creación de la
# propia variable de la función.

def my_function():
    print("Do I know that variable?", var)  # Do I know that variable? 1


var = 1
my_function()
print(var)  # 1


def my_function():  # Do I know that variable? 2
    var = 2
    print("Do I know that variable?", var)  # 2


var = 1
my_function()
print(var)

# Hay un método especial de Python que puede ampliar el alcance de una variable de una manera que incluye los cuerpos
# de las funciones (incluso si desea no solo leer los valores, sino también modificarlos). Tal efecto es causado por una
# palabra clave llamada global:


def my_function():
    global var
    var = 2
    print("Do I know that variable?", var)


var = 1
my_function() # Do I know that variable? 2
print(var) # 2


def mi_func(nombre, apellido):  # parámetro la variable con la que se define
    if nombre == '':
        return
    print(nombre, apellido)  # forma parte lo que está dentro de la indentación


mi_func('Jesus', 'Gomez')  # argumento valor que le paso


# si una función no devuelve un determinado valor mediante una cláusula de expresión return, se supone que
# implícitamente devuelve None.
def strange_function(n):
    if(n % 2 == 0):
        return True


print(strange_function(2))
print(strange_function(1))

# ******* Pasando el argumento con un valor predefinido si no se introduce


def introduction(first_name, last_name="Smith"):
    print("Hello, my name is", first_name, last_name)


introduction("Henry")
introduction("James", "Doe")


def suma(n=0, m: int = 0) -> int:  # = x valor por defecto -> pista :pista
    return n + m


re = suma()
print(f'Resultado de suma: {re}')
print(f'Resultado de suma: {suma(6, 8)}')  # Podemos llamar a la función

# La siguiente función da error porque un argumento sin valor por defecto va antes que uno que si. deben ir todos los
# que tienen valor sin defecto antes

# def add_numbers(a, b=2, c):
#    print(a + b + c)
# add_numbers(a=1, c=3)
# SyntaxError - a non-default argument (c) follows a default argument (b=2)


# ******* Pasando el argumento de palabras clave
# Python ofrece otra convención para pasar argumentos, donde el significado del argumento está dictado por su nombre,
# no por su posición: se llama paso de argumentos de palabras clave. Echa un vistazo al fragmento:
def introduction(first_name, last_name):
    print("Hello, my name is", first_name, last_name)


introduction(first_name = "James", last_name = "Bond")
introduction(last_name = "Skywalker", first_name = "Luke")


# Puede mezclar ambas modos si lo desea: solo hay una regla inquebrantable: debe poner los argumentos posicionales antes
# que los argumentos de palabras clave.
def adding(a, b, c):
    print(a, "+", b, "+", c, "=", a + b + c)


adding(4, 3, c=2)


# ******** definir función cuando NO sabemos cuantas variables vamos a recibir
# lo toma como una tupla
def listarNombres(*nombres):  # en docu oficial *args pero no tienes que elegir ese nombre
    for nombre in nombres:
        print(nombre)


listarNombres('Pablo', 'Pablito', 'Pablete', 'Juanete')


# ******** Es legal, y posible, tener una variable con el mismo nombre que el parámetro de una función.El fragmento
# ilustra el fenómeno:


def message(number):
    print("Enter a number:", number)


number = 1234
message(1)
print(number)

# Una situación como está activa un mecanismo
# llamado shadowing: El parámetro x sombrea cualquier variable del mismo nombre,
# pero...... sólo dentro de la función que define el parámetro.
# El parámetro denominado number es una entidad completamente diferente de la variable
# denominada number. Esto significa que el fragmento anterior producirá el siguiente
# resultado: Introduzca un número: 1
# 1234

# ++++ Indice de masa corporal +++++++


def IMC(weight, height):
    if height < 1.0 or height > 2.5 or \
    weight < 20 or weight > 200:
        # \ hace que se siga en la siguiente línea no solo
        # en comentarios o string sino en codigo también
        # pero tiene que ser la siguiente un comentario no lo permite
        return None

    return weight / height ** 2


print(IMC(352.5, 1.65))


# ++++  EJERCICIO FUNCIONES ++++++
#   Función con argumentos variables para sumar todos los valores recibidos
def suma(*args):
    x = 0
    for arg in args:
        x += arg
    return x


print(suma(1, 2, 3, 4, 5, 6, 7, 8))


# ++++  EJERCICIO FUNCIONES 2 ++++++++
#  Función con argumentos variables para multiplicar los valores
def mult(*args):
    a = 1
    for arg in args:
        try:
            int(arg)
            a *= arg
        except:
            print(f'solo numeros, {arg} ignorado.')
    return a


print(mult(1, 2, 3, 'l', 4, 5, 6, 'ñ'))


# ++++  funcion que permite recibir diccionarios ++++++++
def listarTerminos(**terminos):  # (**Kwargs)
    for llave, valor in terminos.items():
        print(f'{llave}:{valor}')


listarTerminos(IDE='blabla', PK='blublu')
listarTerminos(A=16, PK='blublu')


# ++++   Funcion lista +++++++++
def desplegarNombres(nombres):
    for nombre in nombres:
        print(nombre)


nombres = ['pa', 'pe', 'pi']
desplegarNombres(nombres)
desplegarNombres('carlos')  # Itera los caracteres, es decir el str
# desplegarNombres(6) desplegarNombres(6,8)int SI falla
desplegarNombres((6, 8))  # Como es una tupla no falla, lista como hemos visto idem
desplegarNombres([6, 9])


# ++++   Función Es primo ++++++
def is_prime(num):
    r = True
    for i in range(2,num):
        if i % 2 == 0:
            r = False
            break
    return r


for i in range(1, 20):
    if is_prime(i + 1):
        print(i + 1, end=" ")
print()  # 2 3 5 7 11 13 17 19

# ++++   Palíndromo +++++++++
'''suponga que una cadena vacía no es un palíndromo;
tratar las letras mayúsculas y minúsculas como iguales;
los espacios no se tienen en cuenta durante la verificación; trátelos como inexistentes;
hay más de unas pocas soluciones correctas; trate de encontrar más de una.'''


def palindromo(string):
    string = string.upper().replace(' ','')
    frase = 'Es un palíndromo'
    for a in range(len(string) - 1):
        if not string[a] == string[(len(string) - 1) - a]:
            frase = 'No es un palíndromo'
        if frase[0] == 'N' or a == (len(string) - 1) - a:
            break
    print(frase)


palindromo('Ten animals I slam in a net')

# ++++   Anagrama +++++++
'''Un anagrama es una nueva palabra formada al reorganizar las letras de una palabra, usando todas las letras originales
exactamente una vez. Por ejemplo, las frases "seguridad ferroviaria" y "cuentos de hadas" son anagramas,
 mientras que "yo soy" y "tú eres" no lo son.

pide al usuario dos textos separados; comprueba si los textos introducidos son anagramas e imprime el resultado.
suponga que dos cadenas vacías no son anagramas; tratar las letras mayúsculas y minúsculas como iguales;
los espacios no se tienen en cuenta durante la verificación; trátelos como inexistentes Pruebe su código usando los
datos que le hemos proporcionado.'''


def palindromo(string,string2):
    string = string.upper().replace(' ', '')
    string2 = string2.upper().replace(' ', '')
    frase = 'Es un anagrama'
    for a in range(len(string) - 1):
        if not string[a] in string2:
            frase = 'No es un anagrama'
        string2 = string2.replace(string[a], '', 1)
        string = string[a:]
        if frase[0] == 'N' or len(string) != len(string2):
            break
    print(frase)


palindromo('Listen','Silent')


# ++++   ejemplo funciones variables  +++++++
def printer(*args, **dics):
    for arg in args:
        print(arg)

    for clave, valor in dics.items():
        print(f'{clave} -->{valor} ')


printer(1, 2, 3, 'll', a='lol', b='lal')

# ******** Problema listas en Funciones ####
"""
El siguiente ejemplo arrojará algo de luz sobre el problema:

def my_function(my_list_1):
    print("Print #1:", my_list_1)
    print("Print #2:", my_list_2)
    my_list_1 = [0, 1]
    print("Print #3:", my_list_1)
    print("Print #4:", my_list_2)

my_list_2 = [2, 3]
my_function(my_list_2)
print("Print #5:", my_list_2)

salida:

Print #1: [2, 3]
Print #2: [2, 3]
Print #3: [0, 1]
Print #4: [2, 3]
Print #5: [2, 3]
output

Parece que la regla anterior todavía funciona. Finalmente, puedes ver la diferencia en el siguiente ejemplo:

def my_function(my_list_1):
    print("Print #1:", my_list_1)
    print("Print #2:", my_list_2)
    del my_list_1[0]  # OJO AQUI.
    print("Print #3:", my_list_1)
    print("Print #4:", my_list_2)


my_list_2 = [2, 3]
my_function(my_list_2)
print("Print #5:", my_list_2)


No cambiamos el valor del parámetro my_list_1 (ya sabemos que no afectará al argumento), sino que modificamos la lista 
identificada por él. El resultado puede ser sorprendente. Ejecute el código y verifique:

Print #1: [2, 3]
Print #2: [2, 3]
Print #3: [3]
Print #4: [3]
Print #5: [3]
output

si el argumento es una lista, cambiar el valor del parámetro correspondiente no afecta a la lista (recuerde: las 
variables que contienen listas se almacenan de una manera diferente a los escalares), Pero si cambia una lista 
identificada por el parámetro (nota: ¡la lista, no el parámetro!), La lista reflejará el cambio.
"""


# ++++  Es Triángulo y es triángulo equilátero +++++++
def is_a_triangle(a, b, c):
    return a + b > c and b + c > a and c + a > b


a = float(input('Enter the first side\'s length: '))
b = float(input('Enter the second side\'s length: '))
c = float(input('Enter the third side\'s length: '))

if is_a_triangle(a, b, c):
    print('Yes, it can be a triangle.')
else:
    print('No, it can\'t be a triangle.')


def is_a_right_triangle(a, b, c):
    if not is_a_triangle(a, b, c):
        return False
    if c > a and c > b:
        return c ** 2 == a ** 2 + b ** 2
    if a > b and a > c:
        return a ** 2 == b ** 2 + c ** 2


print(is_a_right_triangle(5, 3, 4))
print(is_a_right_triangle(1, 3, 4))


def heron(a, b, c):
    p = (a + b + c) / 2
    return (p * (p - a) * (p - b) * (p - c)) ** 0.5


def area_of_triangle(a, b, c):
    if not is_a_triangle(a, b, c):
        return None
    return heron(a, b, c)


print(area_of_triangle(1., 1., 2. ** .5))

# ++++    Conversión de Temperatura +++++++
# Realizar dos funciones para convertir de grados celsius a fahrenheit y viceversa. Función 1. Recibir un parámetro
# llamado celscius y regresar el valor equivalente a fahrenheit  La función se llama: celsius_fahrenheit(celsius)
# La fórmula para convertir de celsius a fahrenheit es: celsius * 9/5 + 32 Función 2. Recibir un parámetro llamado
# fahrenheit y regresar el valor equivalente a celsius: fahrenheit_celsius(fahrenheit)
# La fórmula para convertir de fahrenheit a celsius es:  (fahrenheit-32) * 5/9 Los valores los debe proporcionar el
# usuario, utilizando la función input y convirtiéndolo a tipo float. Deben hacer al menos dos pruebas, una donde
# conviertan de grados celscius a grados fahrenheit, y otra donde conviertan de grados fahrenheit a grados celsius y
# mandar a imprimir los resultados.


def celsius_fahrenheit(celsius):
    return (celsius * 1.8) + 32


def fahrenheit_celsius(farhenheit):
    return (farhenheit - 32) / 1.8


grados = float(input('introduce la temperatura en grados celsius para convertir a fahrenheit:'))
fahrenheit = celsius_fahrenheit(grados)
print(f'Son {fahrenheit} ºC')

grados = float(input('introduce la temperatura en grados celsius para convertir a fahrenheit:'))
celsius = fahrenheit_celsius(grados)
print(f'Son {celsius} ºC')


# ++++    ejercicio función año bisisesto. +++++
# Debe retornar True si el año es bisisto, False en caso contrario
def is_year_leap(year):
    r = False
    if year % 4 == 0: # divisible entre 4
        # salvo que sea año secular -último de cada siglo, terminado en «00»-, en cuyo caso también ha de ser divisible
        # entre 400.
        if str(year)[-2:] != '00' or (str(year)[-2:] == '00' and year % 400 == 0):
            r=True
    return r


test_data = [1900, 2000, 2016, 1987]
test_results = [False, True, True, False]
for i in range(len(test_data)):
    yr = test_data[i]
    print(yr,"->",end="")
    result = is_year_leap(yr)
    if result == test_results[i]:
        print("OK")
    else:
        print("Failed")


# ++++     EJERCIO FUNCION PANTALLA DE 13 LED (NUMEROS) ++++++
''' Su tarea es escribir un programa que pueda simular el trabajo de un dispositivo de siete pantallas, aunque
utilizará LED individuales en lugar de segmentos. Cada dígito se construye a partir de 13 LED, puede ser de cualquier
longitud de números
input: 9081726354 --> output:
### ### ###   # ### ### ### ### ### # #
# # # # # #   #   #   # #     # #   # #
### # # ###   #   # ### ### ### ### ###
  # # # # #   #   # #   # #   #   #   #
### ### ###   #   # ### ### ### ###   #
'''


def pantalla(str):
    # Simulamos un "led de 13 luces" 0 APAGADO, 1 ENCENDIDO
    # A1A2A3
    # F-  B-
    # G1G2G3
    # E-  C-
    # D1D2D3
    led = {
        #  [A1, A2, A3, B, C, D1, D2, D3, E, ,G1, G2, G3, F]
        0: [1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  0,  1,  1],
        1: [0,  0,  1,  1, 1,  0,  0,  1, 0,  0,  0,  1,  0],
        2: [1,  1,  1,  1, 0,  1,  1,  1, 1,  1,  1,  1,  0],
        3: [1,  1,  1,  1, 1,  1,  1,  1, 0,  1,  1,  1,  0],
        4: [1,  0,  1,  1, 1,  0,  0,  1, 0,  1,  1,  1,  1],
        5: [1,  1,  1,  0, 1,  1,  1,  1, 1,  1,  1,  1,  1],
        6: [1,  0,  0,  0, 1,  1,  1,  1, 1,  1,  1,  1,  1],
        7: [1,  1,  1,  1, 1,  0,  0,  1, 0,  0,  0,  1,  0],
        8: [1,  1,  1,  1, 1,  1,  1,  1, 1,  1,  1,  1,  1],
        9: [1,  1,  1,  1, 1,  1,  1,  1, 0,  1,  1,  1,  1]
    }
    # IMPRIME PRIMERA LINEA DEL LED: A1A2A3
    for s in str:
        for a in range(3):
            if led.get(int(s))[a] == 1:
                print('#', end='')
            else:
                print(' ', end='')
        print('', end=' ')
    print(end='\n')

    # IMPRIME SEGUNDA LINEA DEL LED: F B
    for s in str:
        if led.get(int(s))[-1] == 1:
            print('#', end='')
        else:
            print(' ', end='')
        print(' ', end='')

        if led.get(int(s))[3] == 1:
                print('#', end='')
        else:
            print(' ', end='')
        print('', end=' ')
    print()

    # IMPRIME TERCERA LINEA DEL LED: G1G2G3
    for s in str:
        for a in range(9, 12):
            if led.get(int(s))[a] == 1:
                print('#', end='')
            else:
                print(' ', end='')
        print('', end=' ')
    print()

    # IMPRIME CUARTA LINEA DEL LED: E C
    for s in str:
        if led.get(int(s))[8] == 1:
            print('#', end='')
        else:
            print(' ', end='')
        print(' ', end='')

        if led.get(int(s))[4] == 1:
            print('#', end='')
        else:
            print(' ', end='')
        print('', end=' ')
    print()

    # IMPRIME QUINTA LINEA DEL LED: D1D2D3
    for s in str:
        for a in range(5, 8):
            if led.get(int(s))[a] == 1:
                print('#', end='')
            else:
                print(' ', end='')
        print('', end=' ')
    print()


pantalla('9081726354')

# Versión con menos codigo pero mas complejidad (n*n vs MIA 5*n)
# using a list containing patterns （0~9）

list= [['###', '# #', '# #', '# #', '###'],
[' #', ' #', ' #', ' #', ' #'],
['###', '  #', '###', '#  ', '###'],
['###', '  #', '###', '  #', '###'],
['# #', '# #', '###', '  #', '  #'],
['###', '#  ', '###', '  #', '###'],
['###', '#  ', '###', '# #', '###'],
['###', '  #', '  #', '  #', '  #'],
['###', '# #', '###', '# #', '###'],
['###', '# #', '###', '  #', '###']
]


def print_number(num):
    string = str(num)
    for i in range(0,5):
        for character in string:
            print(list[int(character)][i],end = ' ')
        print()


print_number(int(input("Enter the number you wish to display: ")))


# ***********************************
# ******** FUNCIONES RECURSIVAS
# ***********************************

# ++++     Factorial +++++++
# 5! = 5 * 4 * 3 * 2 * 1
# 5! = 5 * 4 * 3 * 2
# 5! = 5 * 4 * 6
# 5! = 5 * 24
# 5! = 120
# sin recursividad:
def factorial_function(n):
    if n < 0:
        return None
    if n < 2:
        return 1

    product = 1
    for i in range(2, n + 1):
        product *= i
    return product


for n in range(1, 6):  # testing
    print(n, factorial_function(n))


# con recursividad
def factorial(numero):
    if numero == 1:  # El factorial de 1 es 1 CASO BASE
        return 1
    else:
        return numero * factorial(numero - 1)  # LLAMADA RECURSIVA


print(factorial(5))


# ++++    Fibonacci ++++++
# fib_1 = 1
# fib_2 = 1
# fib_3 = 1 + 1 = 2
# fib_4 = 1 + 2 = 3
# fib_5 = 2 + 3 = 5
# fib_6 = 3 + 5 = 8
# fib_7 = 5 + 8 = 13
# sin recursividad:
def fib(n):
    if n < 1:
        return None
    if n < 3:
        return 1

    elem_1 = elem_2 = 1
    the_sum = 0
    for i in range(3, n + 1):
        the_sum = elem_1 + elem_2
        elem_1, elem_2 = elem_2, the_sum
    return the_sum


for n in range(1, 10):  # testing
    print(n, "->", fib(n))


# con recursividad:
def fib(n):
    if n < 1:
        return None
    if n < 3:
        return 1
    return fib(n - 1) + fib(n - 2)


# ++++     EJERCICIO RECURSIVIDAD SUMA ++++++++++
# Imprimir números de 5 a 1 de manera descendente usando funciones recursivas. Puede ser cualquier valor positivo,
# ejemplo, si pasamos el valor de 5, debe imprimir: 5 4 3 2 1 Si se pasa el valor de 3, debe imprimir: 3 2 1 Si se
# pasan valores negativos no imprime nada
def imprimeNumerosPositivos(numero):
    try:
        numero = int(numero)
        if numero > 0:
            print(numero)
            imprimeNumerosPositivos(numero - 1)
    except:
        print("Debes introducir un número")


imprimeNumerosPositivos(16)


# ******** EJERCICIO Calculadora de impuestos
# Crear una función para calcular el total de un pago incluyendo un impuesto aplicado.La función se llama
# calcular_total() La función recibe dos parámetros:
# 1. pago_sin_impuesto
# 2. impuesto (Ej. Valor de 10, significa 10% de impuesto, Valor de 16 significa el 16% de impuesto)
# La función debe regresar el total del pago incluyendo el porcentaje de impuesto proporcionado.
# Ej. Si llamamos a la función calcular_total(1000, 16) debe retornar el valor 1,160.0
# Los valores los debe proporcionar el usuario y se procesados con la función input, convirtiendolos a tipo float.
def calculadoraImpuestos(numero, impuesto):
    return valor * (1 + impuesto / 100)


#    return valor + valor*(impuesto/100)
valor = float(input('introduce un valor'))
impuesto = float(input('introduce el impuesto para el valor introducido'))
resultado = calculadoraImpuestos(valor, impuesto)
print(f'Pago con impuesto: {resultado}')


# ***********************************
# ********  lambda ##########
# ***********************************
'''Una función lambda es una función sin nombre (también puede llamarla función anónima). Por supuesto, tal declaración 
plantea inmediatamente la pregunta: ¿cómo se usa algo que no se puede identificar? Afortunadamente, no es un problema, 
ya que puede nombrar dicha función si realmente lo necesita, pero, de hecho, en muchos casos, la función lambda puede 
existir y funcionar mientras permanece completamente incógnita. La declaración de la función lambda no se parece en nada
a una declaración de función normal; compruébelo usted mismo:

lambda parameters: expression

Dicha cláusula devuelve el valor de la expresión cuando se tiene en cuenta el valor actual del argumento lambda actual.
'''
two = lambda: 2
sqr = lambda x: x * x
pwr = lambda x, y: x ** y

for a in range(-2, 3):
    print(sqr(a), end=" ")
    print(pwr(a, two()))

'''
4 4
1 1
0 0
1 1
4 4
la primera lambda es una función anónima sin parámetros que siempre devuelve 2. Como la hemos asignado a una variable
llamada dos, podemos decir que la función ya no es anónima y podemos usar el nombre para invocarla.la segunda es una 
función anónima de un parámetro que devuelve el valor de su argumento al cuadrado. También lo hemos nombrado como tal.
la tercera lambda toma dos parámetros y devuelve el valor del primero elevado a la potencia del segundo.

La parte más interesante de usar lambdas aparece cuando puede usarlas en su forma pura, como partes anónimas de código 
destinadas a evaluar un resultado. Imagine que necesitamos una función (la llamaremos print_function) que imprime los 
valores de una función dada (otra) para un conjunto de argumentos seleccionados. Queremos que print_function sea 
universal; debe aceptar un conjunto de argumentos colocados en una lista y una función para evaluar, ambos como 
argumentos; no queremos codificar nada.
'''
def print_function(args, fun):
    for x in args:
        print('f(', x,')=', fun(x), sep='')


def poly(x):
    return 2 * x**2 - 4 * x + 2


print_function([x for x in range(-2, 3)], poly)
'''
Vamos a analizarlo. La función print_function() toma dos parámetros:

el primero, una lista de argumentos para los que queremos imprimir los resultados;
el segundo, una función que debe ser invocada tantas veces como valores se recogen dentro del primer parámetro.
Nota: también hemos definido una función llamada poly() - esta es la función cuyos valores vamos a imprimir. El cálculo
que realiza la función no es muy sofisticado: es el polinomio f(x) = 2x2 - 4x + 2

Luego, el nombre de la función se pasa a print_function() junto con un conjunto de cinco argumentos diferentes: 
el conjunto se crea con una cláusula de comprensión de lista. El código imprime en varias líneas: f(-2)=18 f(-1)=8 
f(0)=2 f(1)=0 f(2)=2

¿Podemos evitar definir la función poly(), ya que no la usaremos más de una vez? Sí, podemos: este es el beneficio que 
puede brindar una lambda.
'''
def print_function(args, fun):
    for x in args:
        print('f(', x,')=', fun(x), sep='')

print_function([x for x in range(-2, 3)], lambda x: 2 * x**2 - 4 * x + 2)
'''
print_function() se ha mantenido exactamente igual, pero no hay función poly(). Ya no lo necesitamos, ya que el 
polinomio ahora está directamente dentro de la invocación de print_function() en forma de lambda.El código se ha vuelto
más corto, más claro y más legible.
'''
# ******** la función map():
'''map (función, lista)
toma dos argumentos: Una función y una lista. el segundo argumento map() puede ser cualquier entidad que se pueda iterar
(por ejemplo, una tupla o simplemente un generador) map() puede aceptar más de dos argumentos. La función map() aplica 
la función pasada por su primer argumento a todos los elementos de su segundo argumento y devuelve un iterador que 
entrega todos los resultados de la función subsiguiente. Puede usar el iterador resultante en un bucle o convertirlo en 
una lista usando la función list().
'''
list_1 = [x for x in range(5)]
list_2 = list(map(lambda x: 2 ** x, list_1))
print(list_2)  # [1, 2, 4, 8, 16]

for x in map(lambda x: x * x, list_2):
    print(x, end=' ')
print()  # 1 4 16 64 256


'''
construye list_1 con valores de 0 a 4; a continuación, usa map junto con la primera lambda para crear una nueva lista 
en la que todos los elementos se hayan evaluado como 2 elevados a la potencia tomada del elemento correspondiente de 
list_1; list_2 se imprime entonces; en el siguiente paso, use la función map() nuevamente para hacer uso del generador 
que devuelve e imprimir directamente todos los valores que entrega; como puede ver, hemos activado la segunda lambda 
aquí: simplemente cuadra cada elemento de list_2.
'''

# ******** la función filter():
'''Otra función de Python que puede embellecerse significativamente mediante la aplicación de una lambda es filter().

Espera el mismo tipo de argumentos que map(), pero hace algo diferente: filtra su segundo argumento mientras se guía por
las direcciones que fluyen de la función especificada como primer argumento (la función se invoca para cada elemento de
la lista, al igual que en map ()). Los elementos que devuelven True de la función pasan el filtro; los demás son 
rechazados.
'''
from random import seed, randint
seed()
data = [randint(-10,10) for x in range(5)]
filtered = list(filter(lambda x: x > 0 and x % 2 == 0, data))
print(data)
print(filtered)
'''
Nota: hemos utilizado el módulo radmon para inicializar el generador de números aleatorios (que no debe confundirse 
con los generadores de los que acabamos de hablar) con la función seed()  y para producir cinco valores enteros 
aleatorios de -10 a 10 usando la función randint().Luego se filtra la lista y solo se aceptan los números pares y 
mayores que cero.'''


# ++++ Ejemplos +++++
x = lambda a: a + 10
print(x(5))

x = lambda a, b: a * b
print(x(5, 6))

x = lambda a, b, c: a + b + c
print(x(5, 6, 2))


nombre_completo = lambda n, a: n.strip().title() + " " + a.strip().title()
print(nombre_completo("   jesus", "   GOMEZ"))

# ++++ Ordenar lista por apellido +++++
#  minúsculas por si acaso
# split separa por el espacio y empezamos por el final hasta ese espacio, lower todo a
lista = ["Jesus Gomez", "María Macanás", "Marisa Baños", "Maria Canovas"]
lista.sort(key=lambda name: name.split(" ")[-1].lower())
print(lista)


# tienes una definición de función que toma un argumento, y ese argumento se multiplicará por un número desconocido:
def myfunc(n):
    return lambda a: a * n


mydoubler = myfunc(2)  # ese numero por el que se va a multiplicar

print(mydoubler(11))  # es 'x' de lambda el valor a coger

mytripler = myfunc(3)

print(mydoubler(11))
print(mytripler(11))


# función constructora de ecuaciones cuadráticas
def constructor_ecuación_cuadrática(a, b, c):
    """
    :return: devuelve la función f(x) = ax^2 +bx + c
    """
    return lambda x: a * x ** 2 + b * x + c


# Construimos una ecuación específica
f = constructor_ecuación_cuadrática(2, 3, -5)
# probamos valores
print(f(2))
print(f(4))

# Construimos y probamos a la vez
print(constructor_ecuación_cuadrática(2, 0, 1)(3))

# ***********************************
# ******** Módulos e import #############
# ***********************************
# Python tiene una forma de poner definiciones en un archivo y usarlas en un script o en una instancia interactiva
# del intérprete. Dicho archivo se denomina módulo; Las definiciones de un módulo se pueden importar a otros módulos
# o al módulo principal (la colección de variables a las que tiene acceso en un script ejecutado en el nivel superior
# y en modo calculadora).
# Un módulo es un archivo que contiene definiciones y declaraciones de Python. El nombre de archivo es el nombre del
# módulo con el sufijo.py anexado. Dentro de un módulo, el nombre del módulo (como una cadena) está disponible
# como el valor de la variable global __name__.

# podemos importar un modulo de varias formas:
import math # importamos el módulo en su totalidad no se comparte el espacio de nombres ( no entra en conflicto
# otra función con el mismo nombre que tengas creada y para acceder a ella tienes que usar el nombre.
# podemos usar varias líneas para cada módulo o varios con , import math,sys
print(math.pi)

from math import sin # podemos importar una funcion de un módulo, aquí SI PUEDE ENTRAR EN CONFLICTO C
# ON NUESTRO ESPACIO DE NOMBRES,  es decir si tenemos una funcion = la sobrescribimos y al revés
sin(16) # lo usamos sin nombre de módulo ni nada

import math as m # se le puede dar un alias al módulo ( deja de ser accesible el nombre origianl.
print(m.pi)

from math import sin as s, cos as cc # podemos poner alias de los from también
print(s(3), cc(16))

# ****** dir()
# La función devuelve una lista ordenada alfabéticamente que contiene todos los nombres de entidades disponibles
# en el módulo identificado por un nombre pasado a la función como argumento: dir(módulo)
import math
print(dir(math)) # devuelve una lista

import math
for name in dir(math):
    print(name, end="\t")


# ***** math
# modulo de matemáticas
math.sin(x)     # el seno de x;
math.cos(x)     # el coseno de x;
math.tan(x)     # la tangente de x.
# Por supuesto, también están sus versiones inversas:
math.asin(x)    # el arcoseno de x;
math.acos(x)    # el arccoseno de x;
math.Atan(x)    # la arcotangente de x.
# Todas estas funciones toman un argumento (una medida de ángulo expresada en radianes) y devuelven el resultado
# apropiado (tenga cuidado con tan() - no todos los argumentos son aceptados).
# Análogos hiperbólicos todos igual pero terminan en h:

# Para operar eficazmente en mediciones de ángulo, el módulo matemático le proporciona las siguientes entidades:
math.pi    # Una constante con un valor que es una aproximación de π;
math.radians(x)     # Una función que convierte x de grados a radianes;
math.degrees (x)    # Actuando en la otra dirección (de radianes a grados)

# Exponenciación:
math.e   # una constante con un valor que es una aproximación del número de Euler (e)
math.exp(x)  # encontrar el valor de e elevado a x;
math.log(x)  # el logaritmo natural de x
math.log(x, b)  # el logaritmo de x a base b
math.log10(x)   # el logaritmo decimal de x (más preciso que log(x, 10))
math.log2(x)    # el logaritmo binario de x (más preciso que log(x, 2))
pow(x, y)   # encontrar el valor de x elevado a y (cuidado con los dominios)
# Esta es una función incorporada y no tiene que importarse.

# Propósito general
math.ceil(x)    # el entero más pequeño MAYOR o igual que x
math.floor(x)   # el entero más grande MENOR o igual que x)
math.trunc(x)   # el valor de x truncado a un entero (ten cuidado, no es un equivalente ni de ceil ni de floor)
math.factorial(x)  # devuelve x! (x tiene que ser una integral y no una negativa)
math.hypot(x, y)  # devuelve la longitud de la hipotenusa de un triángulo rectángulo con las longitudes de las piernas
# iguales a x e y (igual que sqrt(pow(x, 2) + pow(y, 2)) pero más precisa)

# ********* random  modulo de números aleatorios
# Un generador de números aleatorios toma un valor llamado semilla,
# lo trata como un valor de entrada, calcula un número "aleatorio" basado en él (el método depende de un algoritmo
# elegido) y produce un nuevo valor de semilla.
from random import random, seed, randrange, randint, choice, sample

beg, end,step, left, right, sequence, elements_to_choose = 0, 10, 1, 2, 5, [1,2,3,4,5,6,7,8,9], 5
random()    # produce un número flotante x procedente del rango (0,0, 1,0)
seed()      # La función es capaz de establecer directamente la semilla del generador.
            # Te mostramos dos de sus variantes: seed() - establece la semilla con la hora actual;
            # seed(int_value): establece la semilla con el valor entero int_value.

for i in range(5):
    # seed(0) # con seed0 0 establecemos la semilla en 0 con lo que ya no es aleatorio, los numeros generados serán los
    # mismos
    print(random())

# valores aleatorios enteros, exclusión implícita del lado derecho es como un aleatorio del range
randrange(end)
randrange(beg, end)
randrange(beg, end, step)
randint(left, right)

# Las funciones anteriores tienen una desventaja importante: pueden producir valores repetitivos
# incluso si el número de invocaciones posteriores no es mayor que el ancho del rango especificado.
# Afortunadamente, hay una mejor solución que escribir su propio código para verificar la singularidad de los números
# "dibujados".


choice(sequence)  # La primera variante elige un elemento "aleatorio" de la secuencia de entrada y lo devuelve.
sample(sequence, elements_to_choose) # Elige algunos de los elementos de entrada, devolviendo una lista del tamaño
# indicado con la opción. Los elementos de la muestra se colocan en orden aleatorio
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(choice(my_list))
print(sample(my_list, 5))
print(sample(my_list, 10))


# ****** platform
# información del sistema operativo
# El módulo de plataforma le permite acceder a los datos de la plataforma subyacente, es decir, hardware, sistema
# operativo e información de versión del intérprete.
from platform import platform
platform(aliased = False, terse = False)
# aliased → cuando se establece en True (o cualquier valor distinto de cero) puede hacer que la función presente
# los nombres de capa subyacentes alternativos en lugar de los comunes;
# terse cuando se establece en True (o cualquier valor distinto de cero) puede convencer a la función
# ara que presente una forma más breve del resultado (si es posible)

from platform import platform, machine, processor, system, version, python_implementation, python_version_tuple

print(platform()) # macOS-13.0.1-arm64-arm-64bit
print(platform(0, 1)) # macOS-13.0.1
print(machine()) # arm64, el nombre genérico del procesador que ejecuta su sistema operativo
print(processor()) # , nombre real del procesador ( si es posible)
print(system()) # Darwin, devuelve el nombre genérico del sistema operativo como una cadena.
print(version()) # Darwin Kernel Version 22.1.0: Sun Oct  9 20:15:09 PDT 2022; root:xnu-8792.41.9~2/RELEASE_ARM64_T6000
print(python_implementation())  # CPython, devuelve una cadena que indica la implementación de Python
print(python_version_tuple())   # ('3', '11', '0'), la mayor parte de la versión de Python; la parte menor;
# El número de nivel de parche.
# La versión del sistema operativo se proporciona como una cadena

# ****** Os
# módulo del sistema operativo

# uname
''' permite obtener información del sistema operativo UNIX, en windows hay que usar uname de platform, info similar
systemname — almacena el nombre del sistema operativo;
nodename: almacena el nombre de la máquina en la red;
release: almacena la release del sistema operativo;
version: almacena la versión del sistema operativo;
machine: almacena el identificador de hardware, por ejemplo, x86_6'''
import os
print(os.uname())
'''posix.uname_result(
sysname='Darwin', 
nodename='MacBook-Pro-de-Jesus.local',
release='22.2.0', 
version='Darwin Kernel 
Version 22.2.0: Fri Nov 11 02:03:51 PST 2022; root:xnu-8792.61.2~4/RELEASE_ARM64_T6000', machine='arm64')
'''
# name
'''te da el nombre del sistema operativo
posix: obtendrá este nombre si usa Unix;
nt: obtendrá este nombre si usa Windows;
java: obtendrá este nombre si su código está escrito en Jython.
'''
import os
print(os.name)  # posix
# mkdir
'''le permite crear un directorio. La función mkdir requiere una ruta que puede ser relativa o absoluta.
my_first_directory - ruta relativa, crea la carpeta en la ruta actual.
./my_first_directory: ruta relativa, apunta explícitamente al directorio actual. mismo efecto que el caso anterior;
../my_first_directory — ruta relativa que creará  my_first_directory en el directorio principal del directorio actual;
/python/my_first_directory: esta es la ruta absoluta que creará el directorio my_first_directory, que a su vez se 
encuentra en el directorio python en el directorio raíz.
No podemos crear un directorio si ya existe (FileExistsError). Además del argumento de ruta, opcionalmente el argumento 
de modo, que especifica los permisos de directorio. Sin embargo, en algunos sistemas, el argumento de modo se ignora.
Para cambiar los permisos del directorio, recomendamos la función chmod, similar al comando chmod en los sistemas Unix.
listdir() devuelve una lista con los nombres de los archivos y directorios que se encuentran en la ruta pasada 
como argumento. Si no se le pasa ningún argumento, se usará el directorio de trabajo actual. Es importante que el 
resultado de la función listdir omita las entradas '.' y '..'.'''
import os
os.mkdir("my_first_directory")
print(os.listdir())

# makedirs
'''makedirs permite la creación recursiva de directorios, lo que significa que se crearán todos los directorios de la 
ruta. Para moverse entre directorios, puede usar una función llamada chdir, que cambia el directorio de trabajo actual a 
 la ruta especificada.'''
import os
os.makedirs("my_first_directory/my_second_directory")
os.chdir("my_first_directory")
print(os.listdir())

# getwcd
'''obtener el directorio completo de trabajo actual Se llama getcwd.'''
import os

os.makedirs("my_first_directory/my_second_directory")
os.chdir("my_first_directory")
print(os.getcwd())
os.chdir("my_second_directory")
print(os.getcwd())

'''
os.mkdir("my_first_directory")
print(os.listdir())
os.rmdir("my_first_directory") # eliminar directorio
os.removedirs("my_first_directory/my_second_directory")  # eliminar directorio y subdirectorios
print(os.listdir())
'''

# system
# Ejecuta el comando pasado como argumento.
import os

returned_value = os.system("mkdir my_first_directory")
print(returned_value)

# +++++ buscador de rutas absolutas ++++
'''Escriba una función o método llamado find que tome dos argumentos llamados path y dir. El argumento ruta debe 
aceptar una ruta relativa o absoluta a un directorio donde debe comenzar la búsqueda, mientras que el argumento dir 
debe ser el nombre de un directorio que desea encontrar en la ruta dada. Su programa debería mostrar las rutas 
absolutas si encuentra un directorio con el nombre dado. La búsqueda en el directorio debe hacerse recursivamente. Esto 
significa que la búsqueda también debe incluir todos los subdirectorios en la ruta dada.'''
import os

class DirectorySearcher:
    def find(self, path, dir):
        try:
            os.chdir(path)  # este intento de ir al directorio es lo que hace que corte , caso base cuando ya no sea un
            # directorio volverá gracias al return de abajo
        except OSError:
            # Doesn't process a file that isn't a directory.
            return

        current_dir = os.getcwd()
        for entry in os.listdir("."):
            if entry == dir:
                print(os.getcwd() + "/" + dir)
            self.find(current_dir + "/" + entry, dir)  # llamada recursiva


directory_searcher = DirectorySearcher()
directory_searcher.find('Cursos', "cursoscrap")

# ******* datetime
# date y today
# Los objetos de esta clase representan una fecha que consiste en el año, el mes y el día.
from datetime import date

today = date.today()  # today: objeto de fecha que representa la fecha local actual

print("Today:", today)  # 2022-12-23
print("Year:", today.year)  # 2022
print("Month:", today.month)  # 12
print("Day:", today.day)  # 23

my_date = date(2019, 11, 4)  # crear un objeto de tipo fecha necesitas pasar 3 parametros correctos de año, mes y día
print(my_date)  # 2019-11-04

'''
La clase de fecha nos brinda la capacidad de crear un objeto de fecha a partir de un timestamp, que es la diferencia 
entre una fecha determinada (incluida la hora) y el 1 de enero de 1970, 00:00:00 (UTC), expresada en segundos. Para 
crear un objeto de date a partir de un timestamp, debemos pasar un timestamp de Unix al método fromtimestamp. 
podemos usar el módulo time, que proporciona funciones relacionadas con el tiempo. Una de ellas es una función llamada 
time() que devuelve el número de segundos desde el 1 de enero de 1970 hasta el momento actual en forma de número 
flotante. en los sistemas Unix y Windows, los segundos bisiestos no se cuentan.
'''
from datetime import date
import time

timestamp = time.time()
print("Timestamp:", timestamp)  # Timestamp: 1671822545.616088
d = date.fromtimestamp(timestamp)
print("Date:", d)  # Date: 2022-12-23

'''El módulo de datetime proporciona varios métodos para crear un objeto de fecha. Uno de ellos es el método 
fromisoformat, que toma una fecha en el formato AAAA-MM-DD conforme a la norma ISO 8601.'''
from datetime import date
d = date.fromisoformat('2019-11-04')
print(d)  # 2019-11-04

# replace()
'''
No puedes cambiar los datos con los atributos de year, month y day porque son de solo lectura. En este caso, puede usar 
el método llamado replace().
'''
from datetime import date
d = date(1991, 2, 5)
print(d)
# opcionales los parámetros
d = d.replace(year=1992, month=1, day=16) # recordar asignar a una variable
print(d)

#  weekday , isoweekday
# devuelve el día de la semana de un date 0 es lunes y 6 es Domingo
from datetime import date
d = date(2022, 12, 23)
print(d.weekday())  # 4, viernes
d.isoweekday()  # devuelve el dia de la semana CON FORMATO ISO 1 LUNES 7 DOMINGO

# time
'''devuelve la hora time(hour, minute, second, microsecond, tzinfo, fold)
El parámetro tzinfo está asociado con zonas horarias, mientras que fold con tiempos de pared. No los usaremos durante 
este curso, pero lo alentamos a que se familiarice con ellos.'''
from datetime import time

t = time(14, 53, 20, 1)
print("Time:", t)  # Time: 14:53:20.000001
print("Hour:", t.hour)  # 14
print("Minute:", t.minute)  # 53
print("Second:", t.second)  # 20
print("Microsecond:", t.microsecond)  # 1

# sleep()
# suspende la ejecución del programa por el número dado de segundos, acepta sólo un número entero o de coma flotante
import time


class Student:
    def take_nap(self, seconds):
        print("I'm very tired. I have to take a nap. See you later.")
        time.sleep(seconds)
        print("I slept well! I feel great!")


student = Student()
student.take_nap(5)

# ctime
# convierte el tiempo en segundos desde el 1 de enero de 1970 a string
import time

timestamp = 1572879180
print(time.ctime(timestamp)) # Mon Nov  4 14:53:00 2019
print(time.ctime()) # current , Fri Dec 23 19:30:15 2022

# gmtime() y localtime()
''' Algunas de las funciones disponibles en el módulo de tiempo requieren conocimiento de la clase struct_time, pero 
antes de conocerlas, veamos cómo se ve la clase:
tiempo.struct_time:
     tm_year # especifica el año
     tm_mon # especifica el mes (valor de 1 a 12)
     tm_mday # especifica el día del mes (valor de 1 a 31)
     tm_hour # especifica la hora (valor de 0 a 23)
     tm_min # especifica el minuto (valor de 0 a 59)
     tm_sec # especifica el segundo (valor de 0 a 61)
     tm_wday # especifica el día de la semana (valor de 0 a 6)
     tm_yday # especifica el día del año (valor de 1 a 366)
     tm_isdst # especifica si se aplica el horario de verano (1: sí, 0: no, -1: no se sabe)
     tm_zone # especifica el nombre de la zona horaria (valor en forma abreviada)
     tm_gmtoff # especifica el desplazamiento al este de UTC (valor en segundos)

La clase struct_time también permite el acceso a valores mediante índices. El índice 0 devuelve el valor en tm_year, 
mientras que el 8 devuelve el valor en tm_isdst. Las excepciones son tm_zone y tm_gmoff, a las que no se puede acceder 
mediante índices. Veamos cómo usar la clase struct_time en la práctica. Ejecute el código en el editor.
'''
import time
timestamp = 1572879180
print(time.gmtime(timestamp))
print(time.localtime(timestamp))
'''
time.struct_time(tm_year=2019, tm_mon=11, tm_mday=4, tm_hour=14, tm_min=53, tm_sec=0, tm_wday=0, tm_yday=308, tm_isdst=0)
time.struct_time(tm_year=2019, tm_mon=11, tm_mday=4, tm_hour=14, tm_min=53, tm_sec=0, tm_wday=0, tm_yday=308, tm_isdst=0)
producción
El ejemplo muestra dos funciones que convierten el tiempo transcurrido desde la época de Unix al objeto struct_time. La 
diferencia entre ellos es que la función gmtime devuelve el objeto struct_time en UTC, mientras que la función localtime
devuelve la hora local. Para la función gmtime, el atributo tm_isdst siempre es 0.'''

# asctime() and mktime()
'''
modulo time tiene funciones que esperan un objeto struct_time o una tupla que almacena valores de acuerdo con los 
índices presentados anteriormente. 
'''
import time
timestamp = 1572879180
st = time.gmtime(timestamp)
print(time.asctime(st))  # Mon Nov  4 14:53:00 2019
print(time.mktime((2019, 11, 4, 14, 53, 0, 0, 308, 0)))  #1572879180.0
'''
La primera de las funciones, llamada asctime, convierte un objeto struct_time o una tupla en una cadena. Tenga en cuenta
que la función familiar gmtime se usa para obtener el objeto struct_time. Si no proporciona un argumento a la función 
asctime, se usará la hora devuelta por la función localtime. La segunda función llamada mktime convierte un objeto 
struct_time o una tupla que expresa la hora local en el número de segundos desde la época de Unix. En nuestro ejemplo, 
le pasamos una tupla, que consta de los siguientes valores:'''

# datetime
'''La clase que combina fecha y hora 
datetime(year, month, day, hour, minute, second, microsecond, tzinfo, fold)'''

from datetime import datetime
dt = datetime(2019, 11, 4, 14, 53) # Timestamp: 1572879180.0
print("Datetime:", dt)  # Datetime: 2019-11-04 14:53:00
print("Timestamp:", dt.timestamp()) #
print("Date:", dt.date())  # 2019-11-04
print("Time:", dt.time())  # 14:53:00
'''La clase datetime tiene varios métodos que devuelven la fecha y la hora actuales. Estos métodos son:
today() — devuelve la fecha y hora locales actuales con el atributo tzinfo establecido en Ninguno;
now() — devuelve la fecha y hora local actual igual que el método today, a menos que le pasemos el argumento opcional 
tz. El argumento de este método debe ser un objeto de la subclase tzinfo;
utcnow(): devuelve la fecha y hora UTC actual con el atributo tzinfo establecido en Ninguno.
'''

from datetime import datetime
print("today:", datetime.today())  # today: 2022-12-23 20:47:56.821865
print("now:", datetime.now())  # now: 2022-12-23 20:47:56.821882
print("utcnow:", datetime.utcnow())  # utcnow: 2022-12-23 19:47:56.821886


# Formato de fechas - strftime ( cadena )  strptime (objeto datetime)
'''Todas las clases de módulos de fecha y hora presentadas hasta ahora tienen un método llamado strftime. Este es un
 método muy importante, porque nos permite devolver la fecha y la hora en el formato que especifiquemos. toma solo un 
 argumento en forma de cadena que especifica el formato que puede consistir en directivas. Una directiva es una cadena 
 que consta del carácter % (porcentaje) y una letra minúscula o mayúscula, por ejemplo, la directiva %Y significa el año
 con el siglo como número decimal.
'''
from datetime import date
d = date(2020, 1, 4)
print(d.strftime('%Y/%m/%d'))  # 2020/01/04
'''
En el ejemplo, pasamos un formato que consta de tres directivas separadas por / (barra oblicua) al método strftime. Por 
supuesto, el carácter separador puede ser reemplazado por otro carácter, o incluso por una cadena. Puede poner cualquier
carácter en el formato, pero solo las directivas reconocibles se reemplazarán con los valores apropiados. En nuestro 
formato hemos usado las siguientes directivas:
%Y: devuelve el año con el siglo como número decimal. En nuestro ejemplo, esto es 2020.
%m: devuelve el mes como un número decimal con ceros. En nuestro ejemplo, es 01.
%d: devuelve el día como un número decimal con ceros. En nuestro ejemplo, es 04.

El formato de hora funciona de la misma manera que el formato de fecha, pero requiere el uso de directivas apropiadas. 
Echemos un vistazo más de cerca a algunos de ellos en el editor.
'''
from datetime import time
from datetime import datetime
t = time(14, 53)
print(t.strftime("%H:%M:%S"))  # 14:53:00
dt = datetime(2020, 11, 4, 14, 53)
print(dt.strftime("%y/%B/%d %H:%M:%S"))  # 20/November/04 14:53:00'
'''
%H devuelve la hora como un número decimal con ceros, %M devuelve el minuto como un número decimal con ceros, mientras 
que %S devuelve el segundo como un número decimal con ceros. En nuestro ejemplo, %H se reemplaza por 14, %M por 53 y 
%S por 00. El segundo formato utilizado combina directivas de fecha y hora. Hay dos directivas nuevas, %Y y %B. 
La directiva %Y devuelve el año sin un siglo como un número decimal con ceros (en nuestro ejemplo es 20). La directiva 
%B devuelve el mes como el nombre completo de la configuración regional (en nuestro ejemplo, es noviembre).

La función strftime está disponible en el módulo time. Se diferencia ligeramente de los métodos strftime en las clases 
proporcionadas por el módulo datetime porque, además del argumento de formato, también puede tomar (opcionalmente) un 
objeto tupla o struct_time. Si no pasa un objeto tuple o struct_time, el formato se realizará utilizando la hora local 
actual.'''
import time
timestamp = 1572879180
st = time.gmtime(timestamp)
print(time.strftime("%Y/%m/%d %H:%M:%S", st))
print(time.strftime("%Y/%m/%d %H:%M:%S"))

# strptime()
'''
Saber cómo crear un formato puede ser útil cuando se usa un método llamado strptime en la clase de fecha y hora. A 
diferencia del método strftime, crea un objeto de fecha y hora a partir de una cadena que representa una fecha y una 
hora. El método strptime requiere que especifique el formato en el que guardó la fecha y la hora. 
'''
from datetime import datetime
print(datetime.strptime("2019/11/04 14:53:00", "%Y/%m/%d %H:%M:%S"))  # 2019-11-04 14:53:00
'''
En el ejemplo, hemos especificado dos argumentos obligatorios. El primero es una fecha y hora como una cadena: 
"2019/11/04 14:53:00", mientras que el segundo es un formato que facilita el análisis de un objeto de fecha y hora. 
Tenga cuidado, porque si el formato que especifica no coincide con la fecha y la hora en la cadena, generará un 
ValueError. 
En el módulo time, puede encontrar una función llamada strptime, que analiza una cadena que representa un tiempo en un 
objeto struct_time. Su uso es análogo al método strptime en la clase datetime:

import time
print(time.strptime("2019/11/04 14:53:00", "%Y/%m/%d %H:%M:%S"))
'''

#  Operaciones de fecha y hora
'''
Tarde o temprano tendrás que realizar algunos cálculos sobre la fecha y la hora. Afortunadamente, hay una clase llamada 
timedelta en el módulo de fecha y hora que se creó con ese propósito. Para crear un objeto timedelta, solo reste los 
objetos date o datetime.
'''
from datetime import date
from datetime import datetime

d1 = date(2020, 11, 4)
d2 = date(2019, 11, 4)
print(d1 - d2)  # 366 days, 0:00:00
dt1 = datetime(2020, 11, 4, 0, 0, 0)
dt2 = datetime(2019, 11, 4, 14, 53, 0)
print(dt1 - dt2)  # 365 days, 9:07:00
'''
El ejemplo muestra la resta para los objetos de fecha y fecha y hora. En el primer caso, recibimos la diferencia en 
días, que son 366 días. Tenga en cuenta que también se muestra la diferencia en horas, minutos y segundos. En el 
segundo caso, recibimos un resultado diferente, porque especificamos el tiempo que se incluyó en los cálculos. Como 
resultado, recibimos 365 días, 9 horas y 7 minutos.

también puede crear un objeto usted mismo. los argumentos son: días, segundos, microsegundos, milisegundos, minutos, 
horas y semanas. Cada uno de ellos es opcional y su valor predeterminado es 0. Los argumentos deben ser números enteros 
o de punto flotante, y pueden ser positivos o negativos. 

'''
from datetime import timedelta
delta = timedelta(weeks=2, days=2, hours=3)
print(delta)  # 16 days, 3:00:00
'''
El resultado de 16 días se obtiene convirtiendo el argumento de semanas a días (2 semanas = 14 días) y sumando el 
argumento de días (2 días). Este es un comportamiento normal, porque el objeto timedelta solo almacena días, segundos y 
microsegundos internamente. De manera similar, el argumento de la hora se convierte en minutos. Eche un vistazo al 
'''
from datetime import timedelta
delta = timedelta(weeks=2, days=2, hours=3)
print("Days:", delta.days)  # Días: 16
print("Seconds:", delta.seconds)  # Segundos: 10800
print("Microseconds:", delta.microseconds)  # Microsegundos: 0
'''
El resultado de 10800 se obtiene convirtiendo 3 horas en segundos. De esta forma, el objeto timedelta almacena los 
argumentos pasados durante su creación. Las semanas se convierten en días, las horas y los minutos en segundos y los 
milisegundos en microsegundos.
'''
from datetime import timedelta
from datetime import date
from datetime import datetime
delta = timedelta(weeks=2, days=2, hours=2)
print(delta)  # 16 days, 2:00:00
delta2 = delta * 2 # se puede multiplicar por un entero
print(delta2)  # 32 days, 4:00:00
d = date(2019, 10, 4) + delta2 # sumarlo a un date
print(d)  # 2019-11-05
dt = datetime(2019, 10, 4, 14, 53) + delta2  # o a un datetime
print(dt)  # 2019-11-05 18:53:00


# ++++++ ejercicio tiempo
'''
Escriba un programa que cree un objeto de fecha y hora para el 4 de noviembre de 2020 a las 14:53:00. El objeto creado 
debe llamar al método strftime con el formato apropiado para mostrar el siguiente resultado:
'''
from datetime import datetime
my_date = datetime(2020, 11, 4, 14, 53)
print(my_date.strftime("%Y/%m/%d %H:%M:%S"))  # 2020/11/04 14:53:00
print(my_date.strftime("%y/%B/%d %H:%M:%S %p"))  # 20/November/04 14:53:00 PM (B mes letrea y p AM o PM)
print(my_date.strftime("%a, %Y %b %d"))  # Wed, 2020 04 de nov (a día de la semana abreviado, b mes abreviado)
print(my_date.strftime("%A, %Y %B %d"))  # Wednesday, 2020 November 04 ( A dia de la semana, B mes)
print(my_date.strftime("Weekday: %w"))  # Weekday: 3 (w, dia de la semana numero)
print(my_date.strftime("Day of the year: %j"))  # Day of the year: 309 (j numero del año)
print(my_date.strftime("Week number of the year: %W"))  # (W, número de semana del año)

# ****** Calendar
'''
Monday	0	calendar.MONDAY
Tuesday	1	calendar.TUESDAY
Wednesday	2	calendar.WEDNESDAY
Thursday	3	calendar.THURSDAY
Friday	4	calendar.FRIDAY
Saturday	5	calendar.SATURDAY
Sunday	6	calendar.SUNDAY
Para los meses, los valores enteros están indexados desde 1, es decir, enero está representado por 1 y diciembre por 
12. Desafortunadamente, no hay constantes que expresen los meses.
'''
# calendar
# permite moestrar el calendario de todo el año
import calendar
print(calendar.calendar(2022))
'''para cambiar el formato de calendario predeterminado, parámetros:
w – ancho de columna de fecha (predeterminado 2)
l – número de líneas por semana (predeterminado 1)
c – número de espacios entre las columnas del mes (predeterminado 6)
m – número de columnas (predeterminado 3)
La función de calendario requiere que especifique el año, mientras que los otros parámetros responsables del formato 
son opcionales. Le animamos a que pruebe estos parámetros usted mismo.
la función llamada prcal, que también toma los mismos parámetros que la función de calendario, pero no requiere el uso 
de la función de impresión para mostrar '''
import calendar
calendar.prcal(2020)

# month
'''le permite mostrar un calendario para un mes específico. Su uso es realmente simple, solo necesita especificar el
año y el mes, formato: w – ancho de columna de fecha (predeterminado 2),l – número de líneas por semana (predeterminado 
1)'''
import calendar
print(calendar.month(2020, 11))
calendar.prmonth(2020, 11)  # sin print

# setfirstweekday()
# cambia el primer dia de la semana (por defecto lunes)
import calendar
# El método requiere un parámetro que exprese el día de la semana en forma de un valor entero
calendar.setfirstweekday(calendar.SUNDAY) # puedes usar el 6
calendar.prmonth(2020, 12)
'''
   December 2020
Su Mo Tu We Th Fr Sa
       1  2  3  4  5
 6  7  8  9 10 11 12
13 14 15 16 17 18 19
20 21 22 23 24 25 26
27 28 29 30 31
'''

# weekday
# devuelve el día de la semana para ese año, mes y día
import calendar
print(calendar.weekday(2020, 12, 24))  # 3

#  weekheader()
'''contiene encabezados semanales en forma abreviada, requiere que especifique el ancho en caracteres para un día de 
la semana. si le pones un valor mayor a 3 pero que no quepan todas las letras seguirá ponendo 3 con mas o menos espacio
setfirstweekday le afecta'''
import calendar
print(calendar.weekheader(3))  # Mon Tue Wed Thu Fri Sat Sun
print(calendar.weekheader(10))  #   Monday    Tuesday   Wednesday   Thursday    Friday    Saturday    Sunday

#  años bisiestos
import calendar
print(calendar.isleap(2020)) # True o False si el año es bisiesto
print(calendar.leapdays(2010, 2020))  # 3, el numero de años bisiestos

# Clases para crear calendarios
'''
calendar.Calendar: proporciona métodos para preparar los datos del calendario para formatear;
calendar.TextCalendar: se utiliza para crear calendarios de texto regulares;
calendar.HTMLCalendar: se utiliza para crear calendarios HTML;
calendar.LocalTextCalendar: es una subclase de la clase calendar.TextCalendar. El constructor de esta clase toma el 
                            parámetro locale, que se utiliza para devolver los meses y los nombres de los días de la 
                            semana apropiados.
calendar.LocalHTMLCalendar: es una subclase de la clase calendar.HTMLCalendar. El constructor de esta clase toma el 
                            parámetro locale, que se utiliza para devolver los meses y los nombres de los días de la 
                            semana apropiados.

El constructor de la clase Calendario toma un parámetro opcional llamado primer día de la semana, por defecto igual a 0 
(lunes) hasta 6, podemos usar las constantes ya conocidas. El ejemplo de código usa el método de la clase Calendar 
denominado iterweekdays, que devuelve un iterador para los números de los días de la semana. El primer valor devuelto 
siempre es igual al valor de la propiedad firstweekday. Debido a que en nuestro ejemplo el primer valor devuelto es 6, 
significa que la semana comienza en domingo.
'''
import calendar
c = calendar.Calendar(calendar.SUNDAY)
for weekday in c.iterweekdays():
    print(weekday, end=" ")  # 6 0 1 2 3 4 5

'''itermonthdates(año, mes) devuelve un iterador Como resultado, se devuelven todos los días del mes y el año 
especificados, así como todos los días antes del comienzo del mes o del final del mes que son necesarios para obtener 
una semana completa., en objeto datetime.date'''

import calendar
c = calendar.Calendar()
for date in c.itermonthdates(2019, 11):  # el 1 de noviembre fue viernes, completa hasta lunes por eso tiene octubre
    print(date, end=" ")  # 2019-10-28 2019-10-29 2019-10-30 2019-10-31 2019-11-01 ...

'''
itermonthdays(año, mes)devuelve el iterador a los días de la semana representados por números. pone 0 como son días 
fuera del rango de meses especificado que se agregan para mantener la semana completa.
'''
import calendar
c = calendar.Calendar()
for date in c.itermonthdays(2019, 11):
    print(date, end=" ")  # 0 0 0 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 0

'''itermonthdays2 devuelve los días en forma de tuplas que consisten en un número de día del mes y un número de día de 
la semana'''
import calendar
c = calendar.Calendar()
for date in c.itermonthdays2(2019, 11):
    print(date, end=" ")  # (0, 0) (0, 1) (0, 2) (0, 3) (1, 4) (2, 5) (3, 6) (4, 0) (5, 1) (6, 2) (7, 3) (8, 4) ...

'''itermonthdays3 devuelve los días en forma de tuplas que consisten en números de un año, un mes y un día del mes.'''
import calendar
c = calendar.Calendar()
for date in c.itermonthdays3(2019, 11):
    print(date, end=" ")  # (2019, 10, 28) (2019, 10, 29) (2019, 10, 30) ...

'''itermonthdays4 devuelve días en forma de tuplas que consisten en un año, un mes, un día del mes y el número del día 
de la semana '''

import calendar
c = calendar.Calendar()
for date in c.itermonthdays4(2019, 11):
    print(date, end=" ")  # (2019, 10, 28, 0) (2019, 10, 29, 1) (2019, 10, 30, 2) ...

'''
El método monthdays2calendar() ( hay mas en la docu )
. Uno de ellos es el método monthdays2calendar, que toma el año y e
l mes y luego devuelve una lista de semanas en un mes específico. Cada semana es una tupla que consta de números de días
y números de días de la semana. Mira el código en el editor. Tenga en cuenta que los números de días fuera del mes están
 representados por 0, mientras que los números de días de la semana son un número de 0 a 6, donde 0 es lunes y 
 6 es domingo.'''

# +++++++++ Ejercicio fechas lab   ++++++++++

'''
amplíe la funcionalidad de claendar con un nuevo método llamado count_weekday_in_year, que toma un año y un día de la
semana como parámetros y luego devuelve el número de ocurrencias de un día de la semana específico en el año.
como:
- Cree una clase llamada MyCalendar que amplíe la clase Calendar;
- cree el método count_weekday_in_year con los parámetros de año y día de la semana. El parámetro del día de la semana
  debe tener un valor entre 0 y 6, donde 0 es lunes y 6 es domingo. El método debe devolver el número de días como un
  número entero;
- en su implementación, use el método monthdays2calendar de la clase Calendar.
'''

import calendar
class MyCalendar(calendar.Calendar):
    def count_weekday_in_year(self, year, weekday):
        current_month = 1
        number_of_days = 0
        while (current_month <= 12):
            for data in self.monthdays2calendar(year, current_month):
                if data[weekday][0] != 0:
                    number_of_days = number_of_days + 1

            current_month = current_month + 1
        return number_of_days

my_calendar = MyCalendar()
number_of_days = my_calendar.count_weekday_in_year(2019, calendar.MONDAY)

print(number_of_days)  # 52


# ***********************************
# ******** Paquetes #############
# ***********************************
'''
Un módulo es una especie de contenedor lleno de funciones Paquete: Agrupa sus módulos con un rol similar a una carpeta 
/ directorio en el mundo de los archivos. Si creamos un módulo (un archivo.py )aunque sea vacío y lo importamos en otro,
Aparece una nueva subcarpeta,  __pycache__. Dentro  Hay un archivo llamado module.cpython-xy.pyc donde x e y. son dígitos 
derivados de su versión de Python (por ejemplo, serán 3 y 8 si usa Python 3.8). El nombre del archivo es el mismo 
que el nombre de su módulo (módulo aquí). La parte después del primer punto dice qué implementación de Python ha creado 
el archivo (CPython aquí) y su número de versión. La última parte (pyc) proviene de las palabras Python y compilado.
El contenido es completamente ilegible para los humanos. Tiene que ser así, ya que el archivo está destinado solo para 
uso de Python. Cuando Python importa un módulo por primera vez, traduce su contenido en una forma algo compilada.
Cuando se importa un módulo, Python ejecuta implícitamente su contenido. Le da al módulo la oportunidad de inicializar 
algunos de sus aspectos internos.
Python recuerda los módulos importados y omite silenciosamente todas las importaciones posteriores. Cuando ejecuta un 
archivo directamente, su variable __name__ se establece en __main__; Cuando un archivo se importa 
como un módulo, su variable __name__ se establece en el nombre del archivo (excluyendo.py)
'''
if __name__ == "__main__":
    print("I prefer to be a module.")
else:
    print("I like to be a module.")
'''  
variables privadas: precediendo el nombre de la variable con _ (un guión bajo) o __ (dos guiones bajos),
pero recuerde, es solo una convención. Los usuarios de su módulo pueden obedecerlo o no.
la línea que comienza con #! indica al sistema operativo cómo ejecutar el contenido del archivo,Esta convención no 
tiene ningún efecto bajo MS Windows. ( para python es un comentario)

path: variable especial (en realidad una lista) que almacena todas las ubicaciones (carpetas/directorios) que se buscan 
para encontrar un módulo que ha sido solicitado por la instrucción de importación está en el módulo sys Python explora 
estas carpetas en el orden en que aparecen en la lista:
si el módulo no se puede encontrar en ninguno de estos directorios, la importación falla.
De lo contrario, se tendrá en cuenta la primera carpeta
que contenga un módulo con el nombre deseado (si alguna de las carpetas restantes contiene un módulo con ese nombre, 
se ignorará).
 '''
import sys

for p in sys.path:
    print(p)

"""
/Users/jesusgomezcanovas/Dropbox/guapo/programar/python/venv/bin/python /Users/jesusgomezcanovas/Library/CloudStorage/Dropbox/guapo/programar/python/Deiphone.py 
/Users/jesusgomezcanovas/Library/CloudStorage/Dropbox/guapo/programar/python
/Users/jesusgomezcanovas/Library/CloudStorage/Dropbox/guapo/programar/python
/Library/Frameworks/Python.framework/Versions/3.11/lib/python311.zip
/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11
/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/lib-dynload
/Users/jesusgomezcanovas/Library/CloudStorage/Dropbox/guapo/programar/python/venv/lib/python3.11/site-packages
"""
sys.path.append('..\\modules') # Añadimos El nombre relativo de la carpeta \\, para añadir al path, es un ejemplo

# La inicialización de un módulo se realiza mediante un código independiente (que no forma parte de ninguna función)
# ubicado dentro del archivo del módulo. Como un paquete no es un archivo, esta técnica es inútil para inicializar
# paquetes. En su lugar, debe usar un truco diferente: Python espera que haya un archivo con un nombre muy único dentro
# de la carpetadel paquete: __init__.py. El contenido del archivo se ejecuta cuando se importa cualquiera de los
# módulos del paquete. No desea ninguna inicialización especial, puede dejar el archivo vacío, pero no debe omitirlo.

# pregunta
# Algunos paquetes adicionales y necesarios se almacenan dentro del directorio D:\Python\Project\Modules.
# Escriba un código que garantice que Python recorre el directorio para encontrar todos los módulos solicitados.
import sys

# note the double backslashes! (windows)
sys.path.append("D:\\Python\\Project\\Modules")

'''
# The directory mentioned in the previous exercise contains a sub-tree of the following structure:
# abc
#  |__ def
#       |__ mymodule.py
# Assuming that D:\Python\Project\Modules has been successfully appended to the sys.path list,
# write an import directive letting you use all the mymodule entities.

# import abc.def.mymodule
# PyPI es el repositorio central de python, pip (pip install packages) la herramienta para usarlo, permite resolver 
# las depdendencias entre modulos.
# pip help, ayuda de pip,
# pip help, install ayuda especifica para instalar
# pip list, version de pip y de la herramienta
# pip show package_name, te da información sobre los paquetes INSTALADOS, ej pip show pip. dentro
# de el Requires: y Required-by, por convención te dice sus dependencias , qué paquetes son necesarios para utilizar
# correctamente el paquete (Requiere:) qué paquetes necesitan que el paquete se utilice correctamente (Requerido por:)
# pip search anystring , El anystring proporcionado busca en el directorio ( repo)  los nombres de todos los paquetes; 
# Las cadenas de resumen de todos los paquetes, no diferencia mayusculas de minusculas
# --user  a la hora de instalar un paquete con --user solo lo instalamos para el usuario ( no necesita privilegios)
# y sin el en el sistema, ej # pip install pygame (admin) y +  --user lo comentado
# -U update, versión específica pip install pygame==1.9.2
# pip uninstall package_name desistalar paquetes
'''

# ***********************************
# ******** CLASES #############
# ***********************************
'''
Una clase es como una plantilla de la cual podemos sacar objetos (instancias)
Una clase es un conjunto de objetos. Un objeto es un ser que pertenece a una clase. Un objeto es una encarnación de
los requisitos, rasgos y cualidades asignados a una clase específica.
Ejemplo clase -> Persona, instancias juan y carlos
Posee Atributos y Métodos
La clase recién definida se convierte en una herramienta que puede crear nuevos objetos. La herramienta tiene que ser 
utilizada explícitamente, bajo demanda.
La definición comienza con la clase de palabra clave. La palabra clave va seguida de un identificador que nombrará la 
clase. no lo confunda con el nombre del objeto, son dos cosas diferentes.
El acto de crear un objeto de la clase seleccionada también se denomina instanciación (ya que el objeto se convierte en
una instancia de la clase).
# Persona.py
'''
# ++++++ Ejemplo Persona +++++
class Person:
    pass  # Palabra reservada para poder crear la función o clase sin contenido
print(type(Person))


class Persona:  # this == self se puede usar ambos
    # init método inicializador, similar a un constructor, en python está oculto y se llama por el lenguaje
    # Permite agregar e inicializar atributos, es decir, inicializar correctamente su estado interno,
    # crear variables de instancia, instanciar cualquier otro objeto si se necesita su existencia, etc.
    # está obligado a tener el parámetro self (se establece automáticamente, como de costumbre);
    # puede (pero no es necesario) tener más parámetros además de uno mismo; si esto sucede, la forma en que se usa el
    # nombre de la clase para crear el objeto debe reflejar la definición de __init__;
    # no puede devolver un valor
    # no se puede invocar directamente ni desde el objeto ni desde dentro de la clase
    def __init__(self, nombre, apellido, edad):

        # Self: Es una referencia al objeto que se va a crear
        # | __x__ es un método dunder
        # *args si queremos pasar una tupla de elementos variable
        # **kwargs si queremos pasar diccionario

        # son diferentes variables, la que no utiliza self es un parámetro
        # La variable que utiliza self es atributo de nuestra clase
        # ATRIBUTOS (características) #############
        self.nombre = nombre
        self.apellido = apellido
        self.edad = edad

    # MÉTODOS #############
    def mostrar_detalle(self):  # en los métodos de instancia siempre vamos a meter la referencia self
        # El parámetro self se utiliza para obtener acceso a las variables de clase e instancia del objeto.
        # así como otros métodos de la clase
        # AL encontrarnos dentro de la clase nos referimos con self al atributo
        print(f'Persona: {self.nombre} {self.apellido}  que tiene {self.edad} años')


persona1 = Persona('Juan', 'Perez', 28)  # Estamos llamando al constructor (init) estamos creando una instancia de
# La clase persona
# creando un objeto. self está apuntando al objeto que se está creando en ese momento.
# Al ser python las variables dinámicas se crean automáticamente con el valor que le pasamos
persona2 = Persona('Pedro', 'Baños', 45)
persona2.mostrar_detalle()
persona1.mostrar_detalle()

# ******** MODIFICAR ATRIBUTO DE LA CLASE

# (no es recomendable, mejor por métodos por encapsulamiento)
persona1.nombre = 'Jesus'
persona1.apellido = 'Gomez'
persona1.edad = 32
persona1.mostrar_detalle()  # Lo común

# Podemos llamar al método de la clase referenciando al objeto
Persona.mostrar_detalle(persona1)

# ******** Añadir atributos
# Ventaja de python y POO podemos añadir atributos al objeto en cualquier momento
# No se van a compartir con el resto de objetos
persona1.telefono = '968888888'
print(persona1.telefono)

# Metodos "ocultos"
class Classy:
    def visible(self):
        print("visible")

    def __hidden(self):
        print("hidden")


obj = Classy()
obj.visible() # visible

try:
    obj.__hidden()
except:
    print("failed") # failed

obj._Classy__hidden()  # hidden

# un método cuyo nombre comienza con __ está (parcialmente) oculto, podremos acceder con obj._Classy__hidden().

# ++++++ Ejericio POO Aritmética ++++++++++
class Aritmetica:
    # Docstring: documentación para nuestra clase
    """
    Clase Aritmetica para realizar las operaciones de sumar, restar, etc
    """

    def __init__(self, operando1, operando2):
        self.operando1 = operando1
        self.operando2 = operando2

    def sumar(self):
        return self.operando1 + self.operando2

    def restar(self):
        return self.operando1 - self.operando2

    def multiplicar(self):
        return self.operando1 * self.operando2

    def dividir(self):
        return self.operando1 / self.operando2


aritmetica1 = Aritmetica(2, 3)
print(aritmetica1.sumar())
print(aritmetica1.restar())
print(aritmetica1.multiplicar())
print(f'división {aritmetica1.dividir():.2f}')  # .2f indicar cuantos decimales mostrar


# ++++++ Ejericio POO rectangulo ++++++++

class Rectangulo:

    def __init__(self):
        self.base = int(input('introduce la base: '))
        self.altura = int(input('introduce la altura: '))

    def calcula_area(self):
        return self.base * self.altura

    def imprime_area(self):
        print(f'Área rectángulo: {self.calcula_area()}')


rectangulo1 = Rectangulo()
rectangulo1.imprime_area()


# ++++++ Ejercicio POO cubo ++++++

class Cubo:

    def __init__(self, ancho, profundo, alto):
        self.ancho = ancho
        self.profundo = profundo
        self.alto = alto

    def volumen(self):
        return self.alto * self.profundo * self.alto


ancho = int(input('Introduce el ancho: '))
alto = int(input('Introduce el ancho: '))
profundo = int(input('Introduce el ancho: '))

cubo1 = Cubo(ancho, profundo, alto)
print(f'El volumen es: {cubo1.volumen()}')


# +++++ Ejercicio tiempo +++++++

#def convert(x: int) -> str:
#    x = str(x)
#    if len(x) == 1:
#        x = '0' + x
#    return x


class Timer:
    def __init__( self, hours=0, minutes=0, seconds=0 ):
        self.__hours = hours
        self.__minutes = minutes
        self.__seconds = seconds

    def __str__(self):
        #return convert(self.__hours) + ':' + convert(self.__minutes) + ':' + convert(self.__seconds)
        # 02d formatea un número entero (d) en un campo de ancho mínimo 2 (2), con relleno de cero a la izquierda (0 inicial):
        return f'{self.__hours:02d}:{self.__minutes:02d}:{self.__seconds:02d}'
    def next_second(self):
        self.__seconds += 1
        if self.__seconds == 60:
            self.__seconds = 0
            self.__minutes += 1
        if self.__minutes == 60:
            self.__minutes = 0
            self.__hours += 1
        if self.__hours == 24:
            self.__hours = 0

    def prev_second(self):
        if self.__seconds == 0:
            self.__seconds = 60
            self.__seconds -= 1
        if self.__minutes == 0:
            self.__minutes = 60
            self.__minutes -= 1
        if self.__hours == 0:
            self.__hours = 24
            self.__hours -=1

timer = Timer(23, 59, 59)
print(timer)
timer.next_second()
print(timer)
timer.prev_second()
print(timer)

# ++++++ Ejercicio dias de la semana ++++
class WeekDayError(Exception):
    def __init__(self):
        Exception.__init__(self)
        self.message = "Sorry, I can't serve your request."


class Weeker:
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def __init__(self, day):
        if day not in Weeker.days:
            raise WeekDayError
        self.day = day

    def __str__(self):
        return self.day

    def add_days(self, n):
        current = Weeker.days.index(self.day)
        current = (current + n) % 7
        self.day = Weeker.days[current]

    def subtract_days(self, n):
        current = Weeker.days.index(self.day)
        current = current - n if n <= current else 7 - ((n - current) % 7)
        self.day = Weeker.days[current]


try:
    weekday = Weeker('Mon')
    print(weekday)
    weekday.add_days(15)
    print(weekday)
    weekday.subtract_days(23)
    print(weekday)
    weekday = Weeker('Monday')
except WeekDayError as e:
    print(e.message)

# ++++++ Ejercicio puntos cartesianos y triangulo por distancia +++++
import math


class Point:
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.__x = x
        self.__y = y

    def getx(self):
        return self.__x

    def gety(self):
        return self.__y

    def distance_from_xy(self, x, y):
        a = self.__x - x
        b = self.__y - y
        return math.hypot(a, b)

    def distance_from_point(self, point):
        x = point.getx()
        y = point.gety()
        return self.distance_from_xy(x, y)


point1 = Point(0, 0)
point2 = Point(1, 1)
print(point1.distance_from_point(point2))
print(point2.distance_from_xy(2, 0))

class Triangle:
    def __init__(self, vertice1, vertice2, vertice3):
        self.__vertice1 = vertice1.distance_from_point(vertice2)
        self.__vertice2 = vertice2.distance_from_point(vertice3)
        self.__vertice3 = vertice3.distance_from_point(vertice1)

    def perimeter(self):
        return self.__vertice1 + self.__vertice2 + self.__vertice3


triangle = Triangle(Point(0, 0), Point(1, 0), Point(0, 1))
print(triangle.perimeter())

# ***********************************
# ******** ENCAPSULAMIENTO, GET y SET y DESTRUCTORES ##########
# ***********************************
#   Con _ indicamos que sólo desde la propia clase podemos acceder a la clase
#   Aunque no deberíamos el lenguaje si te deja, es una sugerencia
#   Con __ si que omite la modificación del valor (no falla) es menos comun y falla al sacarlo
# ************************
# ******** ROBUSTECER METODO INIT

class Persona:

    def __init__(self, nombre, apellido, edad, *valores, **terminos):
        # *args si queremos pasar una tupla de elementos variable
        # **kwargs si queremos pasar diccionario

        self.__nombre = nombre
        self._apellido = apellido
        self._edad = edad
        self.valores = valores
        self.terminos = terminos

    # ********  MÉTODOS
    def mostrar_detalle(self):  # en los métodos de instancia siempre vamos a meter la referencia self
        # AL encontrarnos dentro de la clase nos referimos con self al atributo
        print(
            f'Persona: {self.__nombre} {self.apellido}  que tiene {self.edad} años,  con {self.valores} y diccionario {self.terminos}')

    # ********  SOBREESCRITURA
    def __str__(self):  # estamos sobrescribiendo la clase str de la clase padre

        return f'Persona: {self.__nombre} {self.apellido}  que tiene {self.edad} años'

    # get nos permite recuperar el valor y set modificarlos
    @property  # es un decorador, encapsula el atributo y lo hace accesible solo desde el método
    # entonces se hace accesible como un atributo (no tenemos que poner ())
    def nombre(self):
        return self.__nombre

    @nombre.setter  # con este decorador debemos indicar el nombre del atributo sin _
    # y .setter porque va a modificar el valor seguimos llamándolo si paréntesis
    # si lo comentamos no podemos modificar el valor si es con __ o no debemos si es con _
    def nombre(self, nombre):
        self.__nombre = nombre

    @property
    def apellido(self):
        return self._apellido

    @apellido.setter
    def apellido(self, apellido):
        self._apellido = apellido

    @property
    def edad(self):
        return self._edad

    @edad.setter
    def edad(self, edad):
        self._edad = edad

    def __del__(self):  # es una clase donder, procedente de object,
        # todas las clases en Phyton heredan de object
        print('LLamada al destructor'.center(30, '-'))
        print(f'Persona: {self.__nombre} {self._apellido}')
        # metemos un identificador de que estamos eliminando


# ********  LLamada pasandole tupla y diccionario
# Despues de los 3 valores principales, cualquier valor sera de la tupla
# hasta que le metamos clave=valor
# OJO PRIMERO LAS TUPLAS Y LUEGO LOS DICCIONARIOS
persona3 = Persona('Jose', 'Lopez', 30, 1, 2, 3, 'lacasitos', m='manzana', p='puerro')
persona3.mostrar_detalle()
persona2 = Persona('Pau', 'Gasol', 40, )  # LAS TUPLAS Y DICCIONARIOS SON OPCIONALES
persona2.mostrar_detalle()  # método
print(persona2)  # te imprime lo definido en el str, print(a1.__str__()) igual
persona2.nombre = 'pei'
persona2.edad = 25
print(f'Persona: {persona2.nombre} {persona2.apellido} , con edad {persona2.edad}')
del persona2  # Llamamos al método dunder y hará

# Si lo creamos en un archivo podemos acceder a este módulo desde otro archivo
# si hemos llamado el archivo Clases.py y queremos importar la clase persona
# ponemos: from Clases import Persona | si queremos todas: from Clases import *

# __name__ es una propiedad que indica nombre del módulo
# si lo ejecutamos desde el propio archivo donde aparece pondrá main
# si no el nombre dle archivo
print(__name__)

if __name__ == '__main__':
    print('Me ejecuto solo si estoy dentro del modulo donde lo defino')
    # esto sirve para códigos de prueba dentro del módulo

# ********  DESTRUCTORES
print('Creación de objetos'.center(50, '-'))  # con .center(50, '-') centramos
# lo que imprimimos metiendo - hasta 50 caracteres
persona1 = Persona('Fran', 'Villa', 33)
persona1.mostrar_detalle()

print('Eliminación objetos'.center(50, '-'))
del persona1  # eliminción explicita


# Es raro en python por la existencia del concepto del recolector de basura
# Esto es porque todos los objetos que no estén apuntados por una variable se van
# a destruir de forma automática y al finalizar el programa igual

# ***********************************
# ******** HERENCIA ##########
# ***********************************
'''
Todas las clases heredan de object, La jerarquía crece de arriba a abajo, como las raíces de los árboles, no las ramas
Cualquier objeto enlazado a un nivel específico de una jerarquía de clases hereda todos los rasgos (así como los 
requisitos y cualidades) definidos dentro de cualquiera de las superclases. La clase padre del objeto puede definir
nuevos rasgos (así como requisitos y cualidades) que serán heredados por cualquiera de sus subclases.
issubclass(ClassOne, ClassTwo) La función devuelve True si ClassOne es una subclase de ClassTwo y False en caso 
contrario, IMPORTANTE, cada clase se considera una subclase de sí misma.

Como ya sabes, un objeto es una encarnación de una clase. Esto significa que el objeto es como un pastel horneado 
con una receta que se incluye dentro de la clase. puede ser crucial si el objeto tiene (o no tiene) ciertas 
características. En otras palabras, si es un objeto de cierta clase o no.

Tal hecho podría ser detectado por la función llamada isinstance(): isinstance(nombreObjeto, NombreClase) que 
devuelve True si el objeto es una instancia de la clase, o False en caso contrario. Ser una instancia 
de una clase significa que el objeto (el pastel) ha sido preparado utilizando una receta contenida en la clase o en una 
de sus superclases.

'''
class Empleado(Persona):  # con (Padre) indicamos en la declaración que heredamos
    def __init__(self, nombre, apellido, edad, sueldo):
        # tenemos que inicializar los atributos del padre
        super().__init__(nombre, apellido, edad) # no necesitamos saber el nombre ni hacer referencia a self
        # super metodo que nos permite acceder a los atributos del padre
        # con super().__init__(atributos padre) estamos inicializando los atr padre
        # Persona.__init__(self, nombre, apellido, edad)  # sería lo mismo
        self.sueldo = sueldo

    # ********  SOBREESCRITURA
    def __str__(self):  # estamos sobrescribiendo la clase str de la clase padre
        # no tenemos visibilidad sin sobrescribir sobre sueldo porque por defecto
        # estaríamos usando el __str__ de Persona por lo que sobreescribimos
        # con super podemos acceder al atributo o método de la clase padre
        return f'Empleado: {super().__str__()}, y con sueldo {self.sueldo}'


empleado1 = Empleado('Juan', 'garcia', 23, 5000)
print(empleado1.nombre)
print(empleado1.sueldo)
print(empleado1)
print(issubclass(Empleado, Persona))  # True porque empleado es subclase de persona
print(isinstance(empleado1, Persona))  # True porque el objeto empleado1 es instancia de la clase Persona


'''
objeto_uno is  objeto_dos 

El operador is verifica si dos variables (objeto_uno y objeto_dos aquí) se refieren al mismo objeto.No olvide que las 
variables no almacenan los objetos en sí, sino solo los identificadores que apuntan a la memoria interna de Python. 
Asignar un valor de una variable de objeto a otra variable no copia el objeto, sino solo su identificador. Es por esto 
que un operador como is puede ser muy útil en circunstancias particulares.

'''


class SampleClass:
    def __init__(self, val):
        self.val = val


object_1 = SampleClass(0)
object_2 = SampleClass(2)
object_3 = object_1
object_3.val += 1

print(object_1 is object_2)  # False
print(object_2 is object_3)  # False
print(object_3 is object_1)  # True
print(object_1.val, object_2.val, object_3.val)  # 1 2 1

string_1 = "Mary had a little "
string_2 = "Mary had a little lamb"
string_1 += "lamb"

print(string_1 == string_2, string_1 is string_2)  # Ture False
'''
Los resultados prueban que object_1 y object_3 son en realidad los mismos objetos, mientras que string_1 y string_2 
no lo son, a pesar de que su contenido es el mismo.
'''

# +++++++ Ejercicio Herencia en Python ++++++
# Definir una clase padre llamada Vehículo y dos clases hijas llamadas Coche y
# Bicicleta, las cuales heredan de la clase Padre Vehíuculo

class Vehiculo:
    def __init__(self, color, ruedas):
        self._color = color
        self._ruedas = ruedas

    @property
    def color(self):
        return self._color

    @property
    def ruedas(self):
        return self._ruedas

    @color.setter
    def color(self, color):
        self._color = color

    @ruedas.setter
    def ruedas(self, ruedas):
        self._ruedas = ruedas

    def __str__(self):
        return f'Vehiculo de {self.color} y con {self.ruedas}'


class Coche(Vehiculo):
    def __init__(self, color, velocidad):
        super().__init__(color, 4) # en curso cert no habla de super()
        self._velocidad = str(velocidad) + 'km/hr'

    def __str__(self):
        return f'Coche: {super().__str__()} con velocidad {self.velocidad}'

    @property
    def velocidad(self):
        return self._velocidad

    @velocidad.setter
    def velocidad(self, velocidad):
        self._velocidad = velocidad


class Bicicleta(Vehiculo):
    def __init__(self, color, tipo):
        super().__init__(color, 2)
        self._tipo = tipo

    def __str__(self):
        return f'Bicicleta: {super().__str__()} de tipo {self._tipo}'

    @property
    def tipo(self):
        return self._tipo

    @tipo.setter
    def tipo(self, tipo):
        self.tipo = tipo


coche1 = Coche('Rojo', 45)
print(coche1)

bicicleta1 = Bicicleta('blanca', 'montaña')
print(bicicleta1)

# ++++++ Ejemplo pila en versión procedimental VS por clases ++++++

stack = []

def push(val):
    stack.append(val)

def pop():
    val = stack[-1]
    del stack[-1]
    return val

push(3)
push(2)
push(1)

print(pop())  # 1
print(pop())  # 2
print(pop())  # 3

# +++++ Parte clases

class Stack:
    def __init__(self):
        self.__stack_list = []

    def push(self, val):
        self.__stack_list.append(val)

    def pop(self):
        val = self.__stack_list[-1]
        del self.__stack_list[-1]
        return val

stack_object = Stack()
stack_object.push(3)
stack_object.push(2)
stack_object.push(1)
print(stack_object.pop())
print(stack_object.pop())
print(stack_object.pop())

'''hemos utilizado la notación punteada, al igual que cuando se invocan métodos; Esta es la convención general para 
acceder a las propiedades de un objeto: debe nombrar el objeto, poner un punto (.) después de él y especificar el 
nombre de la propiedad deseada; ¡No uses paréntesis! No desea invocar un método, desea acceder a una propiedad'''


class AddingStack(Stack):
    # Python te obliga a invocar explícitamente el constructor de una superclase
    def __init__(self):
        Stack.__init__(self)  # se recomienda invocar el constructor de la superclase antes de cualquier otra
        # inicialización que desee realizar dentro de la subclase
        self.__sum = 0

class AddingStack(Stack):
    def __init__(self):
        Stack.__init__(self)
        self.__sum = 0

    def get_sum(self):
        return self.__sum

    def push(self, val):
        self.__sum += val
        Stack.push(self, val)

    def pop(self):
        val = Stack.pop(self)
        self.__sum -= val
        return val


stack_object = AddingStack()

for i in range(5):
    stack_object.push(i)
print(stack_object.get_sum())

for i in range(5):
    print(stack_object.pop())


class CountingStack(Stack):
    def __init__(self):
        Stack.__init__(self)
        self.__counter = 0

    def get_counter(self):
        return self.__counter

    def pop(self):
        self.__counter += 1
        Stack.pop(self)


stk = CountingStack()
for i in range(100):
    stk.push(i)
    stk.pop()
print(stk.get_counter())

class QueueError(Exception):
    def __init__(self,mensaje):
        self.messaje = mensaje


class Queue:
    def __init__(self):
        self.__cola = []

    def put(self, elem):
        self.__cola.append(elem)

    def get(self):
        try:
            re = self.__cola[0]
            del self.__cola[0]
            return re
        except QueueError as e:
            print(e)

# versión curso
    def __init__(self):
        self.queue = []

    def put(self, elem):
        self.queue.insert(0, elem)

    def get(self):
        if len(self.queue) > 0:
            elem = self.queue[-1]
            del self.queue[-1]
            return elem

que = Queue()
que.put(1)
que.put("dog")
que.put(False)
try:
    for i in range(4):
        print(que.get())
except:
    print("Queue error") #1 dog False Queue error


class SuperQueue(Queue):
    def __init__(self):
        Queue.__init__(self)

    def isempty(self):
        st = True
        if len(self.queue) > 0:
            st = False
        return st


que = SuperQueue()
que.put(1)
que.put("dog")
que.put(False)
for i in range(4):
    if not que.isempty():
        print(que.get())
    else:
        print("Queue empty")  # 1 dog False Queue empty

# ***********************************
# ******** HERENCIA MULTIPLE y ABSTRACTA ##########
# ***********************************
# ABSTRACTA no se pueden crear instancias de ella (figura = FiguraGeometrica() )
# Obliga a las clases hijas a realizar una implementación
# ABC = Abstract Base clase base para convertir una clase en abstracta
from abc import ABC, abstractmethod


class FiguraGeometrica(ABC):  # al extender de ABC es abstracta
    def __init__(self, ancho, alto):
        # añadimos comprobación de entrada con valor numérico positivo
        if self.__validar_valor(ancho):
            self._ancho = ancho
        else:
            print(f'Valor erróneo ancho: {ancho}')
        if self.__validar_valor(alto):
            self._alto = alto
        else:
            print(f'Valor erróneo alto,: {ancho}')

    def __str__(self):
        return f'Figura Geométrica con alto: {self._alto} y ancho {self._ancho}'

    def __validar_valor(self, valor):
        return True if 0 < valor < 10 else False

    @property
    def ancho(self):
        return self._ancho

    @property
    def alto(self):
        return self._alto

    @ancho.setter
    def ancho(self, ancho):
        if self.__validar_valor(ancho):
            self._ancho = ancho
        else:
            self._ancho = 0
            print(f'Valor erróneo para ancho, no se modifica: {ancho}')

    @alto.setter
    def alto(self, alto):
        if self.__validar_valor(alto):
            self._alto = alto
        else:
            self._ancho = 0
        print(f'Valor erróneo para alto, no se modifica: {alto}')

    # ********   METODO ABSTRACTO
    @abstractmethod
    def area(self):
        pass  # sin implementación


class Color:
    def __init__(self, color):
        self._color = color

    def __str__(self):
        return f'Color: {self._color} '

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color


# El orden en el que se indican las clases que se heredan es importante
class Cuadrado(FiguraGeometrica, Color):
    def __init__(self, lado, color):
        # la primera clase que encuentra es la que va a llamar
        # si pasamos dos parámetros se llamara a la FiguraGeometrica y si pasamos 1 a Color
        # esta forma, confusion
        # super().__init__(lado, lado)
        FiguraGeometrica.__init__(self, lado, lado)
        Color.__init__(self, color)
        # con self estamos pasando la referencia de la clase hija de la clase de cuadrado por usar el attr

    def __str__(self):
        return f'{FiguraGeometrica.__str__(self)}, con lados iguales, color {Color.__str__(self)} y área {self.area()}'

    def area(self):
        return self.ancho * self.alto
        # Se heredan de padre y podemos acceder directamente con self a los de la clase padre


class Rectangulo(FiguraGeometrica, Color):

    def __init__(self, ancho, alto, color):
        FiguraGeometrica.__init__(self, ancho, alto)
        Color.__init__(self, color)

    def __str__(self):
        return f'{FiguraGeometrica.__str__(self)}, tiene color {Color.__str__(self)} y área {self.area()}'

    def area(self):
        return self.ancho * self.alto


print('Creación Objeto Cuadrado'.center(2, '-'))
cuadrado1 = Cuadrado(1, 'rojo')
cuadrado1.ancho = -5  # si no queremos que se modifique quitamos el setter,
#                       se convierten en lectura y no se puede
print('Creación Objeto Rectangulo'.center(5, '-'))
rectangulo1 = Rectangulo(4, 5, 'azul')
# Rectangulo.alto = -50
print(cuadrado1.area())
print(cuadrado1)  # Llama a str
print(rectangulo1)

# MRO - Method resolution Order, para ver en que orden de resolución en que se van a ejecutar
# Si cambiamos el orden en que se hereda cambiaría. Nos indica el orden en que ira buscando
# Los métodos por ejemplo
print(Cuadrado.mro())


# ***********************************
# ******** Variable de clase, Métodos estáticos y método de clase ##########
# ***********************************

'''
variables de instancia:
Este tipo de propiedad de clase existe cuando y solo cuando se crea y agrega explícitamente a un objeto. 
Esto se puede hacer durante la inicialización del objeto, realizada por el constructor. Tal enfoque tiene algunas 
consecuencias importantes:
- diferentes objetos de la misma clase pueden poseer diferentes conjuntos de propiedades;
- debe haber una manera de verificar de manera segura si un objeto específico posee la propiedad que desea utilizar 
- cada objeto tiene su propio conjunto de propiedades: no interfieren entre sí de ninguna manera.
- Las variables de instancia están perfectamente aisladas entre sí.
'''
class ExampleClass:
    def __init__(self, val = 1):
        self.first = val

    def set_second(self, val):
        self.second = val


example_object_1 = ExampleClass()
example_object_2 = ExampleClass(2)
example_object_2.set_second(3)

example_object_3 = ExampleClass(4)
example_object_3.third = 5  # propiedad al vuelo, permitido

print(example_object_1.__dict__)  # {'first': 1}
print(example_object_2.__dict__)  # {'first': 2, 'second': 3}
print(example_object_3.__dict__)  # {'first': 4, 'third': 5}

'''Los objetos de Python, cuando se crean, están dotados de un pequeño conjunto de propiedades y métodos predefinidos. 
Cada objeto los tiene, los quieras o no. Uno de ellos es una variable llamada __dict__ (es un diccionario). La variable
contiene los nombres y valores de todas las propiedades (variables) que el objeto lleva actualmente.'''

class ExampleClass:
    counter = 0
    def __init__(self, val = 1):
        self.__first = val
        ExampleClass.counter += 1


    def set_second(self, val = 2):
        self.__second = val


example_object_1 = ExampleClass()
example_object_2 = ExampleClass(2)

example_object_2.set_second(3)

example_object_3 = ExampleClass(4)
example_object_3.__third = 5

print(example_object_1.__dict__)  # {'_ExampleClass__first': 1}
print(example_object_2.__dict__)  # {'_ExampleClass__first': 2, '_ExampleClass__second': 3}
print(example_object_3.__dict__)  # {'_ExampleClass__first': 4, '__third': 5}

'''Cuando Python ve que desea agregar una variable de instancia a un objeto y lo va a hacer dentro de cualquiera de los 
métodos del objeto, altera la operación de la siguiente manera:
- pone un nombre de clase antes de su nombre;
- pone un guión bajo adicional al principio.
Es por eso que __first se convierte en _ExampleClass__first. 
'''
print(example_object_1._ExampleClass__first)  # 1
'''
y obtendrá un resultado válido sin errores ni excepciones. Como puede ver, hacer que una propiedad sea privada es 
limitado. La manipulación no funcionará si agrega una variable de instancia privada fuera del código de clase. 
En este caso, se comportará como cualquier otra propiedad ordinaria.'''
class ExampleClass:
    __counter = 0
    def __init__(self, val = 1):
        self.__first = val
        ExampleClass.__counter += 1


example_object_1 = ExampleClass()
example_object_2 = ExampleClass(2)
example_object_3 = ExampleClass(4)

print(example_object_1.__dict__, example_object_1._ExampleClass__counter)  # {'_ExampleClass__first': 1} 3
print(example_object_2.__dict__, example_object_2._ExampleClass__counter)  # {'_ExampleClass__first': 2} 3
print(example_object_3.__dict__, example_object_3._ExampleClass__counter)  # {'_ExampleClass__first': 4} 3

''' las variables de clase existen incluso cuando no se ha creado ninguna instancia de clase (objeto).
Veamos la diferencia entre estas dos variables __dict__, la de la clase y la del objeto.
Definimos una clase llamada ExampleClass;
La clase define una variable de clase denominada varia;
El constructor de la clase establece la variable con el valor del parámetro;
Nombrar la variable es el aspecto más importante del ejemplo porque:
Cambiar la asignación a self.varia = val crearía una variable de instancia con el mismo nombre que la de la clase;
Cambiar la asignación a varia = val operaría en la variable local de un método; 
La primera línea del código fuera de clase imprime el valor del atributo ExampleClass.varia; nota: usamos el valor 
antes de que se instancia el primer objeto de la clase.'''

class ExampleClass:
    varia = 1
    def __init__(self, val):
        ExampleClass.varia = val


print(ExampleClass.__dict__)
# {'__module__': '__main__', 'varia': 1, '__init__': <function ExampleClass.__init__ at 0x000001F72AA8A160>, '__dict__': <attribute '__dict__' of 'ExampleClass' objects>, '__weakref__': <attribute '__weakref__' of 'ExampleClass' objects>, '__doc__': None}

example_object = ExampleClass(2)

print(ExampleClass.__dict__)
# {'__module__': '__main__', 'varia': 2, '__init__': <function ExampleClass.__init__ at 0x000001F72AA8A160>, '__dict__': <attribute '__dict__' of 'ExampleClass' objects>, '__weakref__': <attribute '__weakref__' of 'ExampleClass' objects>, '__doc__': None}

print(example_object.__dict__)  # {}

'''Como puede ver, el __dict__ de la clase contiene muchos más datos que la contraparte de su objeto. La mayoría 
de ellos son inútiles ahora: el que queremos que revise cuidadosamente muestra el valor actual de varia.
Tenga en cuenta que el __dict__ del objeto está vacío: el objeto no tiene variables de instancia.

La actitud de Python hacia la creación de instancias de objetos plantea un problema importante: a diferencia de otros 
lenguajes de programación, no puede esperar que todos los objetos de la misma clase tengan los mismos conjuntos 
de propiedades.

El objeto creado por el constructor solo puede tener uno de dos atributos posibles: a o b.'''
class ExampleClass:
    def __init__(self, val):
        if val % 2 != 0:
            self.a = 1
        else:
            self.b = 1
example_object = ExampleClass(1)
print(example_object.a)
# print(example_object.b)
#Al ejecutarlo:... print(example_object.b) \n AttributeError: 'ExampleClass' object has no attribute 'b
#Python proporciona una función que puede verificar de manera segura si algún objeto/clase contiene una propiedad
#específica,  hasattr que recibe la clase o objeto y el nombre de la propiedad a comprobar devuelve True o False
if hasattr(example_object, 'b'):
    print(example_object.b)

class ExampleClass:
    a = 1
    def __init__(self):
        self.b = 2


example_object = ExampleClass()

print(hasattr(example_object, 'b'))  # True
print(hasattr(example_object, 'a'))  # True
print(hasattr(ExampleClass, 'b'))  # False
print(hasattr(ExampleClass, 'a'))  # True

# Los atributos son independientes, corresponden a cada instancia
# las variables de clase se comparten porque se asocian con la clase en si misma y se comparte con todos los objetos.
# Las variables de clase son una propiedad que existe en una sola copia y se almacena fuera de cualquier objeto. NO se
# muestran en el __dict__ de un objeto ( no hablamos del de la clase) y siempre presenta el mismo valor en todas las
# instancias de clase (objetos). Esto es porque la clase se carga en memoria al pasar la parte del programa
# cuando creamos un objeto se carga en memoria la variable instancia y podemos acceder
# todos los objetos pueden acceder a la variable de clase
# El contexto dinámico si puede acceder al estático pero no al revés



class MiClase:
    # si definimos una variable fuera del método init se crea una variable de clase
    # los atributos se declaran dentro de __init__
    variables_clase = 'valor variable clase'

    def __init__(self, variable_instancia):
        # contexto dinámico !!
        self.variable_instancia = variable_instancia

    # Para hacer un método estático usamos el decorador @staticmethod con lo que se asocia a la clase
    # YA NO DEFINIMOS SELF no puede acceder a ello porque se asocia a la clase
    # contexto estático !!
    # No recibe ninguna información de la clase en sí misma
    # lo podemos sustituir por una función normal
    @staticmethod
    def metodo_estatico():
        return f'El método estático dice: Pavo real, el valor de variable clase: {MiClase.variables_clase}'
        # si podemos acceder a la variable de clase de forma indirecta, pero no recibe la referencia

    # un método de clase, sin embargo, sí que recibe un contexto de clase
    # recibe un parámetro cls que significa class (podría ser cualquiera, pero se recomienda)
    @classmethod
    # contexto estático !!
    def metodo_clase(cls):
        print(cls.variables_clase)  # recibe correctamente la referencia de nuestra clase

    def metodo_instancia(self):  # contexto dinámico !!
        self.metodo_clase()  # podemos acceder al contexto estático


print(MiClase.variables_clase)  # no hace falta crear una instancia para consultarla
objeto1 = MiClase('valor variable instancia')
objeto2 = MiClase('Otro valor variable instancia')
print(objeto1.variable_instancia)
print(objeto2.variable_instancia)  # cada uno objeto accede a su
#                                     variable de instancia que se asocia al mismo
print(objeto1.variables_clase)
print(objeto2.variables_clase)
# Como en python todas clas clases son objetos, podremos en cualquier momento
# añadir una variable de clase "al vuelo"
MiClase.variables_clase2 = 'valor variable clase 2'
print(MiClase.variables_clase2)  # Directamente no se puede (sin nombre clase)
print(objeto2.variables_clase2)
print(MiClase.metodo_estatico())
MiClase.metodo_clase()
objeto1.metodo_clase()  # se pasa cls auto
objeto1.metodo_instancia()

print(MiClase.__name__)  # MiClase
print(type(objeto1).__name__)  # MiClase
print(MiClase.__module__)  # __main__ al problarlo en main.py
print(type(objeto1).__module__)  # __main__ al problarlo en main.py
'''
__name__ es un string que contiene el nombre de la clase, solo lo tienen las clases no los objetos
__module__ almacena el nombre del módulo que contiene la definición de la clase
__bases__ es una tupla. La tupla contiene clases (no nombres de clases) que son superclases directas para la clase.
por efecto apunta a Object
sólo las clases lo tienen'''

class SuperOne:
    pass


class SuperTwo:
    pass


class Sub(SuperOne, SuperTwo):
    pass


def printBases(cls): # diseñado para presentar claramente el contenido de la tupla.
    print('( ', end='')

    for x in cls.__bases__:
        print(x.__name__, end=' ')
    print(')')


printBases(SuperOne)  # ( object )
printBases(SuperTwo)  # ( object )
printBases(Sub)  # ( SuperOne SuperTwo )

# ++++++++++  ejemplo __dict__ ++++++++
class Classy:
    varia = 1
    def __init__(self):
        self.var = 2

    def method(self):
        pass

    def __hidden(self):
        pass
obj = Classy()

print(obj.__dict__)  # {'var': 2}
print(Classy.__dict__)

# {'__module__': '__main__', 'varia': 1, '__init__': <function Classy.__init__ at 0x7f4b4b66d320>, 'method': <function Classy.method at 0x7f4b4b66d3b0>, '_Classy__hidden': <function Classy.__hidden at 0x7f4b4b66d440>, '__dict__': <attribute '__dict__' of 'Classy' objects>, '__weakref__': <attribute '__weakref__' of 'Classy' objects>, '__doc__': None}


# ******** Obtención y establecimiento del valor de un atributo de una clase desde fuera.
'''introspección, que es la capacidad de un programa para examinar el tipo o las propiedades de un objeto en tiempo de 
ejecución; reflexión, que va un paso más allá, y es la capacidad de un programa para manipular los valores, 
propiedades y/o funciones de un objeto en tiempo de ejecución.

La función de ejemplo llamada incIntsI() obtiene un objeto de cualquier clase, escanea su contenido para encontrar todos
los atributos enteros con nombres que comienzan con i y los incrementa en uno.

'''
class MyClass:
    pass


obj = MyClass()  # define una clase muy simple y rellenamos con atributos
obj.a, obj.b, obj.i, obj.ireal, obj.integer, obj.z = 1, 2, 3, 3.5, 4, 5


def incIntsI(obj):  # función comentada
    for name in obj.__dict__.keys():  # recorremos el dict
        if name.startswith('i'):  # si un nombre comienza con i.
            val = getattr(obj, name)  # usa la función getattr() para obtener su valor actual
            if isinstance(val, int):  # verifique si el valor es de tipo entero y use la función isinstance() para ello
                setattr(obj, name, val + 1)
                '''# setattr(); la función toma tres argumentos: un objeto, el nombre de la propiedad (como una cadena) 
                y el nuevo valor de la propiedad. '''


print(obj.__dict__)
incIntsI(obj)
print(obj.__dict__)

# ******* Propiedades y atributos
'''Podemos acceder a las variables de clase y a las variables de instancia de la superclase desde la subclase. 
Cuando intente acceder a la entidad de cualquier objeto, Python intentará (en este orden):
- Encontrarlo dentro del objeto mismo
- Encontrarlo en todas las clases involucradas en la línea de herencia del objeto de abajo hacia arriba
- si hay más de una clase en una ruta de herencia particular, Python las escanea de izquierda a derecha
Si ambos de los anteriores fallan, se genera una excepción (AttributeError).
Python busca una entidad de abajo hacia arriba y está completamente satisfecho con la primera entidad del nombre 
deseado. Esto quiere decir que si 2 clases tienen una misma variable la buscara primero en el 'padre' y si la encuentra 
ya esta si no al 'abuelo' '''


class Super:
    a = 0
    x = 'abuelo'

    def __init__(self):
        self.supVar = 11


class Sub(Super):
    b = 1
    x = 'padre'

    def __init__(self):
        super().__init__()
        self.subVar = 12


class Sub2(Super):
    b = None
    x = 'PADRE'

    def __init__(self):
        super().__init__()
        self.subVar = 112


class Subsub(Sub, Sub2):
    pass


obj = Subsub()

print(obj.subVar, obj.b, obj.x)  # 12 1 padre
print(obj.supVar, obj.a)  # 11 0


# +++++++++++++ Ejercicios teóricos +++++++++++++:
# ¿Cuáles de las propiedades de clase de Python son variables de instancia y cuáles son variables de clase?
# ¿Cuáles de ellos son privados?
class Python:
    population = 1
    victims = 0
    def __init__(self):
        self.length_ft = 3
        self.__venomous = False
# población y víctimas son variables de clase, mientras que longitud y __venomous
# son variables de instancia (esta última también es privada)

# Vas a negar la propiedad __venomous del objeto version_2, ignorando el hecho de que la propiedad es privada.
# ¿Cómo harás ésto?
version_2 = Python()
version_2._Python__venomous = not version_2._Python__venomous

# Escriba una expresión que verifique si el objeto versión_2 contiene una propiedad de instancia llamada constrictor
# (¡sí, constrictor!).
hasattr(version_2, 'constrictor')



# ***********************************
# ******** constantes python ##########
# ***********************************
# Es una convención porque no existe tal cual
# Entendiéndolo como una variable a la que una vez asignado el valor no se le puede cambiar
# Mayúsculas
# se suelen crear en un archivo diferente e importarlo
# constantes.py
CONSTANTE = 'Valor de mi constante'
MI_CONSTANTE = 'Otro valor de mi constante'
# from constantes import *

print(MI_CONSTANTE)
MI_CONSTANTE = 'otro'  # No debemos cambiar el valor
print(MI_CONSTANTE)


class Matematicas:
    PI = 3.1416


print(Matematicas.PI)


# ++++++ Ejercicio contador de clases +++++++
class Personac:
    __contador_personas = 0  # lo podemos usar de un fichero o bbdd si lo calculamos....

    def __init__(self, nombre, edad):
        self.id_persona = Personac.__generar_siguiente_valor()
        self.nombre = nombre
        self.edad = edad

    def __str__(self):
        return f' Persona [{self.id_persona} {self.nombre} {self.edad}]'

    @classmethod
    def __generar_siguiente_valor(cls):
        cls.__contador_personas += 1
        return cls.__contador_personas


persona1 = Personac('p', 2)
# Persona.__generar_siguiente_valor()   # si no lo hacemos privado se puede llamar...
persona2 = Personac('a', 34)
print(persona2)


# ***********************************
# ******** Diseño de clases ##########
# ***********************************
# UML - Undefined modeling lenguaje, realizado con umlet (umletino.com) Simulación de venta de productos agregarla
# a una orden como si tuviéramos un ticket de venta en el cual vamos a vender varios productos y se van a agregar a una
# orden A partir de esa orden vamos a calcular el total de todos los productos que se han vendido utilizando el precio
# de producto para ello la clase de producto va a tener: id_producto mediante un contador nombre, precio (nos permitirá
# obtener el total del ticket generado por producto agregado, orden método str para imprimir los atributos por cada
# producto que creemos lo vamos a agregar a la clase de orden, como hemos comentado

# ****** RELACION AGREGACION
# mediante una lista de objetos de tipo prod al que se agregará podremos tener varias órdenes y cada una de productos
# tendremos un contador de ordenes, ID , str la primera clase que se recomienda crear es la que no tiene relación con
# ninguna, en este caso producto ya que orden puede recibir un listado de productos

class Producto:
    __contador_productos = 0

    def __init__(self, nombre, precio):
        # Producto.__contador_productos += 1
        self._id_producto = Producto.__generar_siguiente_valor()
        self._nombre = nombre
        self._precio = precio

    # str
    def __str__(self):
        return f'ID Producto {self._id_producto}, Nombre {self.nombre}, Precio: {self.precio}'

    # metodo de clase
    @classmethod
    def __generar_siguiente_valor(cls):
        cls.__contador_productos += 1
        return cls.__contador_productos

    # getter
    @property
    def nombre(self):
        return self._nombre

    @property
    def precio(self):
        return self._precio

    # setter
    @nombre.setter
    def nombre(self, nombre):
        self._nombre = nombre

    @precio.setter
    def precio(self, precio):
        self._precio = precio


class Orden:
    __contador_ordenes = 0

    def __init__(self, productos):  # recibe una lista de productos
        self._id_orden = Orden.__generar_siguiente_valor()
        self._productos = list(productos)  # si no es una lista falla

    @classmethod
    def __generar_siguiente_valor(cls):
        cls.__contador_ordenes += 1
        return cls.__contador_ordenes

    # para poder agregar productos a posteriori de crearlo
    def agregar_producto(self, producto):
        self._productos.append(producto)  # es una lista

    # Para calcular el total del precio de productos
    def calcular_total(self):
        # variable tmp
        total = 0
        for producto in self._productos:  # vamos a recorrer para cada lista de productos de la orden
            total += producto.precio  # porque tenemos el get | hacemos la suma
        return total  # devolvemos el valor

    # Para poder convertir todos nuestros productos a string
    def __str__(self):
        productos_str = ''  # vamos a guardar temporalmente cada llamada \n para q se vea limpio
        # a su vez se llamará a str del producto
        for producto in self._productos:  # por cada producto de la lista
            productos_str += producto.__str__() + '\n'  # para separar
        return f'Orden: {self._id_orden},  productos: \n{productos_str}'


# prueba en este fichero
if __name__ == '__main__':
    # Archivo test_ordenes.py
    # from Orden import Orden
    # form Producto import Producto
    pr1 = Producto('camisa', 100.00)
    pr2 = Producto('pantalon', 150.00)
    pr3 = Producto('blusa', 70.00)
    pr4 = Producto('pinza', 0.5)

    productos = [pr1, pr2]
    productos2 = [pr3, pr4]
    or1 = Orden(productos)
    or2 = Orden(productos2)
    or1.agregar_producto(pr3)  # probamos agregar producto
    print(or1)
    print(f'total orden: {or1.calcular_total()}')
    print(or2)
    print(f'total orden: {or2.calcular_total()}')

# ****** RELACION COMPOSICION
'''
La herencia no es la única forma de construir clases adaptables. Puede lograr los mismos objetivos (no siempre, pero 
muy a menudo) utilizando una técnica llamada composición.es el proceso de componer un objeto utilizando otros objetos 
diferentes. Los objetos usados en la composición entregan un conjunto de rasgos deseados (propiedades y/o métodos) por 
lo que podemos decir que actúan como bloques usados para construir una estructura más complicada.

Puede decirse que: la herencia extiende las capacidades de una clase agregando nuevos componentes y modificando 
los existentes; en otras palabras, la receta completa está contenida dentro de la propia clase y todos sus ancestros; 
el objeto toma todas las pertenencias de la clase y hace uso de ellas; La composición proyecta una clase como un 
contenedor capaz de almacenar y usar otros objetos (derivados de otras clases) donde cada uno de los objetos implementa 
una parte del comportamiento de una clase deseada.

Ilustremos la diferencia utilizando los vehículos. El enfoque de herencia nos llevó a una jerarquía de clases en la que
la clase superior conocía las reglas generales que se usaban para girar el vehículo, pero no sabía cómo controlar los 
componentes apropiados (ruedas u orugas). Las subclases implementaron esta habilidad mediante la introducción de 
mecanismos especializados. Hagamos (casi) lo mismo, pero usando composición. La clase, sabe cómo girar el vehículo, 
pero el giro real lo realiza un objeto especializado almacenado en una propiedad llamada controlador. El controlador 
puede controlar el vehículo manipulando las partes relevantes del vehículo.
'''
import time

class Tracks:
    def change_direction(self, left, on):
        print("tracks: ", left, on)


class Wheels:
    def change_direction(self, left, on):
        print("wheels: ", left, on)


class Vehicle:
    def __init__(self, controller):
        self.controller = controller

    def turn(self, left):
        self.controller.change_direction(left, True)
        time.sleep(0.25)
        self.controller.change_direction(left, False)


wheeled = Vehicle(Wheels())
tracked = Vehicle(Tracks())

wheeled.turn(True)  # wheels:  True True \n wheels:  True False
tracked.turn(False)  # tracks:  False True tracks:  False False
'''
Hay dos clases llamadas Tracks y wheels: saben cómo controlar la dirección del vehículo. También hay una clase llamada 
Vehículo que puede usar cualquiera de los controladores disponibles (los dos ya definidos o cualquier otro definido 
en el futuro): el controlador en sí se pasa a la clase durante la inicialización. De esta forma, la capacidad de giro
del vehículo se compone utilizando un objeto externo, no implementado dentro de la clase de Vehículo.
En otras palabras, tenemos un vehículo universal y podemos instalarle una u otra.

la herencia múltiple viola el principio de responsabilidad única ya que crea una nueva clase de dos (o más) clases que 
no saben nada entre sí; sugerimos encarecidamente la herencia múltiple como la última de todas las soluciones posibles: 
si realmente necesita las muchas funcionalidades diferentes que ofrecen las diferentes clases, la composición puede ser 
una mejor alternativa.
'''




# ***********************************
# ******** Sobrecarga ##########
# ***********************************
# Operador +
a = 2
b = 3
print(a + b)  # 5
a = 'h'
b = 'o'
print(a + b)  # ho
a = [1, 2, 3]
b = [4, 5, 6]
print(a + b)  # [1, 2, 3, 4, 5, 6]


# se puede sobrecargar una clase, pero tenemos que agregar la sobrecarga de un método del operador ( clase object )
# la sobrecarga de un operador significa que un mismo operador
# Se comporta de manera distinta dependiendo de los operandos por ejemplo la suma (+)
# al realizarse sobre números o listas
# sobrecarga de operadores:
#   + --> __add__(self, other) #   += --> __iadd__(self, other) # UNARIO  - --> __neg__(self, other)
#   - --> __sub__(self, other) #   -= --> __isub__(self, other) # UNARIO  + --> __pos__(self, other)
#   * --> __mul__(self, other) #   *= --> __imul__(self, other) # UNARIO   --> __invent__(self, other)
#   / --> __truediv__(self, other) #   /= --> __idiv__(self, other)
#   // --> __floordiv__(self, other) #   //= --> __ifloordiv__(self, other)
#   % --> __mod__(self, other) #   %= --> __imod__(self, other)
#   ** --> __pow__(self, other) #   **= --> __ipow__(self, other)
# SOBRECARGA : Se comporta de manera distinta dependiendo de los operandos
# SOBRRESCRITURA : herencia se ejecuta el de la clase a la q sea empezando por el hijo


# si queremos sumar una clase nuestra por defecto no podemos sin sobrecargarlo
class Per:

    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad

    def __add__(self, otro):
        # lo que quiere decir es que al realizar la suma + lo que
        # estamos realizando es sobre el objeto de la izquierda
        # llamar al metodo add con el segundo objeto como argumento
        # miobjeto1 + miobjeto2 es lo mismo que obj1.__add__(obj2)
        # en este ejemplo vamos a concatenar los nombres
        return f'{self.nombre} {otro.nombre}'

    def __sub__(self, otro):
        # en este estamos restando la edad de los objetos
        return self.edad - otro.edad


p1 = Per('Juan', 53)
p2 = Per('carlos', 23)
print(p1 + p2)
print(p1 - p2)


# ***********************************
# ******** Polimorfismo ##########
# ***********************************
'''Multiples formas en tiempo de ejecución
una misma variable puede ejecutar varios métodos de distintos objetos dependiendo del objeto
al cual esté apuntando en tiempo en ejecución
Si tenemos una variable de una clase que tiene el método str, y otra que tiene gerente y ejecutarse el q sea
Es decir ejecutar multiples métodos en tiempo de ejecución dependiendo del objeto al cual esté apuntando
se ejecuta uno dependiendo de cuál apunte. En Python no tienen que tener relación
Es La situación en la que la subclase es capaz de modificar el comportamiento de su superclase. Ayuda al desarrollador
a mantener el código limpio y consistente.'''

class Empleado:
    def __init__(self, nombre, sueldo):
        self._nombre = nombre
        self._sueldo = sueldo

    def __str__(self):
        return f'Empleado: [Nombre: {self._nombre}, Sueldo: {self.sueldo}]'

    @property
    def nombre(self):
        return self._nombre

    @property
    def sueldo(self):
        return self._sueldo

    @nombre.setter
    def nombre(self, nombre):
        self._nombre = nombre

    @sueldo.setter
    def nombre(self, sueldo):
        self._sueldo = sueldo

    def mostrar_detalles(self):
        return self.__str__()


class Gerente(Empleado):
    def __init__(self, nombre, sueldo, departamento):
        super().__init__(nombre, sueldo)  # ==  Empleado.__init__(self, nombre, sueldo)
        self._departamento = departamento

    @property
    def departamento(self):
        return self._departamento

    def __str__(self):
        return f'Gerente: [Departamento: {self.departamento}, {super().__str__()}]'


# creamos una función que imprima un objeto
def imprimir_detalles(objeto):
    print(objeto)
    print(objeto.mostrar_detalles())
    # mismo resultado se ejecuta el método del padre pero el str de la que está apuntando
    print(type(objeto))
    if isinstance(objeto, Gerente):  # ****** metodo que pregunta si un objeto es de una determinada clase
        print(objeto.departamento)


empleado = Empleado('Alberto', 1000)
gerente = Gerente('Emilio', 4500, 'BI')
imprimir_detalles(empleado)
imprimir_detalles(gerente)

# ***********************************
# ******** Generadores   ##########
# ***********************************
'''
Un generador de Python es una pieza de código especializado capaz de producir una serie de valores y controlar el
proceso de iteración. Esta es la razón por la que los generadores se denominan con mucha frecuencia iteradores, y aunque
algunos pueden encontrar una distinción muy sutil entre estos dos, los trataremos como uno solo. La función range() es,
de hecho, un generador, que es (de hecho, de nuevo) un iterador.

¿Cuál es la diferencia?
'''
for i in range(5):
    print(i)
'''
Una función devuelve un valor bien definido; puede ser el resultado de una evaluación más o menos compleja de, 
por ejemplo, un polinomio, y se invoca una vez, solo una vez.

Un generador devuelve una serie de valores y, en general, se invoca (implícitamente) más de una vez.

En el ejemplo, el generador range() se invoca seis veces, proporcionando cinco valores subsiguientes de cero a cuatro 
y finalmente señalando que la serie está completa.

El protocolo iterador es una forma en la que un objeto debe comportarse para ajustarse a las reglas impuestas por el 
contexto de las declaraciones for e in. Un objeto que se ajusta al protocolo del iterador se denomina iterador.

Un iterador debe proporcionar dos métodos:

__iter__() que debería devolver el objeto en sí mismo y que se invoca una vez (es necesario para que Python inicie 
con éxito la iteración)
__next__() que está destinado a devolver el siguiente valor (primero, segundo, etc.) de la serie deseada: será invocado 
por las declaraciones for/in para pasar a la siguiente iteración; si no hay más valores para proporcionar, el método 
debe generar la excepción StopIteration.
'''
class Fib:
    def __init__(self, nn):
        print("__init__")
        self.__n = nn
        self.__i = 0
        self.__p1 = self.__p2 = 1

    def __iter__(self):
        print("__iter__")
        return self

    def __next__(self):
        print("__next__")
        self.__i += 1
        if self.__i > self.__n:
            raise StopIteration
        if self.__i in [1, 2]:
            return 1
        ret = self.__p1 + self.__p2
        self.__p1, self.__p2 = self.__p2, ret
        return ret


for i in Fib(10):
    print(i)
'''
Hemos construido una clase capaz de iterar a través de los primeros n valores (donde n es un parámetro del constructor) 
de los números de Fibonacci.

Permítanos recordarle: los números de Fibonacci (Fibi) se definen de la siguiente manera:

Fib1 = 1
Fib2 = 1
Fibi = Fibi-1 + Fibi-2

En otras palabras:

los dos primeros números de Fibonacci son iguales a 1;
cualquier otro número de Fibonacci es la suma de los dos anteriores (por ejemplo, Fib3 = 2, Fib4 = 3, Fib5 = 5, etc.)
Vamos a sumergirnos en el código:

líneas 2 a 6: el constructor de la clase imprime un mensaje (lo usaremos para rastrear el comportamiento de la clase), 
prepara algunas variables (__n para almacenar el límite de la serie, __i para rastrear el número de Fibonacci actual 
para proporcionar y __p1 junto con __p2 para guardar los dos números anteriores);

líneas 8 a 10: el método __iter__ está obligado a devolver el propio objeto iterador; su propósito puede ser un poco 
ambiguo aquí, pero no hay misterio; intente imaginar un objeto que no sea un iterador (por ejemplo, es una colección de 
algunas entidades), pero uno de sus componentes es un iterador capaz de escanear la colección; el método __iter__ debe 
extraer el iterador y confiarle la ejecución del protocolo de iteración; como puede ver, el método inicia su acción 
imprimiendo un mensaje;

líneas 12 a 21: el método __next__ es responsable de crear la secuencia; es algo prolijo, pero esto debería hacerlo más 
legible; primero, imprime un mensaje, luego actualiza la cantidad de valores deseados y, si llega al final de la 
secuencia, el método interrumpe la iteración al generar la excepción StopIteration; las últimas líneas  hacen uso del 
iterador.El código produce el siguiente resultado:
__init__
__iter__
__next__
1
__next__
1
__next__
2
__next__
3
…
primero se crea una instancia del objeto iterador;
a continuación, Python invoca el método __iter__ para obtener acceso al iterador real;
el método __next__ se invoca once veces: las primeras diez veces producen valores útiles, mientras que la undécima 
finaliza la iteración.
'''
class Fib:
    def __init__(self, nn):
        self.__n = nn
        self.__i = 0
        self.__p1 = self.__p2 = 1

    def __iter__(self):
        print("Fib iter")
        return self

    def __next__(self):
        self.__i += 1
        if self.__i > self.__n:
            raise StopIteration
        if self.__i in [1, 2]:
            return 1
        ret = self.__p1 + self.__p2
        self.__p1, self.__p2 = self.__p2, ret
        return ret


class Class:
    def __init__(self, n):
        self.__iter = Fib(n)

    def __iter__(self):
        print("Class iter")
        return self.__iter


object = Class(8)  # se inicializa class que asu vez fib

for i in object: # es este for el que llama a iter() que a su vez llama a next
    print(i)
    '''
Hemos construido el iterador Fib en otra clase (podemos decir que lo hemos integrado en la clase Class). Se crea una
instancia junto con el objeto de Class. El objeto de la clase puede usarse como iterador cuando (y solo cuando) responde
positivamente a la invocación de __iter__; esta clase puede hacerlo, y si se invoca de esta manera, proporciona un
objeto capaz de obedecer el protocolo de iteración. Esta es la razón por la que la salida del código es la misma que
antes, aunque el objeto de la clase Fib no se usa explícitamente dentro del contexto del bucle for.
podemos ahorrar memoria'''


# ++++ Ejemplo myrange +++++
'''
Con iter Lo que se devuelve es un objeto iterable, que se asigna principalmente a la Función __iter__ de la clase, esta 
función devuelve una implementación de la función __next__ del Objeto. Al llamar iter, se genera un objeto de iteración, 
que requiere __iter__ bebe devolver una implementación del método __next__ , next visita el siguiente elemento de este 
objeto y lanza uno si no desea continuar iterando StopIteration La excepción (for La declaración detectará esta 
excepción y finalizará automáticamente for), la función myrange es similar al funcionamiento de  range'''
class MyRange(object):
    def __init__(self, end):
        self.start = 0
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.start < self.end:
            ret = self.start
            self.start += 1
            return ret
        else:
            raise StopIteration


from collections.abc import *

a = MyRange(5)
print(isinstance(a, Iterable))
print(isinstance(a, Iterator))

for i in a:
    print(i)

# ******* Yield
'''
El protocolo del iterador no es particularmente difícil de entender y usar, pero también es indiscutible que el 
protocolo es bastante inconveniente. La principal incomodidad que trae es la necesidad de guardar el estado de la 
iteración entre invocaciones posteriores de __iter__. Por ejemplo, el iterador de Fib se ve obligado a almacenar con 
precisión el lugar en el que se detuvo la última invocación (es decir, el número evaluado y los valores de los dos 
elementos anteriores). Esto hace que el código sea más grande y menos comprensible.

Esta es la razón por la que Python ofrece una forma mucho más efectiva, conveniente y elegante de escribir iteradores.
El concepto se basa fundamentalmente en un mecanismo muy específico y poderoso provisto por la palabra clave yield. 
Puede pensar en la palabra clave yield como un hermano más inteligente de la declaración return, con una 
diferencia esencial.
'''
def fun(n):
    for i in range(n):
        return i
'''
Se ve extraño, ¿no? Está claro que el ciclo for no tiene posibilidad de terminar su primera ejecución, ya que el retorno
lo romperá irrevocablemente. Además, invocar la función no cambiará nada: el ciclo for comenzará desde cero y se 
interrumpirá de inmediato. Podemos decir que dicha función no puede guardar y restaurar su estado entre invocaciones 
posteriores. Esto también significa que una función como esta no se puede usar como generador.
'''
def fun(n):
    for i in range(n):
        yield i


for v in fun(5):
    print(v)  # 0 1 2 3 4
'''
Hemos agregado yield en lugar de retorno. Esta pequeña enmienda convierte la función en un generador. En primer lugar, 
proporciona el valor de la expresión especificada después de la palabra clave yield, al igual que return, pero no pierde 
el estado de la función. Todos los valores de las variables se congelan y esperan la próxima invocación, cuando se 
reanuda la ejecución (no se toma desde cero, como después de la devolución).

Hay una limitación importante: dicha función no debe invocarse explícitamente ya que, de hecho, ya no es una función; 
es un objeto generador ( si lo imprimes ves que es un objeto generador y lo tienes que llamar desde por ejemplo un for 
una compresión de listas, una lista! e incluso el operador in (como for i in renge(x)). La invocación devolverá el 
identificador del objeto, no la serie que esperamos del generador.
'''
# ++++++++ ejemplo yield +++++
def powers_of_2(n):
    power = 1
    for i in range(n):
        yield power
        power *= 2


t = [x for x in powers_of_2(5)]
l = list(powers_of_2(3))
for i in range(20):
    if i in powers_of_2(8):
        print(i, end=' ')
print(t)  # [1, 2, 4, 8, 16]
print(l)  # [1, 2, 4]

# ++++++++ ejemplo Fibonacci con yield +++++
def fibonacci(n):
    p = pp = 1
    for i in range(n):
        if i in [0, 1]:
            yield 1
        else:
            n = p + pp
            pp, p = p, n
            yield n


fibs = list(fibonacci(10))
print(fibs)


# Compresión de listas y de generadores
'''Un solo cambio puede convertir cualquier lista de comprensión en un generador.Son los paréntesis. Los corchetes hacen
una comprensión, los paréntesis hacen un generador. '''
the_list = [1 if x % 2 == 0 else 0 for x in range(10)]
the_generator = (1 if x % 2 == 0 else 0 for x in range(10))

for v in the_list:
    print(v, end=" ")  # 1 0 1 0 1 0 1 0 1 0
print()

for v in the_generator:
    print(v, end=" ")  # 1 0 1 0 1 0 1 0 1 0
print()
'''
Aplique la función len() a ambas entidades. len(the_list) se evaluará como 10. Claro y predecible. len(the_generator) 
generará una excepción TypeError: object of type 'generator' has no len()
la misma apariencia de la salida no significa que ambos bucles funcionen de la misma manera. En el primer ciclo, la 
lista se crea (y se repite) como un todo; en realidad existe cuando se ejecuta el ciclo. En el segundo bucle, no hay 
ninguna lista: solo hay valores posteriores producidos por el generador, uno por uno.'''


# ******* Closures
'''El cierre es una técnica que permite almacenar valores a pesar de que el contexto en el que han sido creados ya no
existe.
'''
# def outer(par):
#     loc = par
#
#
# var = 1
# outer(var)
# print(var)
# print(loc)
'''
Las últimas dos líneas de este ejemplo causarán una excepción NameError: ni par ni loc son accesibles fuera de la 
función. Ambas variables existen cuando y solo cuando se ejecuta la función outer().
'''


def outer(par):
    loc = par

    def inner():
        return loc
    return inner


var = 1
fun = outer(var)
print(fun()) # 1
'''
Ahora Hay  nuevo: una función (llamada inner()) dentro de otra función (llamada outer()). Funciona como cualquier otra 
función, excepto por el hecho de que inner() puede invocarse solo desde outer(). Podemos decir que inner() es la 
herramienta privada de outer() - ninguna otra parte del código puede acceder a ella.
- la función inner() devuelve el valor de la variable accesible dentro de su alcance, ya que inner() puede usar 
cualquiera de las entidades a disposición de outer()
- la función outer() devuelve la función inner() en sí misma; más precisamente, devuelve una copia de la función inner()
, la que estaba congelada en el momento de la invocación de outer(); la función congelada contiene su entorno completo, 
incluido el estado de todas las variables locales, lo que también significa que el valor de loc se conserva con éxito, 
aunque outer() dejó de existir hace mucho tiempo. La función devuelta durante la invocación de outer() es un cierre.

Un cierre debe ser invocado exactamente de la misma manera en que ha sido declarado. la función inner() no tiene 
parámetros, por lo que tenemos que invocarla sin argumentos. 
'''
def make_closure(par):
    loc = par

    def power(p):
        return p ** loc
    return power


fsqr = make_closure(2)
fcub = make_closure(3)
for i in range(5):
    print(i, fsqr(i), fcub(i))
'''
Es totalmente posible declarar un cierre equipado con un número arbitrario de parámetros, por ejemplo, uno, al igual 
que la función power(). Esto significa que el cierre no solo hace uso del entorno congelado, sino que también puede 
modificar su comportamiento al usando valores tomados del exterior. Este ejemplo muestra una circunstancia más 
interesante: puede crear tantos cierres como desee utilizando el mismo código. Esto se hace con una función llamada 
make_closure(). En el ejemplo, el primer cierre obtenido de make_closure() define una herramienta que eleva al cuadrado 
su argumento; el segundo está diseñado para reducir al cubo el argumento.
Es por eso que el código produce el siguiente resultado:
0 0 0
1 1 1
2 4 8
3 9 27
4 16 64

PEP 8, la Guía de estilo para el código de Python, recomienda que las lambdas no se asignen a variables, sino que se 
definan como funciones. Esto significa que es mejor usar una declaración de definición y evitar usar una declaración de
asignación que vincule una expresión lambda a un identificador. Por ejemplo:
Escriba una función lambda, establezca el bit menos significativo de su argumento entero y aplíquelo a la función map() 
para producir la cadena 1 3 3 5 en la consola.
# Recommended:
def f(x): return 3*x  # o sea cierre
# Not recommended:
f = lambda x: 3*x

La vinculación de lambdas a identificadores generalmente duplica la funcionalidad de la instrucción def. El uso de 
sentencias def, por otro lado, genera más líneas de código. Es importante entender que la realidad muchas veces gusta
 de dibujar sus propios escenarios, que no necesariamente siguen las convenciones o recomendaciones formales. Si decide
 seguirlos o no, dependerá de muchas cosas: sus preferencias, otras convenciones adoptadas, las pautas internas de la 
 empresa, la compatibilidad con el código existente, etc. Tenga esto en cuenta.
'''

# +++++++ ejemplo closure +++++
def tag(tg):
    tg2 = tg
    tg2 = tg[0] + '/' + tg[1:]

    def inner(str):
        return tg + str + tg2
    return inner


b_tag = tag('<b>')
print(b_tag('Monty Python'))  # <b>Monty Python</b>

# ++++++++ ejemplo pregunta modulo +++++
class I:
    def __init__(self):
        self.s = ['abc', '123']
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == len(self.s):
            raise StopIteration
        v = self.s[self.i]
        self.i += 1
        return v


for x in I():
    print(x, end='')  # abc123


# ++++++ otro
class Ex(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg+msg)
        self.args = (msg,)


try:
    raise Ex('ex')
except Ex as e:
    print(e)  # ex
except Exception as e:
    print(e)


# +++++ otro

try:
    raise Exception(1, 2, 3)
except Exception as e:
    print(len(e.args))  # 3

# ++++++ otro
class A:
    def __init__(self):
        pass

a = A(1)

print(hasattr(a, 'A'))  # Falla 2 argumentos y solo hemos definido uno!!!

# +++++++ otro
# ¿cual es la salida esperada del siguiente código?
class Vowels:
    def __init__(self):
        self.vow = "aeiouy "  # Yes, we know that y is not always considered a vowel.
        self.pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos == len(self.vow):
            raise StopIteration
        self.pos += 1
        return self.vow[self.pos - 1]

vowels = Vowels()
for v in vowels:
    print(v, end=' ')  # a e i o u y

# +++++++ otro
# Escriba una función lambda, establezca el bit menos significativo de su argumento entero y aplíquelo a la función
# map() para producir la cadena 1 3 3 5 en la consola.
any_list = [1, 2, 3, 4]
# even_list = # Complete the line here.
# print(even_list)

list(map(lambda n: n | 1, any_list))  # recuerda, | es como un or

# +++++++++ otro
# ¿cual es la salida esperada del siguiente código?

def replace_spaces(replacement='*'):
    def new_replacement(text):
        return text.replace(' ', replacement)
    return new_replacement


stars = replace_spaces()
print(stars("And Now for Something Completely Different"))  # And*Now*for*Something*Completely*Different


# ***********************************
# ********  Excepciones ##########
# ***********************************
'''
# Manejo de errores
en python Tenemos la clase base BaseException: 
la más general (abstracta) de todas las excepciones de Python - todas las demás excepciones están incluidas en esta; 
se puede decir que las siguientes dos ramas excepto son equivalentes:
de ella hereda Exception
    de la cual cuelgan más:
        AritmeticError, abstracta(ZeroDivisionError)
        OSError abstracta(FileNotFoundError, PermissionError)
        RuntimeError,
        LookupError abstracta(IndexError, KeyError),
        SyntaxError

Si desea controlar dos o más excepciones de la misma manera, puede usar la siguiente sintaxis:
'''
while True:
    try:
        number = int(input("Enter an int number: "))
        print(5/number)
        break
    except (ValueError, ZeroDivisionError):
        print("Wrong value or No division by zero rule broken.")
    except:
        print("Sorry, something went wrong...")


# ****** una excepción

try:
    '10' / 0
except Exception as e:  # ZeroDivisionError peta
    # Si capturamos un error con clase más específica, no cogerá los que se salgan
    # Solamente clases padre pueden procesar excepciones, incluyendo clases hijas
    print(f'Ocurrió un error: {e}')

# ******  varias excepciones
resultado = None  # Al usarse fuera del try hay que declararla fuera si no falla
try:
    a = int(input('Primer numero: '))
    b = int(input('Segundo numero: '))
    resultado = a / b
except ZeroDivisionError as e:  # División entre 0
    print(f'ZeroDivisionError Ocurrió un error: {e}, {type(e)}')
except TypeError as e:  # Error de tipo de datos
    print(f'TypeError Ocurrió un error: {e}, {type(e)}')
except Exception as e:  # hija de BaseException ( la más general)
    print(f'Exception Ocurrió un error: {e}, {type(e)}')
except:  # general, en este caso solo cogería KeyboardInterrupt o BaseException
    print('no')
else:  # solo se ejecuta si no se lanza NINGUNA excepción
    print('No se arrojó ninguna excepción')  # 10 / 2 es ok, se arroja:  No se arrojó ninguna excepción
finally:  # Siempre se ejecuta incluso si se lanza una excepción
    print('Continuamos')
    print(f'Resultado: {resultado}')


# podemos manejas varias excepciones deben ir de más específico a más genérico
# Si ponemos la más general al principio no se manejarán las más específicas
# 2: Excepción propia


class NumerosIdenticosException(Exception):
    def __init__(self, mensaje):
        self.message = mensaje


resultado = None  # Al usarse fuera del try hay que declararla fuera si no falla
try:
    a = int(input('Primer numero: '))
    b = int(input('Segundo numero: '))
    if a == b:
        raise NumerosIdenticosException('Números iguales')
        # si pasa, ya estamos lanzando la excepción pues raise arroja la excepción
        # raise puede arrojar cualquier error no solo personalizada
    resultado = a / b
except Exception as e:
    print(f'Exception Ocurrió un error: {e}, {type(e)}')
else:  # solo se ejecuta si no se lanza NINGUNA excepción
    print('No se arrojó ninguna excepción')  # 10 / 2 es ok, se arroja:  No se arrojó ninguna excepción
finally:  # Siempre se ejecuta incluso si se lanza una excepción
    print('Continuamos')
    print(f'Resultado: {resultado}')

'''
IndexError
Location: BaseException ← Exception ← LookupError ← IndexError
Una excepción concreta que se genera cuando se intenta acceder al elemento de una secuencia inexistente (por ejemplo, 
el elemento de una lista)
'''
the_list = [1, 2, 3, 4, 5]
ix = 0
do_it = True

while do_it:
    try:
        print(the_list[ix])
        ix += 1
    except IndexError:
        do_it = False

print('Done')

'''
ArithmeticError
Location: BaseException ← Exception ← ArithmeticError
Una excepción abstracta que incluye todas las excepciones causadas por operaciones aritméticas como la división cero 
o el dominio no válido de un argumento
'''

def bad_fun(n):
    # return 1 / n
    raise ZeroDivisionError
try:
    bad_fun(0)
except ArithmeticError:
    print("What happened? An exception was raised!")

print("THE END.")

# al ejecutarlo : 
# What happened? An error?
# THE END.
'''
la excepción planteada puede cruzar los límites de la función y el módulo, 
y viajar a través de la cadena de invocación en busca de una cláusula excepto coincidente capaz de manejarla. 
Si no existe tal cláusula, la excepción permanece sin manejar, y Python resuelve el problema de su manera estándar: 
terminando su código y emitiendo un mensaje de diagnóstico.

raise es una palabra clave. La instrucción le permite: Simular la generación de excepciones reales (por ejemplo, 
para probar su estrategia de manejo) manejar parcialmente una excepción y hacer que otra parte del código sea 
responsable de completar el manejo (separación de preocupaciones).
'''
def bad_fun(n):
    try:
        return n / 0
    except:
        print("I did it again!")
        raise  # !!!!!


try:
    bad_fun(0)
except ArithmeticError:
    print("I see!")

#salida de ejecución:
#I did it again!
#I see!
#THE END.
'''
La instrucción de aumento también se puede utilizar de la siguiente manera (tenga en cuenta la ausencia del nombre de 
la excepción, solo raise): este tipo de instrucción de elevación puede usarse SOLO dentro  de la rama except; Usarlo en
cualquier otro contexto causa un error. 
La instrucción volverá a generar inmediatamente la misma excepción que se maneja actualmente.
Gracias a esto, puede distribuir el manejo de excepciones entre diferentes partes del código. en el ejemplo anterior 
ZeroDivisionError se genera dos veces: 
Primero, dentro de la parte Try del código (esto es causado por la división cero real) 
Segundo, dentro de la parte excepto por la instrucción de elevación.

AssertionError 
BaseException ← Exception ← AssertionError 
Una excepción concreta provocada por la instrucción assert cuando su argumento se evalúa como False, None, 0 o 
una cadena vacía ( si es true o valor no 0 no hará nada)

una excepción AssertionError protege el código para que no produzca resultados no válidos y muestra claramente 
la naturaleza del error; el papel de un airbag.
'''
import math

x = float(input("Enter a number: "))  # -1-> File "main.py", line 4, in <module> \n assert x >= 0.0 \n AssertionError

assert x >= 0.0

x = math.sqrt(x)

print(x)

'''
KeyboardInterrupt 
BaseException ← KeyboardInterrupt
Una excepción concreta planteada cuando el usuario utiliza un atajo de teclado diseñado para 
terminar la ejecución de un programa (Ctrl-C en la mayoría de los sistemas operativos); Si el control de esta excepción 
no conduce a la finalización del programa, el programa continúa su ejecución. Nota: esta excepción no se deriva de la 
clase Exception. Este ejemplo no termina al finalizar con control c
'''
from time import sleep
seconds = 0
while True:
    try:
        print(seconds)
        seconds += 1
        sleep(1)
    except KeyboardInterrupt:
        print("Don't do that!")

'''
MemoryError
BaseException ← Exception ← MemoryError
# Una excepción concreta planteada cuando una operación no se puede completar debido a la falta de memoria libre.'''
string = 'x'
try:
    while True:
        string = string + string
        print(len(string))
except MemoryError:
    print('This is not funny!')
'''
OverflowError
BaseException ← Exception ← ArithmeticError ← OverflowError
Ubicación: BaseException ← Exception ← ArithmeticError ← OverflowError 
una excepción concreta que se produce cuando una operación produce un número demasiado grande para almacenarse 
correctamente'''
from math import exp

ex = 1

try:
    while True:
        print(exp(ex))  # values of exp(k), k = 1, 2, 4, 8, 16, ...
        ex *= 2
except OverflowError:
    print('The number is too big.')
'''
ImportError
Location: BaseException ← Exception ← StandardError ← ImportError
Description: Una excepción concreta que se produce cuando se produce un error en una operación de importación
'''
try:
    import math
    import time
    import abracadabra

except:
    print('One of your imports has failed.')
'''
KeyError Ubicación: BaseException ← Exception ← LookupError ← KeyError Descripción: una excepción concreta que se 
produce cuando se intenta acceder al elemento inexistente de una colección (por ejemplo, el elemento de un diccionario)
'''
dictionary = { 'a': 'b', 'b': 'c', 'c': 'd' }
ch = 'a'

try:
    while True:
        ch = dictionary[ch]
        print(ch)
except KeyError:
    print('No such key:', ch)


# +++++ Ejercicio comprobador de rango numérico
def read_int(prompt, mi, ma):
    while True:
        v = input(prompt)
        try:
            v = int(v)
            if not mi <= v <= ma:
                raise Exception(f'Error: the value is not within permitted range ({mi}..{ma})')
            return v
        except ValueError:
            print('wrong input')
        except Exception as e:
            # as e, le permite interceptar un objeto que lleva información sobre una excepción pendiente.
            # La propiedad del objeto llamada args (una tupla) almacena todos los argumentos pasados al objeto.
            print(e)


v = read_int("Enter a number from -10 to 10: ", -10, 10)
print("The number is:", v)

# +++++ Ejemplo Arbol de excepciones ++++++
'''Las excepciones son las clases.
Todas las excepciones integradas de Python forman una jerarquía de clases. No hay obstáculo para extenderlo si lo 
encuentras razonable. Este programa vuelca todas las clases de excepción predefinidas en forma de una copia impresa 
en forma de árbol. Como un árbol es un ejemplo perfecto de una estructura de datos recursiva, una recursión parece 
ser la mejor herramienta para atravesarla. La función print_exception_tree() toma dos argumentos:
- un punto dentro del árbol desde el cual comenzamos a atravesar el árbol;
- un nivel de anidamiento (lo usaremos para construir un dibujo simplificado de las ramas del árbol)

Comencemos desde la raíz del árbol: la raíz de las clases de excepción de Python es la clase BaseException (es una 
superclase de todas las demás excepciones). Para cada una de las clases encontradas, realice el mismo conjunto de 
operaciones:
- imprime su nombre, tomado de la propiedad __name__;
- iterar a través de la lista de subclases entregadas por el método __subclasses__() e invocar recursivamente 
  la función print_exception_tree(), incrementando el nivel de anidamiento respectivamente.'''
def print_exception_tree(thisclass, nest = 0):
    if nest > 1:
        print("   |" * (nest - 1), end="")
    if nest > 0:
        print("   +---", end="")

    print(thisclass.__name__)

    for subclass in thisclass.__subclasses__():
        print_exception_tree(subclass, nest + 1)


print(BaseException.__subclasses__())  # [<class 'BaseExceptionGroup'>, <class 'Exception'>, <class 'GeneratorExit'>,...
print(Exception.__subclasses__())  # [<class 'ArithmeticError'>, <class 'AssertionError'>, <class 'AttributeError'>, ...
print_exception_tree(BaseException)
''' BaseException
   +---BaseExceptionGroup
   |   +---ExceptionGroup
   +---Exception
   |   +---ArithmeticError
   |   |   +---FloatingPointError
   |   |   +---OverflowError
   |   |   +---
   ...'''

#  ***** detalle de anatomía de excepciones

'''La clase BaseException introduce una propiedad llamada args. Es una tupla diseñada para recopilar todos los 
argumentos pasados al constructor de la clase. Está vacío si la construcción ha sido invocada sin ningún argumento, o si
 contiene solo un elemento cuando el constructor obtiene un argumento (aquí no contamos el argumento self), y así 
 sucesivamente.'''
def print_args(args):
    lng = len(args)
    if lng == 0:
        print("")
    elif lng == 1:
        print(args[0])
    else:
        print(str(args))


try:
    raise Exception
except Exception as e:
    print(e, e.__str__(), sep=' : ', end=' : ')  #  :  :
    print_args(e.args) #

try:
    raise Exception("my exception")
except Exception as e:
    print(e, e.__str__(), sep=' : ', end=' : ')  # my exception : my exception :
    print_args(e.args)  # my exception    (ojo lng es 1)

try:
    raise Exception("my", "exception")
except Exception as e:
    print(e, e.__str__(), sep=' : ', end=' : ')  # ('my', 'exception') : ('my', 'exception') :
    print_args(e.args)  # ('my', 'exception')

'''Hemos usado la función para imprimir el contenido de la propiedad args en tres casos diferentes, donde la excepción 
de la clase Exception se genera de tres maneras diferentes. Para hacerlo más espectacular, también imprimimos el objeto 
en sí, junto con el resultado de la invocación de __str__().

El primer caso parece rutinario: solo hay el nombre Exception después de la palabra clave raise. Esto significa que el 
objeto de esta clase ha sido creado de la forma más rutinaria.

El segundo y tercer caso pueden parecer un poco extraños a primera vista, pero no hay nada extraño aquí: estas son solo 
las invocaciones del constructor. En la segunda declaración de aumento, el constructor se invoca con un argumento y en 
la tercera, con dos.

Como puede ver, la salida del programa refleja esto, mostrando los contenidos apropiados de la propiedad args:

  : :
mi excepción: mi excepción: mi excepción
('mi', 'excepción') : ('mi', 'excepción') : ('mi', 'excepción')
'''

# +++++ ejemplo excepciones propias, jerarquía ++++++

'''Cuando va a construir un universo completamente nuevo lleno de criaturas completamente nuevas que no tienen nada en 
común con todas las cosas familiares, es posible que desee construir su propia estructura de excepción. Por ejemplo, 
si trabaja en un gran sistema de simulación destinado a modelar las actividades de una pizzería, puede ser deseable 
formar una jerarquía separada de excepciones. Puede comenzar a construirlo definiendo una excepción general como una 
nueva clase base para cualquier otra excepción especializada. Lo hemos hecho de la siguiente forma:
'''
class PizzaError(Exception):
    def __init__(self, pizza, message):
        Exception.__init__(self, message)
        self.pizza = pizza

'''
Nota: vamos a recopilar información más específica aquí que una excepción normal, por lo que nuestro constructor tomará
dos argumentos:
- uno especificando una pizza como sujeto del proceso,
- y otra que contenga una descripción más o menos precisa del problema.
Como puede ver, pasamos el segundo parámetro al constructor de la superclase y guardamos el primero dentro de nuestra 
propiedad.

Un problema más específico (como un exceso de queso) puede requerir una excepción más específica. Es posible derivar 
la nueva clase de la clase PizzaError ya definida, como lo hemos hecho aquí:
'''
class TooMuchCheeseError(PizzaError):
    def __init__(self, pizza, cheese, message):
        PizzaError._init__(self, pizza, message)
        self.cheese = cheese

'''
La excepción TooMuchCheeseError necesita más información que la excepción PizzaError normal, por lo que la agregamos al 
constructor; el nombre cheese se almacena para su posterior procesamiento.'''
def make_pizza(pizza, cheese):
    if pizza not in ['margherita', 'capricciosa', 'calzone']:
        raise PizzaError(pizza, "no such pizza on the menu")
    if cheese > 100:
        raise TooMuchCheeseError(pizza, cheese, "too much cheese")
    print("Pizza ready!")

for (pz, ch) in [('calzone', 0), ('margherita', 110), ('mafia', 20)]:
    try:
        make_pizza(pz, ch)
    except TooMuchCheeseError as tmce:
        print(tmce, ':', tmce.cheese)
    except PizzaError as pe:
        print(pe, ':', pe.pizza)
'''
Hemos combinado las dos excepciones definidas anteriormente y las hemos aprovechado para que funcionen en un pequeño 
fragmento de código de ejemplo.Uno de estos se genera dentro de la función make_pizza() cuando se descubre cualquiera 
de estas dos situaciones erróneas: una solicitud de pizza incorrecta o una solicitud de demasiado queso.
salida:
Pizza ready!
too much cheese : 110
no such pizza on the menu : mafia

'''

# +++++ ejercicio excepciones +++++
# ¿Cuál es el resultado esperado del siguiente código?

import math

class NewValueError(ValueError):
    def __init__(self, name, color, state):
        self.data = (name, color, state)

try:
    raise NewValueError("Enemy warning", "Red alert", "High readiness")
except NewValueError as nve:
    for arg in nve.args:
        print(arg, end='! ')  # Enemy warning! Red alert! High readiness!


# ***********************************
# ********  Manejo de Archivos ##########
# ***********************************
# podemos trabajar casi cualquier tipo de archivos desde textos hasta imágenes

# ********  Archivos  DE TEXTO ##########
'''
Cualquier programa escrito en Python (y no solo en Python, porque esa convención se aplica a prácticamente todos los 
lenguajes de programación) no se comunica con los archivos directamente, sino a través de algunas entidades abstractas 
que se nombran de manera diferente en diferentes lenguajes o entornos: los términos más utilizados. son identificadores 
o flujos (aquí los usaremos como sinónimos). El programador, que tiene un conjunto más o menos rico de funciones/métodos
,puede realizar ciertas operaciones en el flujo, que afectan a los archivos reales utilizando mecanismos contenidos en 
el kernel del sistema operativo. De esta forma, puede implementar el proceso de acceso a cualquier archivo, incluso 
cuando se desconoce el nombre del archivo en el momento de escribir el programa.

Acceso a archivos: un concepto de estructura de árbol


Para conectar (vincular) la transmisión con el archivo, es necesario realizar una operación explícita. La operación de 
conectar la secuencia con un archivo se denomina abrir (open()) el archivo, mientras que desconectar este enlace se 
denomina cerrar el archivo close(). Por lo tanto, la conclusión es que la primera operación realizada en la corriente 
siempre está abierta y la última está cerrada. El programa, en efecto, es libre de manipular el flujo entre estos 
dos eventos y manejar el archivo asociado. Esta libertad está limitada, por supuesto, por las características físicas 
del expediente y la forma en que se ha abierto el expediente.

La diferencia principal y más llamativa entre Sistemas Operativos es que debe usar dos separadores diferentes para 
los nombres de los directorios: \ en Windows y / en Unix/Linux. Si estamos en otra carpeta ( si no solo necesitamos el 
nombre y extensión del archivo) podemos especificar la misma  En windows, como en python \ es un carácter especial 
tenemos que poner otro \ para salvarlo e indicar que no es un carácter especial por eso es 'c:\\ (en la 1º) en mac o 
linux como es / no hace falta.

Afortunadamente, también hay una solución más. Python es lo suficientemente inteligente como para poder convertir 
barras diagonales en barras diagonales inversas cada vez que descubre que el sistema operativo lo requiere. Esto 
significa que cualquiera de las siguientes asignaciones funcionará en windows:
nombre = "/dir/archivo"
nombre = "c:/dir/archivo"

Además, los nombres de archivo del sistema Unix/Linux distinguen entre mayúsculas y minúsculas. Los sistemas Windows 
almacenan las mayúsculas y minúsculas de las letras utilizadas en el nombre del archivo, pero no distinguen entre 
mayúsculas y minúsculas en absoluto. Esto significa que estas dos cadenas: ThisIsTheNameOfTheFile y 
thisisthenameofthefile describen dos archivos diferentes en los sistemas Unix/Linux, pero tienen el mismo nombre para 
un solo archivo en los sistemas Windows.

La apertura de la transmisión no solo está asociada con el archivo, sino que también debe declarar la forma en que se 
procesará la transmisión. Esta declaración se denomina modo abierto. Si la apertura es exitosa, el programa podrá 
realizar solo las operaciones que sean consistentes con el modo de apertura declarado.

Hay dos operaciones básicas realizadas en el flujo:
- leer del flujo: las partes de los datos se recuperan del archivo y se colocan en un área de memoria administrada 
por el programa (por ejemplo, una variable);
- escribir en el flujo: las porciones de los datos de la memoria (por ejemplo, una variable) se transfieren al archivo.
Hay tres modos básicos que se utilizan para abrir la secuencia:
+modo de lectura: una secuencia abierta en este modo solo permite operaciones de lectura; intentar escribir en la 
transmisión causará una excepción (la excepción se llama UnsupportedOperation, que hereda OSError y ValueError, 
y proviene del módulo io);
+modo de escritura: una secuencia abierta en este modo solo permite operaciones de escritura; intentar leer la 
transmisión causará la excepción mencionada anteriormente;
+modo de actualización: una secuencia abierta en este modo permite escrituras y lecturas.

Identificadores de archivos
Python asume que cada archivo está oculto detrás de un objeto de una clase adecuada. diferentes archivos pueden requerir
diferentes conjuntos de operaciones y comportarse de diferentes maneras. Un objeto de una clase adecuada se crea cuando
abre el archivo y lo aniquila en el momento de cerrarlo. Entre estos dos eventos, puede usar el objeto para 
especificar qué operaciones se deben realizar en una transmisión en particular. Las operaciones que puede usar están 
impuestas por la forma en que abrió el archivo.

En general, el objeto proviene de
IOBase
    - RawIOBase
    - BufferIOBase
    - TextoIOBase

nunca usas constructores para dar vida a estos objetos. La única forma de obtenerlos es invocando la función llamada 
open(). La función analiza los argumentos que ha proporcionado y crea automáticamente el objeto requerido. Si desea 
deshacerse del objeto, invoque el método llamado close(). La invocación cortará la conexión con el objeto y el archivo 
y eliminará el objeto.

como tipos de identificadores de archivos, tenemos de texto ( secuencia de caracteres)  y  binario (secuencias de bits 
imágenes, videio, bbdd, por ejemplo)

Debido a que los archivos binarios no contienen líneas, las lecturas y escrituras se relacionan con porciones de datos 
de cualquier tamaño. Por lo tanto, los datos se leen/escriben byte a byte, o bloque a bloque, donde el tamaño del bloque
suele oscilar entre uno y un valor elegido arbitrariamente.

Luego viene un problema sutil. En los sistemas Unix/Linux, los extremos de línea están marcados por un solo carácter 
llamado LF (código ASCII 10) designado en los programas de Python como \n. Otros sistemas operativos, especialmente 
los derivados del sistema prehistórico CP/M (que también se aplica a los sistemas de la familia Windows) usan una 
convención diferente: el final de la línea está marcado por un par de caracteres, CR y LF (códigos ASCII 13 y 10 ) 
que se puede codificar como \r\n.

Si crea un programa encargado de procesar un archivo de texto y está escrito para Windows, puede reconocer los extremos 
de las líneas al encontrar los caracteres \r\n, pero el mismo programa que se ejecuta en un entorno Unix/Linux será 
completamente inútil, y viceversa: el programa escrito para sistemas Unix/Linux puede ser inútil en Windows.

Dado que los problemas de portabilidad eran (y siguen siendo) muy serios, se tomó la decisión de resolver 
definitivamente el problema de una manera que no atraiga la atención del desarrollador. Se hizo a nivel de clases, que 
son las encargadas de leer y escribir los caracteres hacia y desde la corriente. Funciona de la siguiente manera: cuando
la secuencia está abierta y se recomienda que los datos en el archivo asociado se procesarán como texto (o no existe 
tal aviso), se cambia al modo de texto;

durante la lectura/escritura de líneas desde/hacia el archivo asociado, no ocurre nada especial en el entorno Unix, 
pero cuando se realizan las mismas operaciones en el entorno Windows, ocurre un proceso llamado traducción de caracteres
de nueva línea: cuando lee una línea del archivo, cada par de caracteres \r\n se reemplaza con un solo carácter \n, y 
viceversa; durante las operaciones de escritura, cada carácter \n se reemplaza con un par de caracteres \r\n;

el mecanismo es completamente transparente para el programa, que puede escribirse como si estuviera diseñado para 
procesar archivos de texto Unix/Linux únicamente; el código fuente ejecutado en un entorno Windows también funcionará 
correctamente; cuando la secuencia está abierta y se recomienda hacerlo, su contenido se toma tal cual, sin ninguna 
conversión; no se agregan ni se omiten bytes.

La apertura del flujo se realiza mediante una función que se puede invocar de la siguiente manera:

'''
archivo = open('prueba.txt', 'w', encoding=None)
'''

- el nombre de la función (open) habla por sí solo; si la apertura tiene éxito, la función devuelve un objeto de flujo; 
de lo contrario, se genera una excepción (por ejemplo, FileNotFoundError si el archivo que va a leer no existe);
- el primer parámetro de la función (archivo) especifica el nombre del archivo que se asociará con el flujo;
- el segundo parámetro (modo) especifica el modo abierto utilizado para la transmisión; es una cadena llena de una 
secuencia de caracteres, y cada uno de ellos tiene su propio significado especial;
- el tercer parámetro (codificación) especifica el tipo de codificación (por ejemplo, UTF-8 cuando se trabaja con 
archivos de texto)

la apertura debe ser la primera operación realizada en la corriente.
Nota: los argumentos de modo y codificación pueden omitirse; entonces se asumen sus valores predeterminados. 
El modo de apertura predeterminado es lectura en modo texto, mientras que la codificación predeterminada depende de la 
plataforma utilizada.

- "r" leer, abre archivo para leer, falla si no existe
- "r+" leer o escribir información, el archivo asociado con la secuencia debe existir y debe poder escribirse.
- "a" añadir, crea el archivo si no existe, si existe, el cabezal de grabación virtual se establecerá al final del 
   archivo (el archivo permanece intacto).
- "w" Escribir, abre el fichero para escritura, crea el fichero si no existe, si existe, será truncado a la longitud 
   de cero (borrado)
- "w+" Escribir o leer información, archivo asociado con la transmisión no necesita existir; si no existe, se creará; 
   el contenido anterior del archivo permanece intacto
- "x" Crear, crea un archivo específico, falla si existe

* Además puedes especificar si es manejado como binario o texto: Si hay una letra 'b' al final de la cadena de modo, 
significa que la secuencia se abrirá en modo binario. Si la cadena de modo termina con una letra 't', la secuencia se 
abre en el modo de texto, para todos los modos, por ejemplo rt-rb, w+t-w+b'''

try:
    # OPEN **** puede abrir un archivo nuevo (si no existe) o existente y puede escribir en él, o leer
    archivo = open('prueba.txt', 'w', encoding='utf8')  # encoding ='utf8' hace que permita acentos
    # WRITE **** escribir en un archivo ya abierto claro
    archivo.write('Agregamos información al archivo\n')
    archivo.write('EEEE')
    # lo está sobreescribiendo si lo hacemos varias veces
    # al no existir lo va a crear | Al no especificar ruta lo crea en la de por defecto
except Exception as e:
    print(e)
finally:
    # la función no devuelve nada, pero genera la excepción IOError en caso de error;
    archivo.close()  # siempre debe cerrarse, después de cerrar falla al escribir claro
    print('Fin del archivo')

try:
    archivo = open('prueba.txt', 'r', encoding='utf8')
    # print(archivo.read())   # al leerlo el recorremos el archivo,
    # con lo que se encontraría apuntando al final del mismo
    # leer un archivo de un terabyte con este método puede dañar su sistema operativo.

    # leer algunos caracteres ***
    # print(archivo.read(5))
    # print(archivo.read(3))  # siguientes 3 sigue recorriendo ( en otra línea)

    # leer una línea completa de texto del archivo y la devuelve como una cadena en caso de éxito. De lo contrario,
    # devuelve una cadena vacía.
    # print(archivo.readline())

    # Iterar archivo
    # for linea in archivo:
    #   print(linea)

    # leer todas las líneas e introducirlo en una lista
    # print(archivo.readlines())
    # leer una linea determinada
    # print(archivo.readlines()[1])

    # abrimos un archivo nuevo para añadir información de uno en otro
    try:
        archivo2 = open('copia.txt', 'a', encoding='utf8')  # Abrimos
        archivo2.write(archivo.read())  # escribimos lo que leemos
        '''puede usar el mismo método para escribir en el flujo stderr, pero no intente abrirlo, ya que siempre está 
        abierto implícitamente.
        sys.stderr.write("Mensaje de error")'''
        print('fin copia')

    except Exception as e:  # Excepción en archivo 2
        print(e)

    print('fin lectura')

except Exception as e:  # Excepción en archivo lectura
    print(e)

finally:

    archivo2.close()
    archivo.close()

'''
Cuando comienza nuestro programa, las tres corrientes ya están abiertas y no requieren preparativos adicionales. Además,
su programa puede usar estos flujos explícitamente si tiene cuidado de importar el módulo sys:
'''
import sys
'''
porque ahí es donde se coloca la declaración de los tres flujos. Los nombres de estas corrientes son:

sys.stdin
el flujo stdin normalmente está asociado con el teclado, preabierto para lectura y considerado como la fuente de datos 
principal para los programas en ejecución; la conocida función input() lee datos de stdin por defecto.

sys.stdout
el flujo de salida estándar normalmente está asociado con la pantalla, preabierto para escritura, considerado como el 
objetivo principal para la salida de datos por parte del programa en ejecución;
la conocida función print() envía los datos al flujo estándar.

sys.stderr
el flujo stderr normalmente está asociado con la pantalla, preabierta para escribir, considerada como el lugar principal
donde el programa en ejecución debe enviar información sobre los errores encontrados durante su trabajo;

la separación de stdout (resultados útiles que produce el programa) de los stderr (mensajes de error, innegablemente 
útiles pero que no dan resultados) da la posibilidad de redirigir estos dos tipos de información a los diferentes 
objetivos.


El objeto IOError está equipado con una propiedad llamada errno (el nombre proviene de la frase número de error) y 
puede acceder a ella de la siguiente manera:
'''
from os import strerror

try:
    cnt = 0
    s = open('text.txt', "rt")  # leer texto
    ch = s.read(1)
    while ch != '':
        print(ch, end='')
        cnt += 1
        ch = s.read(1)
    s.close()
    print("\n\nCharacters in file:", cnt)
except IOError as e:
    print("I/O error occurred: ", strerror(e.errno))

'''
Echemos un vistazo a algunas constantes seleccionadas útiles para detectar errores de transmisión:

- errno.EACCES → Permiso denegado , intenta, por ejemplo, abrir un archivo con el atributo densolo lectura para escribir
- errno.EBADF → Número de archivo erróneo, intenta, por ejemplo, operar con una transmisión sin abrir.
- errno.EEXIST → El archivo existe,  intentas, por ejemplo, renombrar un archivo con su nombre anterior.
- errno.EFBIG → Archivo demasiado grande ( max permitido SO)
- errno.EISDIR → Es un directorio, intenta tratar un nombre de directorio como el nombre de un archivo ordinario.
- errno.EMFILE → Demasiados archivos abiertos ( simultaneamente intenta abrir max permitido SO)
- errno.ENOENT → No existe tal archivo o directorio
- errno.ENOSPC → No queda espacio en el dispositivo

hay una función que puede simplificar drásticamente el código de manejo de errores. Su nombre es strerror(), y 
proviene del módulo os y espera solo un argumento: un número de error. Su función es simple: proporciona un número de 
error y obtiene una cadena que describe el significado del error, si pasa un código de error inexistente (un número que
no está vinculado a ningún error real), la función generará la excepción ValueError.

'''
from os import strerror

try:
    s = open("c:/users/user/Desktop/file.txt", "rt")
    # Actual processing goes here.
    s.close()
except Exception as exc:
    print("The file could not be opened:", strerror(exc.errno))

try:
    fo = open('newtext.txt', 'wt')  # A new file (newtext.txt) is created.
    for i in range(10):
        s = "line #" + str(i + 1) + "\n"
        for ch in s:
            fo.write(ch)  # line #1 \n line #2 ...
    fo.close()
except IOError as e:
    print("I/O error occurred: ", strerror(e.errno))

'''
Un rasgo muy interesante del objeto devuelto por la función open() en modo texto. El objeto es una instancia de la clase
iterable. El protocolo de iteración definido para el objeto de archivo es muy simple: su método __next__ simplemente 
devuelve la siguiente línea leída del archivo.Además, puede esperar que el objeto invoque automáticamente close() cuando
cualquiera de las lecturas del archivo llegue al final del archivo.
'''
from os import strerror

try:
    ccnt = lcnt = 0
    for line in open('text.txt', 'rt'):
        lcnt += 1
        for ch in line:
            print(ch, end='')
            ccnt += 1
    print("\n\nCharacters in file:", ccnt)
    print("Lines in file:     ", lcnt)
except IOError as e:
    print("I/O error occurred: ", strerror(e.errno))


# ******** bytearray ARCHIVOS BINARIOS
'''
Los datos amorfos son datos que no tienen forma o forma específica, son solo una serie de bytes, por ejemplo, gráficos 
de mapa de bits. Lo más importante de esto es que en el lugar donde tenemos contacto con los datos, no podemos, o 
simplemente no queremos, saber nada al respecto. Los datos amorfos no se pueden almacenar utilizando ninguno de los 
medios presentados anteriormente: no son cadenas ni listas. Debe haber un contenedor especial capaz de manejar dichos 
datos.

Python tiene más de un contenedor de este tipo, uno de ellos es un bytearray de nombre de clase especializado, 
como sugiere el nombre, es una matriz que contiene bytes (amorfos). Si desea tener un contenedor de este tipo, por 
ejemplo, para leer una imagen de mapa de bits y procesarla de alguna manera, debe crearlo explícitamente, utilizando 
uno de los constructores disponibles.

'''
datos = bytearray(10)  # crea un objeto bytearray capaz de almacenar diez bytes, el constructor llena la matriz con 0.
'''
Los bytearrays se parecen a las listas en muchos aspectos. Por ejemplo, son mutables, son un sujeto de la función len() 
y puede acceder a cualquiera de sus elementos mediante la indexación convencional.Hay una limitación importante: no debe
establecer ningún elemento de matriz de bytes con un valor que no sea un número entero (la violación de esta regla 
provocará una excepción TypeError) y no puede asignar un valor que no provenga del rango de 0 a 255 inclusive (a menos 
que desee provocar una excepción ValueError). Puede tratar cualquier elemento de matriz de bytes como valores enteros, 
al igual que en el ejemplo del editor.
'''
data = bytearray(10)

for i in range(len(data)):
    data[i] = 10 - i

for b in data:
    print(hex(b))  # hex convierte a hexadecimal

'''
escribir una matriz de bytes en un archivo binario
primero, inicializamos bytearray con valores posteriores a partir de 10; si desea que el contenido del archivo sea 
claramente legible, reemplace 10 con algo como ord('a') - esto producirá bytes que contienen valores correspondientes 
a la parte alfabética del código ASCII (no crea que hará que el archivo sea un texto archivo: sigue siendo binario, 
ya que se creó con un indicador wb); luego, creamos el archivo usando la función open() - la única diferencia en 
comparación con las variantes anteriores es el modo abierto que contiene la bandera b; el método write() toma su 
argumento (bytearray) y lo envía (como un todo) al archivo; la corriente se cierra entonces de forma rutinaria.
El método write() devuelve una cantidad de bytes escritos con éxito. Si los valores difieren de la longitud de los 
argumentos del método, puede anunciar algunos errores de escritura.
'''
from os import strerror

data = bytearray(10)

for i in range(len(data)):
    data[i] = 10 + i

try:
    bf = open('file.bin', 'wb')
    bf.write(data)
    bf.close()
except IOError as e:
    print("I/O error occurred:", strerror(e.errno))
'''
Cómo leer bytes de un flujo
La lectura de un archivo binario requiere el uso de un nombre de método especializado readinto(), ya que el método no 
crea un nuevo objeto de matriz de bytes, sino que llena uno creado previamente con los valores tomados del archivo 
binario.
- el método devuelve el número de bytes leídos con éxito;
- el método intenta llenar todo el espacio disponible dentro de su argumento; si hay más datos en el archivo que espacio
 en el argumento, la operación de lectura se detendrá antes del final del archivo; de lo contrario, el resultado del 
 método puede indicar que la matriz de bytes solo se ha llenado fragmentariamente (el resultado también lo mostrará, y 
 la parte de la matriz que no está siendo utilizada por los contenidos recién leídos permanece intacta)
'''
from os import strerror
data = bytearray(10)

try:
    bf = open('file.bin', 'rb')  # abrimos el archivo con el modo descrito como rb;
    bf.readinto(data)  # leemos su contenido en la matriz de bytes denominada data, de diez bytes de tamaño
    bf.close()

    for b in data:
        print(hex(b), end=' ')  # imprimimos el contenido de la matriz de bytes: 0xa 0xb 0xc 0xd 0xe 0xf 0x10 0x11 0x12
        # 0x13
except IOError as e:
    print("I/O error occurred:", strerror(e.errno))

'''
Cómo leer bytes de un flujo
El método denominado read() ofrece una forma alternativa de leer el contenido de un archivo binario. Invocado sin 
argumentos, intenta leer todo el contenido del archivo en la memoria, convirtiéndolo en parte de un objeto recién creado
de la clase bytes. Esta clase tiene algunas similitudes con bytearray, con la excepción de una diferencia significativa:
es inmutable. Afortunadamente, no hay obstáculos para crear una matriz de bytes tomando su valor inicial directamente 
del objeto de bytes, como aquí:'''
from os import strerror

try:
    bf = open('file.bin', 'rb')
    data = bytearray(bf.read())  # en caso de meter un entero a read() especifica el número máximo de bytes a ser leídos
                                 # si read(5) lee 5 bytes, los siguientes 5 seguirán esperando ser leídos
    bf.close()

    for b in data:
        print(hex(b), end=' ')

except IOError as e:
    print("I/O error occurred:", strerror(e.errno))

'''
cuidado: no utilizar este tipo de lectura si no se está seguro de que el contenido del archivo se ajuste a la memoria 
disponible.
'''

# ++++++ COPIA DE FICHERO BINARIO +++++++++
from os import strerror

srcname = input("Enter the source file name: ")
try:
    src = open(srcname, 'rb')
except IOError as e:
    print("Cannot open the source file: ", strerror(e.errno))
    exit(e.errno)  # detener la ejecución del programa y pasar el código de finalización al sistema operativo

dstname = input("Enter the destination file name: ")
try:
    dst = open(dstname, 'wb')
except Exception as e:
    print("Cannot create the destination file: ", strerror(e.errno))
    src.close()
    exit(e.errno)

buffer = bytearray(65536)  # prepara memoria para pasar la información
# un buffer de 64kb técnicamente un buffer más grande copia más rapido los elementos Actualmente siempre hay un límite
# en el que no aumenta el rendimiento
total = 0.  # contador de bytes copiados
try:
    readin = src.readinto(buffer)  # intenta llenar el búfer por primera vez;
    while readin > 0:  # mientras no recibas bytes en el buffer
        written = dst.write(buffer[:readin])  # escribe el contenido del buffer en el fichero de destino
        total += written  # actualiza el contador
        readin = src.readinto(buffer)  # lee el siguiente trozo al buffer
except IOError as e:
    print("Cannot create the destination file: ", strerror(e.errno))
    exit(e.errno)

print(total, 'byte(s) succesfully written')
src.close()
dst.close()

# +++++++ contador tipo diccionario de carácteres de un fichero ++++++
from os import strerror

myDictionary = {}

try:
    name = input('Please enter the file name: ')  # pide el nombre del fichero
    for line in open(name, 'rt', encoding='utf-8'):  # lo abre con codificación utf-8 y recorre las líneas
        for char in line:   # para cada línea del fichero
            if char.isalpha():  # si son letras
                try:
                    myDictionary[char.lower()] += 1  # intenta aumentar el contador del carácter en el diccionario
                except KeyError:
                    myDictionary[char.lower()] = 1   # mete el carácter en el diccionario
            else:
                continue
except IOError as e:
    print("Unable to read file: ", strerror(e.errno))

for key in sorted(myDictionary.keys()):
    print(key, '->', myDictionary.get(key))  # imprime el contenido del diccionario

myChars = sorted(myDictionary.keys(), key=lambda x: myDictionary[x], reverse=True)
# la clave para ordenar es la del diccionario, y lo ordena al revés para que sea desc

try:
    with open(name + '.hist', 'w') as file:  # abre el fichero de escritura
        for char in myChars:
            file.write(char + ' -> ' + str(myDictionary[char]) + '\n')
except IOError as e:
    print('IO Error: ', strerror(e.errno))


# ++++++++ Ejercicio Lectura de fichero de notas ++++++++

from os import strerror


class StudentsDataError(Exception):
    def __init__(self, message=''):
        super().__init__(message)


class BadLineError(StudentsDataError):
    def __init__(self, message=''):
        super().__init__(message)


class FileEmptyError(StudentsDataError):
    def __init__(self, message=''):
        super().__init__(message)


myStudents = {}

try:
    name = input("Please Enter the file containing Students' data: ")
    with open(name, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        if len(lines) == 0:
            raise FileEmptyError('Empty File Error')
        for i in range(len(lines)):
            line = lines[i]
            columns = line.split()

            if len(columns) != 3:
                raise BadLineError(f'BadLineError in Line #{i + 1}: {line}')

            try:
                student = columns[0] + ' ' + columns[1]
                marks = float(columns[2])
            except ValueError:
                raise BadLineError(f'BadLineError in Line #{i + 1}: {line}')
            try:
                myStudents[student] += marks
            except KeyError:
                myStudents[student] = marks

    for student in sorted(myStudents.keys()):
        print(f"{student}\t\t\t{myStudents[student]}")

except IOError as e:
    print('IO Error: ', strerror(e.errno))
except (BadLineError, FileEmptyError, StudentsDataError) as e:
    print(e)

# ********  Archivos  con with
'''
existe una sintaxis simplificada que automáticamente va a abrir y a cerrar nuestro archivo automáticamente sin tener que
cerrarlo después se ejecutan de forma dinámica con __enter__ ( para abrirlo ) y con __exit__ para cerrarlo. Se pueden 
editar sus métodos'''

with open('prueba.txt', 'w+', encoding='utf8') as archivo:
    print(archivo.write('lolito'))

# Clase de manejo de archivos #######
'''# tiene que implementar __enter__ y __exit__ para considerarse de manejo de archivos heredan de object no hay que 
heredar de nada más aparte de abrir y cerrar archivos podríamos usarlo para abrir o cerrar otros recursos como 
conexiones a BBDD'''


class Manejo_archivos:
    def __init__(self, nombre):
        self.nombre = nombre

    ''' estamos encapsulando el código en el método enter que se llamará automáticamente al momento de abrir el recurso
    con utilizar with se manda a llamar enter y al dejar de ejecutar se llama a exit'''
    def __enter__(self):  # obtener
        print('Obtenemos el recurso'.center(50, '#'))
        self.nombre = open(self.nombre, 'w', encoding='utf8')
        return self.nombre  # Devolvemos el objeto si no no irá

    '''Recibe más parámetros si ocurre una excepción podemos recibir el tipo, su valor y la traza que es el texto del 
    mismo no son obligatorios pero si incluirlos'''
    def __exit__(self, tipo_excepción, valor_excepción, traza_error):  # cerrar
        print('cerramos el recurso'.center(50, '#'))
        # preguntamos si el atributo de nombre está apuntando a algún objeto lo que querría decir que está abierto
        if self.nombre:
            # si está abierto lo cerramos
            self.nombre.close()


with Manejo_archivos('prueba.txt') as archivo:
    print(archivo.write('lolito'))


# ++++++ Ejercicio catálogo películas +++++++++
import os
#Dominio


class Pelicula:
    def __init__(self, nombre):
        self.__nombre = str(nombre)

    @property
    def nombre(self):
        return self.__nombre

    @nombre.setter
    def nombre(self, nombre):
        self.__nombre = nombre

    def __str__(self):
        return f'Película {self.nombre}'


# Servicio
class CatalogoPeliculas:

    ruta_archivo = 'Catalogo.txt'

    @classmethod
    def agregar_pelicula(cls, pelicula):
        with open(cls.ruta_archivo, 'a', encoding='utf8') as archivo:
            archivo.write(f'{pelicula.nombre}\n')

    @classmethod
    def listar_pelicula(cls):
        with open(cls.ruta_archivo, 'r+', encoding='utf8') as archivo:
            print('Catálogo de Peliculas'.center(50, '-'))
            print(archivo.read())

    @classmethod
    def eliminar(cls):
        os.remove(cls.ruta_archivo)
        print(f'Archivo eliminado: {cls.ruta_archivo}')


# test
# from dominio.Pelicula import Pelidula
# from servicio.catalogo_peliculas import CatalogoPeliculas

mensaje = f"""
Menú:
1. Agregar película
2. Listar películas
3. Eliminar catálogo de películas
4. Salir
"""
opcion = None
while opcion != 4:
    try:
        print(mensaje)
        opcion = int(input('Escribe tu opción (1-4): '))
        """switch = {
            1: CatalogoPeliculas().agregar_pelicula(Pelicula('lol')),
            2: CatalogoPeliculas().listar_pelicula(),
            3: CatalogoPeliculas().eliminar(),
        }
        switch.get(opcion, 'Opción no válida')"""
        if opcion == 1:
            nombre_pelicula = input('Proporciona el nombre de la película: ')
            pelicula = Pelicula(nombre_pelicula)
            CatalogoPeliculas().agregar_pelicula(pelicula)
        elif opcion == 2:
            CatalogoPeliculas().listar_pelicula()
        elif opcion == 3:
            CatalogoPeliculas().eliminar()
    except Exception as e:
        print(f'Ocurrió un error {e}')
        opcion = None


def number_to_string(argument):
    match argument:
        case 0:
            return "zero"
        case 1:
            return "one"
        case 2:
            return "two"
        case default:
            return "something"


number_to_string(4)

# ++++++++  LABORATORIO Mundo PC ##########
# diagrama uml en
# https://www.udemy.com/course/universidad-python-desde-cero-hasta-experto-django-flask-rest-web/learn/lecture/26667098#overview
# _ protegido (#)
# __privado
# orden que contiene computadoras, las cual tiene monitor telcado
# y ratón ( los dos últimos pertenecen a dispositivo de entrada)

class DispositivoEntrada:  # RESPONSABILIDAD crear objetos de tipo DispositivoEntrada
    def __init__(self, tipo_entrada, marca):
        self._tipo_entrada = tipo_entrada
        self._marca = marca

    def __str__(self):
        return f'Marca: {self.marca}, Tipo de entrada: {self.tipo_entrada} '

    @property
    def tipo_entrada(self):
        return self._tipo_entrada

    @property
    def marca(self):
        return self._marca

    @tipo_entrada.setter
    def tipoEntrada(self, tipo_entrada):
        self._tipo_entrada = tipo_entrada

    @marca.setter
    def marca(self, marca):
        self._marca = marca


class Raton(DispositivoEntrada):  # RESPONSABILIDAD crear objetos de tipo ratón
    __contador_raton = 0

    def __init__(self, tipo, marca):
        super().__init__(tipo, marca)
        self.__id_raton = Raton.contador_raton()

    @classmethod
    def contador_raton(cls):
        cls.__contador_raton += 1
        return cls.__contador_raton

    @property
    def id_raton(self):
        return self.__id_raton

    def __str__(self):
        return f'Ratón: ID {self.id_raton}, {super().__str__()} '


class Teclado(DispositivoEntrada):
    __contador_teclado = 0

    def __init__(self, tipo, marca):
        super().__init__(tipo, marca)
        self.__id_teclado = Teclado.contador_teclado()

    @classmethod
    def contador_teclado(cls):
        cls.__contador_teclado += 1
        return cls.__contador_teclado

    @property
    def id_teclado(self):
        return self.__id_teclado

    def __str__(self):
        return f'Teclado: ID {self.id_teclado}, {super().__str__()}'


class Monitor:  # RESPONSABILIDAD crear objetos de tipo Monitor
    __contador_monitores = 0

    def __init__(self, marca, tamaño):
        self.__id_monitor = Monitor.contador_monitores()
        self.__marca = marca
        self.__tamaño = tamaño

    @classmethod
    def contador_monitores(cls):
        cls.__contador_monitores += 1
        return cls.__contador_monitores

    @property
    def marca(self):
        return self.__marca

    @property
    def tamaño(self):
        return self.__tamaño

    @property
    def id_monitor(self):
        return self.__id_monitor

    @marca.setter
    def marca(self, marca):
        self.__marca = marca

    @tamaño.setter
    def tamaño(self, tamaño):
        self.tamaño = tamaño

    def __str__(self):
        return f'Monitor: ID {self.id_monitor}, de marca: {self.marca}  y {self.tamaño} pulgadas'


class Ordenador:
    __contador_ordenador = 0

    def __init__(self, nombre, monitor, teclado, raton):
        self.__id_ordenador = Ordenador.contador_ordenador()
        self.__nombre = nombre
        self.__monitor = monitor
        self.__teclado = teclado
        self.__raton = raton

    @classmethod
    def contador_ordenador(cls):
        cls.__contador_ordenador += 1
        return cls.__contador_ordenador

    @property
    def id_ordenador(self):
        return self.__id_ordenador

    @property
    def nombre(self):
        return self.__nombre

    @property
    def monitor(self):
        return self.__monitor

    @property
    def teclado(self):
        return self.__teclado

    @property
    def raton(self):
        return self.__raton

    @nombre.setter
    def nombre(self, nombre):
        self.__nombre = nombre

    @teclado.setter
    def teclado(self, teclado):
        self.__teclado = teclado

    @raton.setter
    def raton(self, raton):
        self.__raton = raton

    def __str__(self):
        return f'''Ordenador {self.id_ordenador}
    Nombre: {self.nombre}
    Monitor: {self.monitor}
    Teclado: {self.teclado}
    Ratón: {self.raton}
'''

    def imprime_ordenador(self):
        return self.__str__()


class Orden:
    __contador_ordenes = 0

    def __init__(self, ordenadores):
        self.__id_orden = Orden.contador_ordendes()
        self.__ordenadores = list(ordenadores)

    @classmethod
    def contador_ordendes(cls):
        cls.__contador_ordenes += 1
        return cls.__contador_ordenes

    @property
    def id_orden(self):
        return self.__id_orden

    @property
    def ordenadores(self):
        return self.__ordenadores

    @ordenadores.setter
    def ordenadores(self, ordenadores):
        self.__ordenadores = ordenadores

    def agregar_ordenador(self, ordenador):
        self.__ordenadores.append(ordenador)

    def str_ordenadores(self):
        lista = ''
        for ordenador in self.__ordenadores:
            lista += ordenador.imprime_ordenador() + '\n'
        return lista

    def __str__(self):
        return f'Orden {self.id_orden}\n{self.str_ordenadores()}'


raton1 = Raton('USB', 'Racer')
telcado1 = Teclado('Calbe', 'Razer')
monitor1 = Monitor('HP', 27)
ordenador1 = Ordenador('Dell', monitor1, telcado1, raton1)
raton2 = Raton('Bluetooth', 'Racer')
telcado2 = Teclado('Bluetooth', 'Razer')
monitor2 = Monitor('Samsung', 32)
ordenador2 = Ordenador('Mac', monitor2, telcado2, raton2)
raton3 = Raton('Bluetooth', 'Apple')
telcado3 = Teclado('Bluetooth', 'Apple')
monitor3 = Monitor('Integrado', 16)
ordenador3 = Ordenador('Macboockpro', monitor2, telcado2, raton2)

lista_ordandores = [ordenador1, ordenador2]
orden1 = Orden(lista_ordandores)
orden1.agregar_ordenador(ordenador3)
print(orden1)


# Refactorizando ejercicio
class OrdenRefactor:
    contadorOrdenes = 0

    def __init__(self, *ordenadores):
        self.__idOrden = OrdenRefactor.contarOrden()
        self.__ordenadores = list(ordenadores)

    @property
    def idOrden(self):
        return self.__idOrden

    @property
    def ordenadores(self):
        return self.__ordenadores

    @classmethod
    def contarOrden(cls):
        cls.contadorOrdenes += 1
        return cls.contadorOrdenes

    def agregarOrdenador(self, ordenador):
        self.ordenadores.add(ordenador)

    def __str__(self):
        salida = f'Orden {self.idOrden}'
        for ordenador in self.ordenadores:
            salida = salida + '\n' + f'{ordenador}'
        return salida


ordA = Ordenador('HP', Monitor('thosiba', '25"'), Teclado('Bluetooth', 'HP'), Raton('Bluetooth', 'Razer'))
ordB = Ordenador('MACBOOK', Monitor('Apple pro Res', '16"'), Teclado('Bluetooth', 'apple'), Raton('bluetooth', 'Razer'))
print(OrdenRefactor(ordA, ordB))

# ++++++++ desafío Edube 1: 3 en raya +++++++++++++
"""
la computadora (es decir, su programa) debe jugar el juego usando 'X'; el usuario (p. ej., usted) debe jugar el juego 
usando 'O'; el primer movimiento pertenece a la computadora: siempre pone su primera 'X' en el medio del tablero;
todos los cuadrados están numerados fila por fila comenzando con 1 (consulte la sesión de ejemplo a continuación como 
referencia) el usuario ingresa su movimiento ingresando el número del cuadrado que elige; el número debe ser válido, es decir,
debe ser un número entero, debe ser mayor que 0 y menor que 10, y no puede apuntar a un campo que ya está ocupado;
el programa verifica si el juego ha terminado: hay cuatro veredictos posibles: el juego debe continuar, el juego termina 
en empate, tú ganas o la computadora gana; la computadora responde con su jugada y se repite la verificación;
no implemente ninguna forma de inteligencia artificial: una elección de campo aleatoria hecha por la computadora es lo 
suficientemente buena para el juego. La sesión de ejemplo con el programa puede tener el siguiente aspecto:
Enter your move: 1
+-------+-------+-------+
|       |       |       |
|   O   |   2   |   3   |
|       |       |       |
+-------+-------+-------+
|       |       |       |
|   4   |   X   |   6   |
|       |       |       |
+-------+-------+-------+
|       |       |       |
|   7   |   8   |   9   |
|       |       |       |
+-------+-------+-------+
+-------+-------+-------+
|       |       |       |
|   O   |   X   |   3   |
|       |       |       |
+-------+-------+-------+
|       |       |       |
|   4   |   X   |   6   |
|       |       |       |
+-------+-------+-------+
|       |       |       |
|   7   |   8   |   9   |
|       |       |       |
+-------+-------+-------+
... playing ...
Enter your move: 7
+-------+-------+-------+
|       |       |       |
|   O   |   X   |   X   |
|       |       |       |
+-------+-------+-------+
|       |       |       |
|   O   |   X   |   X   |
|       |       |       |
+-------+-------+-------+
|       |       |       |
|   O   |   O   |   9   |
|       |       |       |
+-------+-------+-------+
You won!

Requisitos
Implemente las siguientes características:

el tablero debe almacenarse como una lista de tres elementos, mientras que cada elemento es otra lista de tres elementos
(las listas internas representan filas) para que se pueda acceder a todos los cuadrados usando la siguiente sintaxis:

tablero[fila][columna]
cada uno de los elementos de la lista interna puede contener 'O', 'X' o un dígito que represente el número del cuadrado
 (tal cuadrado se considera libre)la apariencia del tablero debe ser exactamente igual a la presentada en el ejemplo.
implementar las funciones definidas para usted en el editor.

Se puede dibujar un número entero aleatorio utilizando una función de Python llamada randrange().
El programa de ejemplo a continuación muestra cómo usarlo (el programa imprime diez números aleatorios del 0 al 8).
Nota: la instrucción from-import proporciona acceso a la función randrange definida
dentro de un módulo externo de Python llamado random.

from random import randrange

for i in range(10):
    print(randrange(8))
"""
from random import randrange


def display_board(board):
    # The function accepts one parameter containing the board's current status
    # and prints out a "TABLE" of a matrix[x][y] to the console.
    length = len(board[0])
    middle = ("/" + '\t\t') * (length + 1)
    begend = "+-------" * length + "+"

    for fila in board:
        if fila == board[0]:
            print(begend)
        print(middle)
        pos = 0
        for elemento in fila:
            if pos == 0:
                print("/", elemento, "/", sep="\t", end="")
            else:
                print("", elemento, "/", sep="\t", end="")
            if pos == len(fila) - 1:
                print()
                print(middle)
                print(begend)
            pos += 1


def enter_move(board):
    # The function accepts the board's current status, asks the user about their move,
    # checks the input, and updates the board according to the user's decision.
    correct_move = False
    while not correct_move:
        try:
            move = int(input("Enter your move: "))
            board = moving(board,'O',move)
            correct_move = True
        except TypeError:
            print(f'move not allowed, only integers')
        except Exception as e:
            print(e)
    return board


def make_list_of_free_fields(board):
    # The function browses the board and builds a list of all the free squares;
    # the list consists of tuples, while each tuple is a pair of row and column numbers.
    free = []
    for i_b, f in enumerate(board):
        for i_f, pos in enumerate(f):
            if pos not in ('X', 'O'):
                free.append((i_b, i_f))
    return free


def victory_for(board, sign):
    # The function analyzes the board's status in order to check if
    # the player using 'O's or 'X's has won the game
    end = False
    for m in ['123', '456', '789', '147', '258', '369', '159', '357']:
        if (board[movements.get(int(m[0]))[0]][movements.get(int(m[0]))[1]] == sign
        and board[movements.get(int(m[1]))[0]][movements.get(int(m[1]))[1]] == sign
        and board[movements.get(int(m[2]))[0]][movements.get(int(m[2]))[1]]) == sign:
            end = True
    if end:
        if sign == 'O':
            print('You won!')
        else:
            print('You loose!')
    return end


def draw_move(board, ini=False):
    # The function draws the computer's move and updates the board.
    correct_move = False
    while not correct_move:
        try:
            if ini == True:
                move = 5
            else:
                move = randrange(10)
            board = moving(board, 'X', move)
            correct_move = True
        except TypeError:
            print(f'move not allowed, only integers')
        except Exception as e:
            print(e)
    return board


def moving(board, value, move=None):
    if not move:
        move = randrange(10)
    if not movements.get(move) in make_list_of_free_fields(board):
        raise CustomException('Movement already used')
    else:
        board[:] = [[value if pos == move else pos for pos in f] for f in board]
    display_board(board)
    return board


def init_board(f=3, c=3):
    return [[x + (y * f) for x in range(1, f + 1)] for y in range(c)]


class CustomException(Exception):
    def __init__(self, mensaje):
        self.message = mensaje


if __name__ == '__main__':
    movements ={
                1: (0, 0),
                2: (0, 1),
                3: (0, 2),
                4: (1, 0),
                5: (1, 1),
                6: (1, 2),
                7: (2, 0),
                8: (2, 1),
                9: (2, 2)}
    board = init_board()
    draw_move(board, True)
    end = False
    while not make_list_of_free_fields(board) == []:
        board = enter_move(board)
        end = victory_for(board, 'O')
        if end:
            break
        draw_move(board)
        end = victory_for(board, 'O')
        if end:
            break
    if not end:
        print(f'Tie!')



# +++++++ DESAFIO SNAKE +++++++
class SnakeException(Exception):
    def __init__(self, mensaje):
        self.message = mensaje

# IMPORTS
import itertools as it
#import lib.SnakeException as SnakeException
import math, re

#   CLASS
class SnakeManager:
    #   CLASS VARIABLES
    __numberOfPaths = -1
    __paths = {
        'L': -1,
        'R': 1,
        'D': 1,
        'U': -1,
    }

    def __init__(self, board, snake, depth):
        self.__board = self.initBoard(board)
        self.__snake = self.initSnake(snake)
        self.__depth = self.initdepth(depth)
        self.__snakeOnBorad = self.setSnakeOnBoard()
        self.__tamBoard = board

    @property
    def board(self):
        return self.__board

    @board.setter
    def board(self, board):
        self.__board = board

    @property
    def snake(self):
        return self.__snake

    @snake.setter
    def snake(self, snake):
        self.__snake = snake

    @property
    def depth(self):
        return self.__depth

    @depth.setter
    def depth(self, depth):
        self.__depth = depth

    @property
    def snakeOnBorad(self):
        return self.__snakeOnBorad

    @snakeOnBorad.setter
    def snakeOnBorad(self, snakeOnBorad):
        self.__snakeOnBorad = snakeOnBorad

    @property
    def tamBboard(self):
        return self.__tamBoard

    @classmethod
    def initBoard(cls, board):
        if not 1 <= board[0] <= 10:
            raise SnakeException('The board rows isn\'t between 1 and 10')
        elif not len(board) == 2:
            raise SnakeException('The board length must be 2')
        return board

    def initSnake(self, snake):
        if not 1 <= len(snake) <= 7:
            raise SnakeException('The snake rows isn\'t between 3 and 7')
        elif not len(snake[0]) == 2:
            raise SnakeException('The snake[i] length must be 2')
        return snake

    @classmethod
    def initdepth(cls, depth):
        if not 1 <= depth <= 20:
            raise SnakeException('The depth is not correct')
        return depth

    def cut(self, elem):
        pass

    def setSnakeOnBoard(self, popped=[]):
        board_tmp = [[0 for x in range(self.board[0])] for y in range(self.board[1])]
        #   if the snake is moving
        if self.__numberOfPaths != -1:
            board_tmp2 = self.snakeOnBorad
            # Erase last possition
            if popped != []:
                board_tmp2[popped[1]][popped[0]] = 0
            if board_tmp2[self.snake[0][1]][self.snake[0][0]] == 1:
                raise SnakeException('Intersection  at the movement')
        #   Snake array tour
        i = 0
        for pos in self.snake:
            # Checking that the snake fits in the table
            if i == 0:
                incr = 8
            else:
                incr = 1
            if not pos[0] < self.board[0] or not pos[1] < self.board[1]:
                raise SnakeException(f'The snake don\'t fit on board {self.board} in {pos}'
                                     f'\n does not comply with the Guaranteed Restriction 0 ≤ snake[i][j]< board[j] ')
            # We go through the rows
            for row in range(self.board[0]):
                # We go through the colums
                for colum in range(self.board[1]):
                    # If the current position matches the position of the snake
                    if [row, colum] == [pos[0], pos[1]]:
                        if board_tmp[colum][row] != 0:
                            raise SnakeException('self-intersections')
                        board_tmp[colum][row] = incr
            i +=1
        self.__numberOfPaths += 1
        return board_tmp

    def moveSnakeOnBoard(self, path):
        popped = []
        if path in ('R', 'L'):
            incr = self.__paths.get(path)
            # print(incr)
            if self.snake[0][0] + incr < 0:
                raise SnakeException('The snake goes off the board at the movement')
            self.snake.insert(0, [self.snake[0][0] + incr, self.snake[0][1]])
            popped = [self.snake[-1][0], self.snake[-1][1]]
            self.snake.pop()
        elif path in ('D', 'U'):
            incr = self.__paths.get(path)
            # print(incr)
            if self.snake[0][1] + incr < 0:
                raise SnakeException('The snake goes off the board at the movement')
            self.snake.insert(0, [self.snake[0][0], self.snake[0][1] + incr])
            popped = [self.snake[-1][0], self.snake[-1][1]]
            self.snake.pop()
        # print(self.snake)
        self.snakeOnBorad = self.setSnakeOnBoard(popped)

    def isPossible(self):
        possible = possible = list(it.product('LURD', repeat=self.depth))
        partL = possible[:math.ceil(len(possible)*0.25)]
        partU = possible[math.floor(len(possible)*0.25):math.ceil(len(possible)*0.50)]
        partR = possible[math.floor(len(possible)*0.50):math.ceil(len(possible)*0.75)]
        partD = possible[math.floor(len(possible)*0.75):len(possible)]
        validPath = set()

        condition = str(self.snake[0][0])+str(self.snake[0][1])
        match condition:
            # Edge
            case condition if condition == '00':
                if [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] + 1) in s] != []:
                    # R OFF
                    possible = partD
                elif [s for s in self.snake if str(self.snake[0][0] + 1) + str(self.snake[0][1]) in s] != []:
                    # D OFF
                    possible = partR
                else:
                    possible = partR + partD

            case condition if condition == f'{self.tamBboard[1]-1}0':
                if [s for s in self.snake if str(self.snake[0][0] + 1) + str(self.snake[0][1]) in s] != []:
                    # D OFF
                    possible = partL
                elif [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] - 1) in s] != []:
                    # L OFF
                    possible = partD
                else:
                    possible = partL + partD

            case condition if condition == f'0{self.tamBboard[1]-1}':
                if [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] + 1) in s] != []:
                    # R OFF
                    possible = partD
                elif [s for s in self.snake if str(self.snake[0][0] + 1) + str(self.snake[0][1]) in s] != []:
                    # D OFF
                    possible = partR
                else:
                    possible = partR + partD

            case condition if condition == f'{self.tamBboard[1]-1}{self.tamBboard[0]-1}':
                if [s for s in self.snake if str(self.snake[0][0] - 1) + str(self.snake[0][1]) in s] != []:
                    # U OFF
                    possible = partD
                elif [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] - 1) in s] != []:
                    # L OFF
                    possible = partU
                else:
                    possible = partL + partU

            case condition if condition[0] == '0' and condition != '00':
                if [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] + 1) in s] != []:
                    # R OFF
                    possible = partL + partD
                elif [s for s in self.snake if str(self.snake[0][0] + 1) + str(self.snake[0][1]) in s] != []:
                    # D OFF
                    possible = partL + partR
                elif [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] - 1) in s] != []:
                    # L OFF
                    possible = partR + partD
                else:
                    possible = partL + partR + partD

            case condition if condition[1] == '0' and condition != '00':
                if [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] + 1) in s] != []:
                    # R OFF
                    possible = partU + partD
                elif [s for s in self.snake if str(self.snake[0][0] + 1) + str(self.snake[0][1]) in s] != []:
                    # D OFF
                    possible = partU + partR
                elif [s for s in self.snake if str(self.snake[0][0] - 1) + str(self.snake[0][1]) in s] != []:
                    # U OFF
                    possible = partR + partD
                else:
                    possible = partU + partR + partD

            case condition if condition[0] == f'{self.tamBboard[0]-1}' and condition != f'{self.tamBboard[1]-1}{self.tamBboard[0]-1}':
                if [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] + 1) in s] != []:
                    # R OFF
                    possible = partL + partU
                elif [s for s in self.snake if str(self.snake[0][0] - 1) + str(self.snake[0][1]) in s] != []:
                    # U OFF
                    possible = partL + partR
                elif [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] - 1) in s] != []:
                    # L OFF
                    possible = partU + partR
                else:
                    possible = partL + partU + partR

            case condition if condition[1] == f'{self.tamBboard[1]-1}' and condition != f'{self.tamBboard[1]-1}{self.tamBboard[0]-1}':
                if [s for s in self.snake if str(self.snake[0][0] + 1) + str(self.snake[0][1]) in s] != []:
                    # D OFF
                    possible = partL + partU
                elif [s for s in self.snake if str(self.snake[0][0] - 1) + str(self.snake[0][1]) in s] != []:
                    # U OFF
                    possible = partL + partD
                elif [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] - 1) in s] != []:
                    # L OFF
                    possible = partU + partD
                else:
                    possible = partL + partU + partD
            # Snake body
            case condition if condition not in ('00', f'0{self.tamBboard[1]-1}', f'{self.tamBboard[1]-1}0', f'{self.tamBboard[1]-1}{self.tamBboard[0]-1}'):
                if [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] + 1) in s] != []:
                    # R OFF
                    possible = partL + partU + partD
                elif [s for s in self.snake if str(self.snake[0][0] + 1) + str(self.snake[0][1]) in s] != []:
                    # D OFF
                    possible = partL + partU + partR
                elif [s for s in self.snake if str(self.snake[0][0] - 1) + str(self.snake[0][1]) in s] != []:
                    # U OFF
                    possible = partL + partR + partD
                elif [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] - 1) in s] != []:
                    # L OFF
                    possible = partU + partR + partD



        return possible

    def searchL(self):
        validPath = set()
        possible = self.isPossible()
        for p in possible:
            status, path, loop = self.boundedDepthSearch(p)
            if status == 'ok':
                validPath.add(str(p))
        return len(validPath)

    def boundedDepthSearch(self, loopPath):
        photoBoard = self.snakeOnBorad
        photoSnake = self.photoSnake(self.snake)
        curr = []
        loop = 0
        for p in loopPath:
            try:
                self.moveSnakeOnBoard(p)
                curr.append(p)
                if len(curr) == self.depth:
                    status = 'ok'
            except:
                status = 'ko'
        loop = len(curr)-1
        self.snakeOnBorad = photoBoard
        self.snake = photoSnake
        return status, curr, loop

    def photoSnake(self, state):
        tmp = []
        for s in state:
            tmp.append(s)
        return tmp

    def __str__(self):
        out = ''
        for row in pr.snakeOnBorad:
            out = out + row + '\n'
        return out


if __name__ == '__main__':
    # pr = SnakeManager([4, 3], [[2, 2], [3, 2], [3, 1], [3, 0], [2, 0], [1, 0], [0, 0]], 3)
    pr = SnakeManager([10, 10], [[5,5], [5,4], [4,4], [4,5]], 4)
    # pr = SnakeManager([2, 3], [[0,2], [0,1], [0,0], [1,0], [1,1], [1,2]], 10)
    print(pr.searchL())
