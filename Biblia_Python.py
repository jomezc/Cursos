"""
APUNTES JESÚS GÓMEZ CÁNOVAS SOBRE PYTHON

su código (en ejecución) se encuentra en la parte superior del mismo;
Python (más precisamente, su entorno de ejecución) se encuentra directamente debajo de él;
la siguiente capa de la pirámide está llena con el sistema operativo (sistema operativo):
el entorno de Python proporciona algunas de sus funcionalidades utilizando los servicios del sistema operativo;
Python, aunque muy potente, no es omnipotente: se ve obligado a usar muchos ayudantes si va a procesar archivos
o comunicarse con dispositivos físicos; La capa inferior es el hardware: el procesador (o procesadores),
las interfaces de red, los dispositivos de interfaz humana (ratones, teclados, etc.)
y toda otra maquinaria necesaria para que la computadora funcione;
el sistema operativo sabe cómo conducirlo y usa muchos trucos para conducir todas las partes a un ritmo constante.
"""

print("Texto")  # Imprmir por pantalla

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
# representacion de numeros en otras bases:
print(0o123) #Octal: 0octal ,83  0O or 0o prefix [0..7] range
print(0x123) #Hexadecimal: 0x123 ,291 0x or 0X prefix

# ******** STRING
x = "ju-ja"  # se puede utilizar comillas simples y dobles, pero deben comenzar y terminar igual
print(type(x))
print(11_11_11) # 11_11_11 Python 3.6 has introduced underscores in numeric literals
# Como las variables pueden apuntar cualquier tipo de dato podemos agregar una pista (no las define)
x: str = 'hola'
print(x)
print(type(x))
x: str = 10  # al ser dinámicas y no definirlo podemos igualmente meter un número y funciona
print(x)
print('I\'m Monty Python')  # \ salvar caracteres para especiales como \n o para este ejemplo
# ******** FLOAT
x = 10.5 # Con ponerle el punto ya le estamos declarando una variable de tipo flotante
x = .5
print(x)
print(type(x))
print (3e3) # Exponenete 3 x 108.
print (3E8)
print(0.0000000000000000000001) # el resultado de ejecutarlo es 1e-22 python escoge
# el modo mas económico de representación de números
anything = float(input("Enter a number: "))
something = float(anything) ** 2.0
print(anything, "to the power of 2 is", something)

x = float(input("Enter value for x: "))


# ******** BOOL
x = True  # booleano, eso si respetar la mayúscula # False
print(type(x))
# OJO SABER QUE TODOS LOS TIPOS DE DATOS SON ALMACENADOS POR CLASES EN PYTHON
print(True > False) # True
print(True < False) # False
2 == 2. # True
# Due to the low priority of the == operator, the question shall be treated as equivalent to this one:
# black_sheep == (2 * white_sheep)

# ***********************************
# ********  CADENAS  ############# secuencias inmutables
# ***********************************
'''
  ASCII (utilizado principalmente para codificar el alfabeto latino y algunos de sus derivados) 
y UNICODE (capaz de codificar prácticamente todos los alfabetos que utilizan los humanos).

revisa el código de la letra minúscula a. Esto es 97. Y ahora encuentra la A mayúscula. Su código es 65. 
Ahora calcula la diferencia entre el código de ay A. Es igual a 32. Ese es el código de un espacio.

Unicode asigna caracteres únicos (no ambiguos) (letras, guiones, ideogramas, etc.) a más de un millón de puntos de 
código. Los primeros 128 puntos de código Unicode son idénticos a ASCII, y los primeros 256 puntos de código Unicode 
son idénticos a la página de códigos ISO/IEC 8859-1 (una página de códigos diseñada para idiomas de Europa occidental).

Hay más de un estándar que describe las técnicas utilizadas para implementar Unicode en computadoras y sistemas 
de almacenamiento informático reales. El más general de ellos es UCS-4.
El nombre proviene de Universal Character Set. UCS-4 usa 32 bits (cuatro bytes) para almacenar cada carácter, 
y el código es solo el número único de los puntos de código Unicode. 
Un archivo que contiene texto codificado en UCS-4 puede comenzar con una BOM (marca de orden de bytes):
es una combinación especial no imprimible de bits que anuncian la codificación utilizada  por el contenido de un archivo 
(por ejemplo, UCS-4 o UTF-B).

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

Los /n se cuentan cómo carácter en las ‘’’ ( para multilinea son los 3, 1 solo 1 línea) o si se introducen, 
los vacíos no. + concatenar ( no conmutativo, el orden importa), * replicar n veces ( se pone número). 
variantes abreviadas de los operadores anteriores también son aplicables para cadenas (+= y *=)

Si desea conocer el valor del punto de código ASCII/UNICODE de un carácter específico, puede utilizar una función 
denominada ord() (como en ordinal).

La función necesita una cadena de un carácter como argumento; el incumplimiento de este requisito genera una excepción 
TypeError y devuelve un número que representa el punto de código del argumento.

por el contrario si conoces el valor del punto de código y quieres concer el caracter: chr()
'''
# Demonstrating the ord() function.
char_1 = 'a'
char_11 = 'A'
char_2 = ' '  # space

print(ord(char_1))  # 97
print(ord(char_11))  # 65
print(ord(char_2))  # 32
print(chr(97))  # a
print(chr(65))  # A
print(len("\n\n"))  #2.

print("\"I\'m\"\n\"\"learning\"\"\n\"\"\"Python\"\"\"")
#"I'm"
#""learning""
#"""Python"""
# Cadena (String), concatenar valores, con +
miGrupoFavorito = "ACDC" + " " + "The best rock band"
print("Mi grupo favorito es: " + miGrupoFavorito)  # en PRINT
b = "Mi grupo favorito es:"
print(b + " " + miGrupoFavorito)
# podemos también usar comas que mete automáticamente espacio
print(b, miGrupoFavorito)

n1 = "1"
n2 = "2"
print(n1 + n2)  # Concatenación
n1 = 1
n2 = 2
# Sobrecarga
print(n1 + n2)  # suma
n1 = "1"
n2 = 2
print("Concatenación: ", int(n1) + n2)  # conversión a entero int(), tiene que ser valido
print("Programming","Essentials","in", sep="***", end="...") # Programming***Essentials***in...Python
print("Python")
# En las cadenas de Python, la barra invertida (\) es un carácter especial que anuncia que el siguiente carácter tiene
# un significado diferente, por ejemplo, \n (el carácter de nueva línea) inicia una nueva línea de salida.

# rectágulo
print("+" + 10 * "-" + "+")
print(("|" + " " * 10 + "|\n") * 5, end="")
print("+" + 10 * "-" + "+")

leg_a = float(input("Input first leg length: ")) # float a número en punto flotante
leg_b = float(input("Input second leg length: "))
print("Hypotenuse length is " + str((leg_a**2 + leg_b**2) ** .5)) #2 3 -> 3.605551275463989 str a char
x = int(input("Enter a number: ")) # The user enters 2
print(x * "5")

# como son secuencias se pueden recorrer como las listas ( ejemplos en for) tanto por con índices positivos como
# negativos o rangos así como in o not in, al ser inmutables, pues eso NI del, insert o append se puede
the_string = 'silly walk'

for character in the_string:
    print(character, end=' ') # 's i l l y  w a l k'

print(the_string[-1]) # k
print(the_string[1:3])
print("f" in the_string) #False

# min() encuentra el elemento mínimo de la secuencia pasada como argumento. Hay una condición:
# la secuencia (cadena, lista, no importa) no puede estar vacía, de lo contrario obtendrá una excepción ValueError.
# max el mayor
print(min("aAbByYzZ")) #A , es menor ascii
print(max("aAbByYzZ")) #z , es mayor ascii

# Demonstrating min() - Examples 2 & 3:
t = 'The Knights Who Say "Ni!"'
print('[' + min(t) + ']') # espacio es el 32
print('[' + max(t) + ']') # y

t = [0, 1, 2]
print(min(t)) # 0 menor ascii de los numeros
print(max(t)) # 2

# index()
#Busca la secuencia desde el principio, para encontrar el primer elemento del valor especificado en su argumento.
# Encuentra el elemento mínimo de la secuencia pasada como argumento. Hay una condición:
# la secuencia (cadena, lista, no importa) no puede estar vacía, de lo contrario obtendrá una excepción ValueError.
print("aAbByYzZaA".index("b")) # 2
print("aAbByYzZaA".index("Z")) # 7
print("aAbByYzZaA".index("A")) # 1

# La función list() toma su argumento (una cadena) y crea una nueva lista que contiene todos los caracteres de la
# cadena, uno por elemento de la lista. Nota: no es estrictamente una función de cadena: list()
# puede crear una nueva lista a partir de muchas otras entidades (por ejemplo, de tuplas y diccionarios)
print(list("abcabc")) # ['a', 'b', 'c', 'a', 'b', 'c']

# El método count() cuenta todas las ocurrencias del elemento dentro de la secuencia. La ausencia de tales elementos
# no causa ningún problema.
print("abcabc".count("b"))# 2
print('abcabc'.count("d"))# 0


# El método center() hace una copia de la cadena original,
#tratando de centrarla dentro de un campo de un ancho específico.
# o con un numero de ocurrencias del segundo parámetro
print('[' + 'alpha'.center(10) + ']') #[  alpha   ]
print('[' + 'alpha'.center(10,'*') + ']') #[**alpha***]

#El método endswith() verifica si la cadena dada termina con el argumento especificado y devuelve True o False,
# según el resultado de la verificación.
if "epsilon".endswith("on"):
    print("yes")
else:
    print("no")
# yes

# El método find() es similar a index(), que ya conoce: busca una subcadena y devuelve el índice de la primera
# aparición de esta subcadena, pero:
# es más seguro: no genera un error para un argumento que contiene una subcadena inexistente (devuelve -1 entonces)
# funciona solo con cadenas; no intente aplicarlo a ninguna otra secuencia.
# El segundo argumento especifica el índice en el que se iniciará la búsqueda (no tiene que caber dentro de la cadena).
#el tercer argumento  es el límite superior (no incluido) de la búsqueda

print("Eta".find("ta")) #1
print("Eta".find("mma")) #-1
print('kappa'.find('a', 2)) # 4
print('kappa'.find('a', 1, 4)) # 1
print('kappa'.find('a', 2, 4)) #-1

the_text = """A variation of the ordinary lorem ipsum
text has been used in typesetting since the 1960s"""

fnd = the_text.find('the')
while fnd != -1:
    print(fnd)
    fnd = the_text.find('the', fnd + 1)
    # 15 80

# rfind() hacen casi lo mismo que sus contrapartes (los que no tienen el prefijo r), pero comienzan sus búsquedas
# desde el final de la cadena, no desde el principio (por lo tanto el prefijo r, de derecho).
print("tau tau tau".rfind("ta")) # 8
print("tau tau tau".rfind("ta", 9)) # -1
print("tau tau tau".rfind("ta", 3, 9)) #4

# isalnum() comprueba si la cadena contiene solo dígitos o caracteres alfabéticos (letras) y
# devuelve True o FalseTambién
t = 'Six lambdas' # False
print(t.isalnum())
t = 'ΑβΓδ' # True
print(t.isalnum())
t = '20E1' # True
print(t.isalnum())

# isalpha(), solo letras:
print("Moooo".isalpha()) # True
print('Mu40'.isalpha()) # False

# isdigit() , solo dígitos:
print('2018'.isdigit()) #True
print("Year2019".isdigit()) #False

# islower() solo letras minúsculas:
print("Moooo".islower()) #False
print('moooo'.islower()) #True

# isspace() solo espacios:
print(' \n '.isspace()) # True
print(" ".isspace()) # True
print("mooo mooo mooo".isspace()) # False

# Example 3: Demonstrating the isupper() method:
print("Moooo".isupper()) #False
print('MOOOO'.isupper()) #True

# El capitalize() crea una nueva cadena llena de caracteres,
# si el primer carácter dentro de la cadena es una letra (nota: el primer carácter es un elemento con un índice igual a
# 0, no solo el primer carácter visible), se convertirá a mayúsculas; todas las letras restantes de la cadena se
# convertirán a minúsculas.
# la cadena original no cambia de ninguna manera
# la cadena modificada se devuelve como resultado
print("Alpha".capitalize()) # Alpha
print('ALPHA'.capitalize())# Alpha
print(' Alpha'.capitalize())# alpha
print('123'.capitalize()) # 123
print("αβγδ".capitalize())#Αβγδ

# El método swapcase() crea una nueva cadena intercambiando las mayúsculas y minúsculas de todas las letras dentro
# de la cadena de origen: los caracteres en minúsculas se convierten en mayúsculas y viceversa. El resto de caracteres
# no se tocan
print("I know that I know nothing.".swapcase())  # i KNOW THAT i KNOW NOTHING.

# El método title() realiza una función algo similar: cambia la primera letra de cada palabra a mayúsculas y
# cambia todas las demás a minúsculas
print("I know that I know nothing. Part 1.".title())    # I Know That I Know Nothing. Part 1.

# el método upper() hace una copia de la cadena de origen, reemplaza todas
# las letras minúsculas con sus equivalentes en mayúsculas y devuelve la cadena como resultado.
print("I know that I know nothing. Part 2.".upper())    #I KNOW THAT I KNOW NOTHING. PART 2.

# lower() hace una copia de una cadena de origen, reemplaza todas las letras mayúsculas con sus
# equivalentes en minúsculas y devuelve la cadena como resultado. Una vez más, la cadena de origen permanece intacta.
print("SiGmA=60".lower()) # sigma=60

# ***** join()
# como sugiere su nombre, el método realiza una unión: espera un argumento como una lista; debe asegurarse de que
# todos los elementos de la lista sean cadenas; de lo contrario, el método generará una excepción TypeError;
# todos los elementos de la lista se unirán en una cadena pero...
# ...la cadena desde la que se ha invocado el método se usa como separador, se coloca entre las cadenas;
# la cadena recién creada se devuelve como resultado.
print(",".join(["omicron", "pi", "rho"]))   # omicron,pi,rho
print("abc".join(["omicron", "pi", "rho"])) # omicronabcpiabcrho
print("".join(["omicron", "pi", "rho"])) # omicronpirho

# **** split()
# El método split() hace lo que dice: divide la cadena y crea una lista de todas las subcadenas detectadas.
# El método asume que las subcadenas están delimitadas por espacios en blanco: los espacios no participan en la
# operación y no se copian en la lista resultante.Si la cadena está vacía, la lista resultante también está vacía.
# si le pones otra cosa pues es el separador
print("phi       chi\npsi".split()) # ['phi', 'chi', 'psi']
print("phi       chi\npsi".split('\n')) # ['phi', 'chi', 'psi']

# El método lstrip() sin parámetros devuelve una cadena recién creada formada a partir de la original eliminando
# todos los espacios en blanco INICIALES.
print("[" + " tau ".lstrip() + "]") # [tau ] OJO iniciales
# El método lstrip() de un parámetro hace lo mismo que su versión sin parámetros,
# pero elimina todos los caracteres incluidos en su argumento (una cadena), no solo los espacios en blanco:
print("www.cisco.com".lstrip("w.")) # cisco.com
print("pythoninstitute.org".lstrip(".org")) # pythoninstitute.org

# rstrip() lo mismo pero desde el otro extremo:
print("[" + " upsilon ".rstrip() + "]")
print("cisco.com".rstrip(".com"))

# El método strip() combina los efectos causados por rstrip() y lstrip() - crea una nueva cadena que carece de
# todos los espacios en blanco iniciales y finales (ojo no en medio).
print("[" + "   aleph   ".strip() + "]") # [aleph]
print(".orgpythoninstitute.org".strip(".org")) # pythoninstitute

# El método replace() de dos parámetros devuelve una copia de la cadena original en la que todas las apariciones
# del primer argumento han sido reemplazadas por el segundo argumento.
#  Si el segundo argumento es una cadena vacía, reemplazar en realidad es eliminar la cadena del primer argumento.
#  ¿Qué tipo de magia ocurre si el primer argumento es una cadena vacía?
# La variante replace() de tres parámetros usa el tercer argumento (un número) para limitar el número de reemplazos.
print("www.netacad.com".replace("netacad.com", "pythoninstitute.org")) # www.pythoninstitute.org
print("This is it!".replace("is", "are"))# Thare are it!
print("Apple juice".replace("juice", ""))# Apple
print("This is it!".replace("is", "are", 1))# Thare is it!
print("This is it!".replace("is", "are", 2))# Thare are it!

# El método "startswith()" es un reflejo especular de "endswith()": comprueba si una cadena dada comienza
# con la subcadena especificada.
# Demonstrating the startswith() method:
print("omega".startswith("meg")) # False
print("omega".startswith("om")) # True
print("omega".endswith("meg")) # False
print("omega".endswith("a")) # True
print()

# Ejercicio
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

print(mysplit("To be or not to be, that is the question"))  # ['To', 'be', 'or', 'not', 'to', 'be,', 'that', 'is', 'the', 'question']
print(mysplit("To be or not to be,that is the question"))   # ['To', 'be', 'or', 'not', 'to', 'be,that', 'is', 'the', 'question']
print(mysplit("   "))   # []
print(mysplit(" abc "))     # ['abc']
print(mysplit(""))  # []


# ***********************************
# ********  BOOL  ###############
# ***********************************
#Is it guaranteed that False == 0 and True == 1,
miVariable = True
miVariable2 = False
miVariable3 = 2 < 3  # la comprobación devuelve un valor true o false
print(miVariable, miVariable2, miVariable3)



# ***********************************
# ********  None  ###############
# ***********************************
# son solo dos tipos de circunstancias en las que Ninguno se puede usar de manera segura:
# cuando lo asigna a una variable (o lo devuelve como resultado de una función)
# #cuando lo comparas con una variable para diagnosticar su estado interno. Como aquí:
value = None
if value is None:
    print("Sorry, you don't carry any value")


# ******** IF
if miVariable2:  # LOS DOS PUNTOS SON NECESARIOS
    print("el resultado fue verdadero")
else:
    print("el resultado fue falso")

x, y, z = 5, 10, 8
x, y, z = z, y, x

print(x > z)
print((y - 5) == x)



# ***********************************
# ********  Input ***************
# ***********************************
# Función input para la entrada del usuario
# Input devuelve un string con lo que para realizar operaciones numéricas tenemos que convertir
n = int(input("Escribe el primer numero: "))
m = int(input("Escribe el segundo numero: "))  # ahora si no metemos enteros falla claro
r = n + m
print("El resultado es:", r)

# Ejercicio
dia = int(input('valora tu día del 1 al 10: '))
print("Mi dia estuvo de:", dia)

# ***********************************
# ******** OPERADORES #########
# ***********************************
# SUMA +
opA = 3
opB = 2
suma = opA + opB
print('Resultado de la suma: ', suma)
# Literal precedido de 'f' y {}
print(f'resultado de la  suma: {suma}')

# RESTA -
resta = opA - opB
print(f'Resultado de la resta: {resta}')
print(-4 - 4) #-8
print(4. - 8) #-8.
print(-1.1)#-1.1
# Multiplicación *
mult = opA * opB
print(f'Resultado de la multiplicación: {mult}')
# division normal /
div = opA / opB
print(f'Resultado de la división: {div}')
# division entera ( sin parte fraccionaria) //
division = opA // opB
print(f'resultado division entera: {division}')
# modulo %
mod = opA % opB
print(f'Resultado del modulo: {mod}')
# exponente **
exp = opA ** opB
print(f'Resultado del exponente: {exp}')

# Ejercicio Rectángulo
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

print(9 % 6 % 2) # 1 left-sided binding
print(2 ** 2 ** 3) #256  the exponentiation operator uses right-sided binding.
#unary operator is an operator with only one operand, e.g., -1, or +3.
print((-2 / 4), (2 / 4), (2 // 4), (-2 // 4)) # -0.5 0.5 0 -1

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
string >= number --> falla.'''

a = 4
b = 2
r = (a == b)  # lo del lado derecho se evalúa primero y luego se asigna
print(f'Resultado: {r}')
r = a != b
print(f'Resultado: {r}')
r = a > b  # mayor
r = a < b  # menor
r = a >= b  # menor o igual
r = a <= b  # menor o igual
print(f'Resultado: {r}')

s1 = '12.8'
i = int(s1) # value error aqui porque no podemos pasar el string 12.8 a entero a float si podriamos


'''
En general, Python ofrece dos formas diferentes de ordenar listas.
El primero se implementa como una función llamada sorted().
La función toma un argumento (una lista) y devuelve una nueva lista, llena con los elementos del argumento ordenado. 
La lista original permanece intacta. Mire el código en el editor y ejecútelo. 
El segundo método afecta a la lista en sí: no se crea ninguna lista nueva. La ordenación se realiza in situ mediante 
el método denominado sort().
'''
first_greek = ['omega', 'alpha', 'pi', 'gamma']
first_greek_2 = sorted(first_greek)
print(first_greek) # ['omega', 'alpha', 'pi', 'gamma']
print(first_greek_2) # ['alpha', 'gamma', 'omega', 'pi']

second_greek = ['omega', 'alpha', 'pi', 'gamma']
print(second_greek) # ['omega', 'alpha', 'pi', 'gamma']
second_greek.sort()
print(second_greek)  ['alpha', 'gamma', 'omega', 'pi']

# ***********************************
# ******** Ejemplo prioridad##
# ***********************************
# 1
#------
# x + 1
#    ---
#  x + 1
#      --
#  x +  1
#       -
#       x
print("y =", 1/(x+(1/(x+(1/(x+(1/x)))))))


# ***********************************
# ********conversor millas km
# ***********************************

kilometers = 12.25
miles = 7.38

miles_to_kilometers = miles * 1.61
kilometers_to_miles = kilometers / 1.61

print(miles, "miles is", round(miles_to_kilometers, 2), "kilometers")
print(kilometers, "kilometers is", round(kilometers_to_miles, 2), "miles")

# ***********************************
# ******** ecuación sencilla
# ***********************************
x =  -1
x = float(x)
y = 3*(x**3) - 2*(x**2) + 3*x -1
print("y =", y)

# ***********************************
# ********Algoritmo pa_impar
# ***********************************
n = int(input("Ingresa un numero: "))
if n % 2 == 0:
    print(f'{n} es par')
else:
    print(f'{n} es impar')

# ***********************************
# ******** Algoritmo determinaMayorEdad
# ***********************************
ADULTO = 18
edad = int(input('Dime tu edad: '))
if edad >= ADULTO:
    print(f'tienes {edad}, por ello eres mayor de edad')
else:
    print(f'tienes {edad} por ello eres un cri@')

# ***********************************
# ******** paso de minutos en reloj
# ***********************************
"""
Scenario
Your task is to prepare a simple code able to evaluate the 
end time of a period of time, given as a number of minutes (it could be arbitrarily large). The start time is given as a pair of hours (0..23) and minutes (0..59). 
The result has to be printed to the console.
For example, if an event starts at 12:17 and lasts 59 minutes, 
it will end at 13:16.
Don't worry about any imperfections in your 
code - it's okay if it accepts an invalid time - the most important
thing is that the code produce valid results for valid input data.
Test your code carefully. Hint: using the % operator may be the key to success.
12
17
59
Expected output: 13:16
Sample input:
23
58
642
Expected output: 10:40
"""
hour = int(input("Starting time (hours): "))
mins = int(input("Starting time (minutes): "))
dura = int(input("Event duration (minutes): "))
print(f'{(((mins+dura)//60)+hour)%24}:{(mins+dura)%60}')

# Write your code here.


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

# ley de Morgan
# not (p and q) == (not p) or (not q)
# not (p or q) == (not p) and (not q)

# Logical values vs. single bits
# Logical operators take their arguments as a whole regardless of how many bits they contain.
# The operators are aware only of the value: zero (when all the bits are reset) means False;
# not zero (when at least one bit is set) means True.
#
# The result of their operations is one of these values: False or True.
# This means that this snippet will assign the value True to the j variable if i is not zero;
# otherwise, it will be False.
"""
Operadores bit a bit ( SOLO ENTEROS)
https://www.geeksforgeeks.org/python-bitwise-operators/
Sin embargo, hay cuatro operadores que le permiten manipular bits individuales de datos. Se llaman operadores bit a bit.

Cubren todas las operaciones que mencionamos antes en el contexto lógico y un operador adicional. Este es el operador xor (como en o exclusivo), y se denota como ^ (signo de intercalación).

Aquí están todos ellos:

& (ampersand) - conjunción bit a bit
    Su salida es 1 solo si todas las entradas son 1. Aquí está la tabla de verdad:
    0 AND 0 = 0
    0 AND 1 = 0
    1 AND 0 = 0
    1 AND 1 = 1;
| (bar) - disyunción bit a bit
    El operador OR también es conocido como disyunción lógica. 
    Da como salida 1 siempre que una o más de sus entradas sean 1. Aquí está la tabla de verdad:
    0 OR 0 = 0
    0 OR 1 = 1
    1 OR 0 = 1
    1 OR 1 = 1;

~ (tilde) - negación bit a bit
    lo contrario 0 -> 1 ; 1-> 0;
^ (signo de intercalación) - bit a bit exclusivo o (xor)
    El operador XOR tiene como salida un 1 siempre que las entradas no coincidan,
     lo cual ocurre cuando una de las dos entradas es exclusivamente verdadera. Esto es lo mismo que la suma mod 2. Aquí está la tabla de verdad:
    0 XOR 0 = 0
    0 XOR 1 = 1
    1 XOR 0 = 1
    1 XOR 1 = 0.

por ello:
& requiere exactamente dos 1 para proporcionar 1 como resultado;
| requiere al menos un 1 para proporcionar 1 como resultado;
^ requiere exactamente un 1 para proporcionar 1 como resultado.
x = 4
y = 1

a = x & y
b = x | y
c = ~x  # tricky!
d = x ^ 5
e = x >> 2
f = x << 2

print(a, b, c, d, e, f) # 0 5 -5 1 1 16
"""
# Python program to show
# bitwise operators

a = 10
b = 4

# Print bitwise AND operation
print("a & b =", a & b)

# Print bitwise OR operation
print("a | b =", a | b)

# Print bitwise NOT operation
print("~a =", ~a)

# print bitwise XOR operation
print("a ^ b =", a ^ b)
"""
Shift Operators
Estos operadores se utilizan para desplazar los bits de un número hacia la izquierda o hacia la derecha, multiplicando 
o dividiendo el número por dos, respectivamente. 
Se pueden utilizar cuando tenemos que multiplicar o dividir un número por dos.
Desplazamiento a la derecha bit a bit: desplaza los bits del número a la derecha y llena 0 en los vacíos a 
la izquierda (llena 1 en el caso de un número negativo) como resultado. 
Efecto similar al de dividir el número con alguna potencia de dos.

Bitwise right shift: Shifts the bits of the number to the right and fills 0 on voids 
left( fills 1 in the case of a negative number) as a result. 
Similar effect as of dividing the number with some power of two.
"""
# Example 1:
a = 10 # = 0000 1010 (Binary)
a >> 1 # = 0000 0101 = 5

# Example 2:
a = -10 #= 1111 0110 (Binary)
a >> 1 #= 1111 1011 = -5

"""Bitwise left shift: Shifts the bits of the number to the left and fills 0 on voids right as a result. 
Similar effect as of multiplying the number with some power of two.
Example: 
"""

#Example 1:
a = 5 #= 0000 0101 (Binary)
a << 1 #= 0000 1010 = 10
a << 2 #= 0001 0100 = 20

#Example 2:
b = -10 #= 1111 0110 (Binary)
b << 1 #= 1110 1100 = -20
b << 2 #= 1101 1000 = -40


# ******** ejemplo and
v = int(input('proporciona un valor numérico: '))
max = 5
r = (v > 0) and (v <= max)  # mas simple.... 0 < v <= maximo
if r:
    print(f'{v} está dentro de rango')
else:
    print(f'{v} está fuera de rango')
# ejemplo or
vacas = True
diaDescanso = False
if vacas or diaDescanso:
    print('A vivir')
else:
    print('niño rata....')
# ejemplo not
if not (vacas or diaDescanso):
    print('A vivir')
else:
    print('niño rata....')

# ******** EJEMPLOS COMPLETOS
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

# ******** EJEMPLOSSSS

n = int(input('Proporciona el numero1: '))
m = int(input('Proporciona el numero2: '))
if n > m:
    print(f'El numero mayor es: {n}')
elif n < m:
    print(f'El numero mayor es: {m}')
else:
    print(f'{n} y {m} son iguales')

# ********

print('Proporcione los siguientes datos del libro:')
nom = input('Proporciona el nombre: ')
pid = int(input('Proporciona el ID: '))
perras = float(input('Proporciona el precio: '))
env = bool(input('Indica si el envio es gratuito: '))

# ******** PRINT de varias lineas (con formato vamos)

print(f'''
nombre: {nom}
Id: {pid}
envio gratuito?: {env}
''')

# ******** mas ejercicio

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

# ******** Ejercicio IF
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
# ******** MAX MIN  #######################
# ***********************************
# max() maximo valor, se puede usar con múltiples parámetros
# min() minimo valor, se puede usar con múltiples parámetros
# Read three numbers.
number1 = int(input("Enter the first number: "))
number2 = int(input("Enter the second number: "))
number3 = int(input("Enter the third number: "))

# Check which one of the numbers is the greatest
# and pass it to the largest_number variable.

largest_number = max(number1, number2, number3)

# Print the result.
print("The largest number is:", largest_number)

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
else:
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


# A program that reads a sequence of numbers
# and counts how many numbers are even and how many are odd.
# The program terminates when zero is entered.

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

# arreglo cadena de caracteres, iterar recorrer cada elemento
# ********   break ###############
cadena = 'holanda'
for letra in cadena:
    if letra == 'a':
        print(f' letra encontrada: {letra}')
        break  # rompe el ciclo incluso el else ##############
else:
    print('fin for')

# ********   continue ###############

# rango de números range()
for i in range(10):
    if i % 2 == 0:
        print(f'Valor: {i}')
        continue  # ******** ejecuta la siguiente iteración si cumple
    print(f'{i} no es par')

for i in range(10):
    if i % 3 == 0:
        print(i)

for n in range(2, 10, 2):  # range(start, stop, step)
    print(n)


# ********   enumerate  ###############
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


#ejercicio
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


# ej 1
"""
Create a program with a for loop and a break statement. 
The program should iterate over characters in an email address, 
exit the loop when it reaches the @ symbol, and print the part before @ on one line. 

"""
for ch in "john.smith@pythoninstitute.org":
    if ch == "@":
        break
    print(ch, end="")
"""
Create a program with a for loop and a continue statement. The program should iterate Use the skeleton below:
"""
for digit in "0165031806510":
    if digit == "0":
        print("x", end="")
        continue
    print(digit, end="")

# ******** Ejercicio

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


# ******** Ejercicio pirámide
"""
Level of difficulty
Medium

Objectives
Familiarize the student with:

using the while loop;
finding the proper implementation of verbally defined rules;
reflecting real-life situations in computer code.
Scenario
Listen to this story: a boy and his father, a computer programmer, are playing with wooden blocks. They are building a pyramid.

Their pyramid is a bit weird, as it is actually a pyramid-shaped wall - it's flat. The pyramid is stacked according to one simple principle: each lower layer contains one block more than the layer above.

The figure illustrates the rule used by the builders:



Your task is to write a program which reads the number of blocks the builders have, and outputs the height of the pyramid that can be built using these blocks.

Note: the height is measured by the number of fully completed layers - if the builders don't have a sufficient number of blocks and cannot complete the next layer, they finish their work immediately.

Test your code using the data we've provided.


Test Data

Sample input: 6

Expected output: The height of the pyramid: 3
Sample input: 1000

Expected output: The height of the pyramid: 44
"""
blocks = int(input("Enter the number of blocks: "))
height = 0
number = 1
while blocks >= number:
    blocks -= number
    number += 1
    height += 1
print("The height of the pyramid:", height)

## ejercicio test hypotesis
# ojo! La cláusula else se ejecuta después de que el ciclo finaliza su ejecución,
# siempre que no haya sido terminado por break, por ejemplo:
"""Objectives
Familiarize the student with:

using the while loop;
converting verbally defined loops into actual Python code.
Scenario
In 1937, a German mathematician named Lothar Collatz formulated an intriguing hypothesis (it still remains unproven) which can be described in the following way:

take any non-negative and non-zero integer number and name it c0;
if it's even, evaluate a new c0 as c0 ÷ 2;
otherwise, if it's odd, evaluate a new c0 as 3 × c0 + 1;
if c0 ≠ 1, skip to point 2.
The hypothesis says that regardless of the initial value of c0, it will always go to 1.

Of course, it's an extremely complex task to use a computer in order to prove the hypothesis for any natural number (it may even require artificial intelligence), but you can use Python to check some individual numbers. Maybe you'll even find the one which would disprove the hypothesis.


Write a program which reads one natural number and executes the above steps as long as c0 remains different from 1. We also want you to count the steps needed to achieve the goal. Your code should output all the intermediate values of c0, too.

Hint: the most important part of the problem is how to transform Collatz's idea into a while loop - this is the key to success.

Test your code using the data we've provided.

Test Data

Sample input: 15

Expected output:

46
23
70
35
106
53
160
80
40
20
10
5
16
8
4
2
1
steps = 17
"""
c0 = int(input("numero: "))
step = 0
while  c0>1:
    if c0 % 2 == 0:
        c0 /= 2
        c0=int(c0)
    else:
        c0 = 3* c0 +1
    step +=1
    print(c0)
print(f'steps = {step}')

# Even par odd impar

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

print(list_3) # ['C']


list_1 = ["A", "B", "C"]
list_2 = list_1
list_3 = list_2

del list_1[0]
del list_2 # borra el apuntador

print(list_3) # ["B", "C"]


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

#bubble sort

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


# ..... metodos propios de pytohn para ordenat y dar la vuelta a una lista

my_list = [8, 10, 6, 2, 4]
my_list.sort()
print(my_list)

lst = [5, 3, 1, 2, 4]
print(lst)

lst.reverse()




#SOLUCIONAR EL ERROR DE LAS LISTAS!!!!!!
# list_2 = list_1 no estas copiando el contenido
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
#1. List comprehension allows you to create new lists from existing ones in a concise and elegant way. The syntax of a list comprehension looks as follows:
#[expression for element in list if conditional]
#which is actually an equivalent of the following code:
#for element in list:
#    if conditional:
#        expression
#Here's an example of a list comprehension - the code creates a five-element list filled with with the first five natural numbers raised to the power of 3:

cubed = [num ** 3 for num in range(5)]
print(cubed)  # outputs: [0, 1, 8, 27, 64]

# Comprensión de listas es una forma de crear nuevas listas basadas en la lista
# existente. Ofrece una sintaxis más corta siendo más compacta y
# rápida que las otras funciones y bucles utilizados para crear una lista.
# Por ejemplo,

py_list = ['a-1', 'b-2', 'c-3', 'a-4']
r = [s for s in py_list if "a" in s]
print(r)
# Producción: ['a-1', 'a-4']
# En el código anterior, la comprensión de listas se utiliza para buscar
# cadenas que tengan a en la lista py_list. Tenga en cuenta que escribir
# el mismo código utilizando otras funciones o bucles
# habría llevado más tiempo, ya que se requiere más código para su
# implementación, pero la comprensión de listas resuelve ese problema.
# También podemos usar la comprensión de listas para
# encontrar cadenas que contengan múltiples valores específicos, es decir,
# podemos encontrar cadenas que contengan “a” y “b”
# en py_list combinando las dos comprensiones. Por ejemplo,

py_list = ['a-1', 'b-2', 'c-3', 'a-4', 'b-8']
q = ['a', 'b']
r = [s for s in py_list if any(xs in s for xs in q)]
print(r)
# Producción: ['a-1', 'b-2', 'a-4','b-8']

#Eliminar duplicados y ordenar
lst = [10, 1, 2, 4, 4, 1, 4, 2, 6, 2, 9, 10]
lst = [lst[l] for l in range(len(lst)) if lst[l] not in lst[0:l]]
lst.sort()
print(lst) # [1, 2, 4, 6, 9, 10]

#anidados
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

#tiempo por hora por mes
temps = [[0.0 for h in range(24)] for d in range(31)]
#
# The matrix is magically updated here.


highest = -100.0

for day in temps:
    for temp in day:
        if temp > highest:
            highest = temp

print("The highest temperature was:", highest)

#tridimensional (ejemplo hotel 3 edificios de 15 plqntas con 20 habitaciones true o false ocupadas
#The first index (0 through 2) selects one of the buildings; the second (0 through 14) selects the floor,
# the third (0 through 19) selects the room number. All rooms are initially free.
rooms = [[[False for r in range(20)] for f in range(15)] for t in range(3)]
#Check if there are any vacancies on the 15th floor of the third building:

vacancy = 0

for room_number in range(20):
    if not rooms[2][14][room_number]:
        vacancy += 1

# Utilice la función filter() para obtener una cadena específica
# en una lista de Python La función filter() filtra el iterable dado con
# la ayuda de una función que comprueba si cada elemento satisface alguna
# condición o no. Devuelve un iterador que aplica la comprobación para cada
# uno de los elementos del iterable. Por ejemplo,

py_lst = ['a-1', 'b-2', 'c-3', 'a-4']
filter(lambda x: 'a' in x, py_lst)
print(filter(lambda x: 'a' in x, py_lst))
# Producción:<filter object at 0x7fd36c1905e0> Tenga en cuenta que la salida
# anterior es un objeto de tipo filtro-iterador
# ya que la función filter() devuelve un iterador en lugar de una lista.
# Podemos usar la función list() como se muestra en el código siguiente
# para obtener una lista.



list(filter(lambda x: 'a' in x, py_lst))


# Producción:['a-1','a-4']
# En el código anterior, hemos utilizado filter()
# para encontrar una cadena con valores específicos en la lista py_list.


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
t = tuple([1,2])

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

# DESEMPAQUETAR TUPLA
tup = 1, 2, 3
a, b, c = tup

print(a * b * c)

# método count() para tuplas Y LISTAS!
tup = 1, 2, 3, 2, 4, 5, 6, 2, 7, 2, 8, 9
duplicado = tup.count(2)

print(duplicado)
# MODIFICAR TUPLA
#
# no buena praxis, si cogemos tupla es porque q
# frutas[0] = 'Pera' # falla porque es una tupla
frutaslista = list(frutas)  # creamos una lista a raiz de la tupla list()
frutaslista[0] = 'Pera'  # Como es una lista si podemos modificarlo
frutas = tuple(frutaslista)  # convertimos la lista a tupla y asignamos
print('\n', frutas)  # por la sentencia de arriba forzamos el \n

# Ejercicio tupla y lista
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

#The sorted() function
#Do you want it sorted? Just enrich the for loop to get such a form:

dictionary = {"cat": "chat", "dog": "chien", "horse": "cheval"}

for key in sorted(dictionary.keys()):
    print(key, "->", dictionary[key])

#The sorted() function will do its best - the output will look like this:
#cat -> chat
#dog -> chien
#horse -> cheval

# 2 DICCIONARIOS en 1 !
d1 = {'Adam Smith': 'A', 'Judy Paxton': 'B+'}
d2 = {'Mary Louis': 'A', 'Patrick White': 'C'}
d3 = {}

for item in (d1, d2):
    d3.update(item)

print(d3)

#
# TUPLAS + DICCIONARIOS
#

# De tupla a diccionario
colors = (("green", "#008000"), ("blue", "#0000FF"))
colors_dictionary = dict(colors)
print(colors_dictionary)



# Las tuplas y los diccionarios pueden trabajar juntos Hemos preparado un ejemplo simple,
# que muestra cómo las tuplas y los diccionarios pueden trabajar juntos. Imaginemos el siguiente
# problema: necesita un programa para evaluar los puntajes promedio de los estudiantes;
# el programa debe pedir el nombre del estudiante, seguido de su puntaje único;
# los nombres podrán introducirse en cualquier orden; ingresar un nombre vacío finaliza
# la introducción de los datos (nota 1: ingresar una puntuación vacía generará la excepción ValueError,
# pero no se preocupe por eso ahora, verá cómo manejar tales casos cuando hablemos de excepciones
# en la segunda parte de la serie de cursos Python Essentials)
# A continuación, se debe emitir una lista de todos los nombres, junto con la puntuación media evaluada.

# Diccionario con media de las notas de los alumnos
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
# definir función def. 1º definir, después llamar (DEBAJO)
# miFunción también notación de camello
#La asignación de un valor al mensaje de nombre hace que Python olvide su rol anterior.
# La función denominada anteriomente deja de estar disponible
# si no ponemos llamamos a return, que devuelve la ejecución a donde estaba antes de la
# llamada a la función se ejecuta implícitamente al final, se puede usar para terminar
# la ejecución de la función a demanda
# Una variable existente fuera de una función tiene un alcance dentro de los cuerpos
# de las funciones, excluyendo aquellos de ellos que definen una variable del mismo nombre.
# También significa que el alcance de una variable existente fuera de una función se admite
# solo cuando se obtiene su valor (lectura). La asignación de un valor obliga a la creación
# de la propia variable de la función.

def my_function():
    print("Do I know that variable?", var)


var = 1
my_function()
print(var)
#Do I know that variable? 1
#1


def my_function():
    var = 2
    print("Do I know that variable?", var)


var = 1
my_function()
print(var)
#Do I know that variable? 2
# 1

#Hay un método especial de Python que puede ampliar el alcance de una variable de una manera
# que incluye los cuerpos de las funciones (incluso si desea no solo leer los valores,
# sino también modificarlos). Tal efecto es causado por una palabra clave llamada global:
def my_function():
    global var
    var = 2
    print("Do I know that variable?", var)


var = 1
my_function()
print(var)

#Do I know that variable? 2
#2

def mi_func(nombre, apellido):  # parámetro la variable con la que se define
    if nombre == '':
        return
    print(nombre, apellido)  # forma parte lo que está dentro de la indentación

mi_func('Jesus', 'Gomez')  # argumento valor que le paso

# si una función no devuelve un determinado valor mediante una cláusula de expresión return,
# se supone que implícitamente devuelve None.
def strange_function(n):
    if(n % 2 == 0):
        return True
print(strange_function(2))
print(strange_function(1))

#******* Pasando el argumento con un valor predefinido si no se introduce
def introduction(first_name, last_name="Smith"):
    print("Hello, my name is", first_name, last_name)
introduction("Henry")
introduction("James", "Doe")

def suma(n=0, m: int = 0) -> int:  # = x valor por defecto -> pista :pista
    return n + m

re = suma()
print(f'Resultado de suma: {re}')
print(f'Resultado de suma: {suma(6, 8)}')  # Podemos llamar a la función

# la siguiente función da error porque un argumento sin valor por defecto va antes que uno que si.
# deben ir todos los que tienen valor sin defecto antes
#def add_numbers(a, b=2, c):
#    print(a + b + c)

#add_numbers(a=1, c=3)
#SyntaxError - a non-default argument (c) follows a default argument (b=2)


#******* Pasando el argumento de palabras clave
# Python ofrece otra convención para pasar argumentos,
# donde el significado del argumento está dictado por su nombre,
# no por su posición: se llama paso de argumentos de palabras clave.
# Echa un vistazo al fragmento:
def introduction(first_name, last_name):
    print("Hello, my name is", first_name, last_name)

introduction(first_name = "James", last_name = "Bond")
introduction(last_name = "Skywalker", first_name = "Luke")

#Puede mezclar ambas modos si lo desea: solo hay una regla inquebrantable:
# debe poner los argumentos posicionales antes que los argumentos de palabras clave.

def adding(a, b, c):
    print(a, "+", b, "+", c, "=", a + b + c)
adding(4, 3, c = 2)


# ******** definir funcion cuando NO sabemos cuantas variables vamos a recibir
# lo toma como una tupla
def listarNombres(*nombres):  # en docu oficial *args pero no tienes que elegir ese nombre
    for nombre in nombres:
        print(nombre)

listarNombres('Pablo', 'Pablito', 'Pablete', 'Juanete')


#++++++++++#Es legal, y posible, tener una variable con el mismo nombre que el parámetro de una función.
# El fragmento ilustra el fenómeno:
def message(number):
    print("Enter a number:", number)

number = 1234
message(1)
print(number)
# Una situación como esta activa un mecanismo
# llamado shadowing: El parámetro x sombrea cualquier variable del mismo nombre,
# pero...... sólo dentro de la función que define el parámetro.
# El parámetro denominado number es una entidad completamente diferente de la variable
# denominada number. Esto significa que el fragmento anterior producirá el siguiente
# resultado: Introduzca un número: 1
# 1234

# Indice de masa corporal
def IMC(weight, height):
    if height < 1.0 or height > 2.5 or \
    weight < 20 or weight > 200:
        # \ hace que se siga en la siguiente línea no solo
        # en comentarios o string sino en codigo también
        # pero tiene que ser la siguiente un comentario no lo permite
        return None

    return weight / height ** 2

print(IMC(352.5, 1.65))

# ******** EJERCICIO FUNCIONES
#   Función con argumentos variables para sumar todos los valores recibidos
def suma(*args):
    x = 0
    for arg in args:
        x += arg
    return x


print(suma(1, 2, 3, 4, 5, 6, 7, 8))


# ******** EJERCICIO FUNCIONES 2
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


# ******** funcion que permite recibir diccionarios

def listarTerminos(**terminos):  # (**Kwargs)
    for llave, valor in terminos.items():
        print(f'{llave}:{valor}')


listarTerminos(IDE='blabla', PK='blublu')
listarTerminos(A=16, PK='blublu')


# ******** Funcion lista
def desplegarNombres(nombres):
    for nombre in nombres:
        print(nombre)


nombres = ['pa', 'pe', 'pi']
desplegarNombres(nombres)
desplegarNombres('carlos')  # Itera los caracteres, es decir el str
# desplegarNombres(6) desplegarNombres(6,8)int SI falla
desplegarNombres((6, 8))  # Como es una tupla no falla, lista como hemos visto idem
desplegarNombres([6, 9])

# ******** Funcion Es primo
def is_prime(num):
    r = True
    for i in range(2,num):
       if i % 2 == 0:
           r = False
           break
    return r
#
# Write your code here.
#

for i in range(1, 20):
	if is_prime(i + 1):
			print(i + 1, end=" ")
print() # 2 3 5 7 11 13 17 19

# ***********************************
# ******** ejemplo funciones variables  *************
# ***********************************

def printer(*args, **dics):
    for arg in args:
        print(arg)

    for clave, valor in dics.items():
        print(f'{clave} -->{valor} ')


printer(1, 2, 3, 'll', a='lol', b='lal')


"""
#### Problema listas en Funciones
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

It seems that the former rule still works.

Finally, can you see the difference in the example below:

def my_function(my_list_1):
    print("Print #1:", my_list_1)
    print("Print #2:", my_list_2)
    del my_list_1[0]  # OJO AQUI.
    print("Print #3:", my_list_1)
    print("Print #4:", my_list_2)


my_list_2 = [2, 3]
my_function(my_list_2)
print("Print #5:", my_list_2)


No cambiamos el valor del parámetro my_list_1 (ya sabemos que no afectará al argumento), 
sino que modificamos la lista identificada por él. 
El resultado puede ser sorprendente. Ejecute el código y verifique:


Print #1: [2, 3]
Print #2: [2, 3]
Print #3: [3]
Print #4: [3]
Print #5: [3]
output

si el argumento es una lista, cambiar el valor del parámetro correspondiente no afecta a la lista (recuerde: 
las variables que contienen listas se almacenan de una manera diferente a los escalares), 
Pero si cambia una lista identificada por el parámetro (nota: ¡la lista, no el parámetro!), 
La lista reflejará el cambio.
"""

# Es Triangulo y es triangulo equilátero

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

# ******** Conversión de Temperatura
# Realizar dos funciones para convertir de grados celsius a fahrenheit y viceversa.
# Función 1. Recibir un parámetro llamado celcius y regresar el valor equivalente a fahrenheit
# La función se llama: celsius_fahrenheit(celsius)
# La fórmula para convertir de celsius a fahrenheit es: celsius * 9/5 + 32
# Función 2. Recibir un parámetro llamado fahrenheit y regresar el valor equivalente a celsius:
# fahrenheit_celsius(fahrenheit)
# La fórmula para convertir de fahrenheit a celsius es:  (fahrenheit-32) * 5/9
# Los valores los debe proporcionar el usuario, utilizando la función input y convirtiendolo a tipo float.
# Deben hacer al menos dos pruebas, una donde conviertan de grados celcius a grados fahrenheit,
# y otra donde conviertan de grados fahrenheit a grados celsius y mandar a imprimir los resultados.

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

# ******** ejercicio función año bisisesto.
#Debe retornar True si el año es bisisto, False en caso contrario

def is_year_leap(year):
    r = False
    if year % 4 == 0: # divisible entre 4
        # salvo que sea año secular -último de cada siglo, terminado en «00»-, en cuyo caso también ha de ser divisible entre 400.
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


# EJERCIO FUNCION PANTALLA DE 13 LED (NUMEROS)
''' Su tarea es escribir un programa que pueda simular el trabajo de un dispositivo de siete pantallas, aunque
utilizará LED individuales en lugar de segmentos. Cada dígito se construye a partir de 13 LED, puede ser de cualquier
longitud de números

input: 9081726354
Sample output:
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

#Versión con menos codigo pero mas complejidad (n*n vs MIA 5*n)

# using a list containing patterns （0~9）

list=[['###', '# #', '# #', '# #', '###'],
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

# Factorial
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

# Fibonacci
#fib_1 = 1
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


# ******** EJERCICIO RECURSIVIDAD SUMA
# Imprimir números de 5 a 1 de manera descendente usando funciones recursivas.
# Puede ser cualquier valor positivo, ejemplo, si pasamos el valor de 5,
# debe imprimir: 5 4 3 2 1 Si se pasa el valor de 3, debe imprimir:
# 3 2 1 Si se pasan valores negativos no imprime nada

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
# Crear una función para calcular el total de un pago incluyendo un impuesto aplicado.
# La función se llama calcular_total()
# La función recibe dos parámetros:
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
# lambda arguments : expression
#   Una función lambda es una pequeña función anónima.
#   Una función lambda puede tomar cualquier número de argumentos, pero solo puede tener una expresión.
x = lambda a: a + 10
print(x(5))

x = lambda a, b: a * b
print(x(5, 6))

x = lambda a, b, c: a + b + c
print(x(5, 6, 2))

# corrector de nombre y apellidos
nombre_completo = lambda n, a: n.strip().title() + " " + a.strip().title()
print(nombre_completo("   jesus", "   GOMEZ"))

# Ordenar lista por apellido split separa por el espacio y empezamos por el
# final hasta ese espacio, lower todo a minusculas por si acaso
lista = ["Jesus Gomez", "María Macanás", "Marisa Baños", "Maria Canovas"]
lista.sort(key=lambda name: name.split(" ")[-1].lower())
print(lista)


# El poder de lambda se muestra mejor cuando los usas como una función anónima dentro de otra función. Digamos que
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

# **************** dir() ########
#La función devuelve una lista ordenada alfabéticamente que contiene todos los nombres de entidades disponibles
# en el módulo identificado por un nombre pasado a la función como argumento: dir(módulo)
import math
print(dir(math)) # devuelve una lista

import math
for name in dir(math):
    print(name, end="\t")


# # **************** math ######## modulo de matemáticas
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
math.log10(x)   #el logaritmo decimal de x (más preciso que log(x, 10))
math.log2(x)    #el logaritmo binario de x (más preciso que log(x, 2))
pow(x, y)   #encontrar el valor de x elevado a y (cuidado con los dominios)
#Esta es una función incorporada y no tiene que importarse.

# Propósito general
math.ceil(x)    # el entero más pequeño MAYOR o igual que x
math.floor(x)   # el entero más grande MENOR o igual que x)
math.trunc(x)   # el valor de x truncado a un entero (ten cuidado, no es un equivalente ni de ceil ni de floor)
math.factorial(x) # devuelve x! (x tiene que ser una integral y no una negativa)
math.hypot(x, y) # devuelve la longitud de la hipotenusa de un triángulo rectángulo con las longitudes de las piernas iguales a x e y (igual que sqrt(pow(x, 2) + pow(y, 2)) pero más precisa)

# **************** random ######## modulo de numeros aleatorios
#Un generador de números aleatorios toma un valor llamado semilla,
# lo trata como un valor de entrada, calcula un número "aleatorio" basado en él (el método depende de un algoritmo elegido) y produce un nuevo valor de semilla.
from random import random, seed, randrange, randint, choice, sample

beg, end,step, left, right, sequence, elements_to_choose = 0, 10, 1, 2, 5, [1,2,3,4,5,6,7,8,9], 5
random()    #produce un número flotante x procedente del rango (0,0, 1,0)
seed()      #La función es capaz de establecer directamente la semilla del generador.
            # Te mostramos dos de sus variantes: seed() - establece la semilla con la hora actual;
            # seed(int_value): establece la semilla con el valor entero int_value.

for i in range(5):
    # seed(0) # con seed0 0 estsblcemos la semilla en 0 con lo que ya no es aleatorio, los numeros generados serán los mismos
    print(random())

# valores aleatorios enteros, exclusión implicita del lado derecho es como un aleatorio del range
randrange(end)
randrange(beg, end)
randrange(beg, end, step)
randint(left, right)

# Las funciones anteriores tienen una desventaja importante: pueden producir valores repetitivos
# incluso si el número de invocaciones posteriores no es mayor que el ancho del rango especificado.
# Afortunadamente, hay una mejor solución que escribir su propio código para verificar la singularidad de los números "dibujados".

choice(sequence) # La primera variante elige un elemento "aleatorio" de la secuencia de entrada y lo devuelve.
sample(sequence, elements_to_choose) # Elige algunos de los elementos de entrada, devolviendo una lista  del tamaño
# indicado con la opción. Los elementos de la muestra se colocan en orden aleatorio
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(choice(my_list))
print(sample(my_list, 5))
print(sample(my_list, 10))


# **************** platform ######## info sist operativo

# El módulo de plataforma le permite acceder a los datos de la plataforma subyacente, es decir, hardware, sistema operativo e información de versión del intérprete.
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
print(python_version_tuple())   # ('3', '11', '0'), la mayor parte de la versión de Python; la parte menor; El número de nivel de parche.
# La versión del sistema operativo se proporciona como una cadena

# ***********************************
# ******** Paquetes #############
# ***********************************
#
# Un módulo es una especie de contenedor lleno de funciones
# Paquete: Agrupa sus módulos con un rol similar a una carpeta / directorio en el mundo de los archivos.
# Si creamos un módulo (un archivo.py )aunque sea vacío y lo importamos en otro,
# Aparece una nueva subcarpeta,  __pycache__. Dentro  Hay un archivo llamado module.cpython-xy.pyc donde x e y
# son dígitos derivados de su versión de Python (por ejemplo, serán 3 y 8 si usa Python 3.8).
# El nombre del archivo es el mismo que el nombre de su módulo (módulo aquí).
# La parte después del primer punto dice qué implementación de Python ha creado el archivo (CPython aquí)
# y su número de versión. La última parte (pyc) proviene de las palabras Python y compilado.
# el contenido es completamente ilegible para los humanos.
# Tiene que ser así, ya que el archivo está destinado solo para uso de Python.
# Cuando Python importa un módulo por primera vez, traduce su contenido en una forma algo compilada.
# Cuando se importa un módulo, Python ejecuta implícitamente su contenido.
# Le da al módulo la oportunidad de inicializar algunos de sus aspectos internos.
# Python recuerda los módulos importados y omite silenciosamente todas las importaciones posteriores.
# Cuando ejecuta un archivo directamente, su variable __name__ se establece en __main__;
# Cuando un archivo se importa como un módulo, su variable __name__ se establece en el nombre del archivo (excluyendo.py)
if __name__ == "__main__":
    print("I prefer to be a module.")
else:
    print("I like to be a module.")
# variables privadas: precediendo el nombre de la variable con _ (un guión bajo) o __ (dos guiones bajos),
# pero recuerde, es solo una convención. Los usuarios de su módulo pueden obedecerlo o no.
# the line starting with #! instructs the OS how to execute the contents of the file,  Esta convención no tiene ningún efecto bajo MS Windows.
# ( para python es un comentario) ejemplo

# path: variable especial (en realidad una lista) que almacena todas las ubicaciones (carpetas/directorios)
# que se buscan para encontrar un módulo que ha sido solicitado por la instrucción de importación está en el módulo sys

# Python explora estas carpetas en el orden en que aparecen en la lista:
# si el módulo no se puede encontrar en ninguno de estos directorios, la importación falla.
# De lo contrario, se tendrá en cuenta la primera carpeta
# que contenga un módulo con el nombre deseado (si alguna de las carpetas restantes contiene un módulo con ese nombre, se ignorará).

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
# ubicado dentro del archivo del módulo. Como un paquete no es un archivo, esta técnica es inútil para inicializar paquetes.
# En su lugar, debe usar un truco diferente: Python espera que haya un archivo con un nombre muy único dentro de la carpeta
# del paquete: __init__.py. El contenido del archivo se ejecuta cuando se importa cualquiera de los módulos del paquete.
# Si no desea ninguna inicialización especial, puede dejar el archivo vacío, pero no debe omitirlo.

# pregunta
#Some additional and necessary packages are stored inside the D:\Python\Project\Modules directory.
# Write a code ensuring that the directory is traversed by Python in order to find all requested modules.
import sys

# note the double backslashes! (windows)

sys.path.append("D:\\Python\\Project\\Modules")

# The directory mentioned in the previous exercise contains a sub-tree of the following structure:
# abc
#  |__ def
#       |__ mymodule.py
# Assuming that D:\Python\Project\Modules has been successfully appended to the sys.path list,
# write an import directive letting you use all the mymodule entities.

# import abc.def.mymodule


# PyPI es el repositorio central de python, pip ( pip install packages) la herramienta para usarlo, permite resolver las depdendencias entre modulos
# pip help, ayuda de pip,
# pip help, install ayuda especifica para instalar
# pip list, version de pip y de la herramienta
# pip show package_name, te da información sobre los paquetes INSTALADOS, ej pip show pip. dentro
# de el Requires: y Required-by, por convención te dice sus dependencias , qué paquetes son necesarios para utilizar
# correctamente el paquete (Requiere:) qué paquetes necesitan que el paquete se utilice correctamente (Requerido por:)
# pip search anystring , El anystring proporcionado busca en el directorio ( repo)  los nombres de todos los paquetes; Las cadenas de resumen
# de todos los paquetes, no diferencia mayusculas de minusculas
# --user  a la hora de instalar un paquete con --user solo lo instalamos para el usuario ( no necesita privilegios)
# y sin el en el sistema, ej # pip install pygame (admin) y +  --user lo comentado
# -U update, versión específica pip install pygame==1.9.2
# pip uninstall package_name desistalar paquetes

# ***********************************
# ******** CLASES #############
# ***********************************
# Una clase es como una plantilla de la cual podemos sacar objetos (instancias)
# Ejemplo clase -> Persona, instancias juan y carlos
# Posee Atributos y Métodos
# Persona.py


class Person:
    pass  # Palabra reservada para poder crear la función o clase sin contenido


print(type(Person))


class Persona:  # this == self se puede usar ambos
    # init método inicializador, similar a un constructor, en python está oculto y se llama por el lenguaje
    # Permite agregar e inicializar atributos
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
        # AL encontrarnos dentro de la clase nos referimos con self al atributo
        print(f'Persona: {self.nombre} {self.apellido}  que tiene {self.edad} años')


persona1 = Persona('Juan', 'Perez', 28)  # Estamos llamando al constructor (init) estamos creando una instancia de
# La clase persona
# creando un objeto. self está apuntando al objeto que se está creando en ese momento.
# Al ser python las variables dinámicas se crean automáticamente con el valor que le pasamos
persona2 = Persona('Pedro', 'Baños', 45)
persona2.mostrar_detalle()
persona1.mostrar_detalle()

# MODIFICAR ATRIBUTO DE LA CLASE #############

# (no es recomendable, mejor por métodos por encapsulamiento)
persona1.nombre = 'Jesus'
persona1.apellido = 'Gomez'
persona1.edad = 32
persona1.mostrar_detalle()  # Lo común

# Podemos llamar al método de la clase referenciando al objeto
Persona.mostrar_detalle(persona1)

# Añadir atributos ****************************
# Ventaja de python y POO podemos añadir atributos al objeto en cualquier momento
# No se van a compartir con el resto de objetos
persona1.telefono = '968888888'
print(persona1.telefono)


# Ejericio POO Aritmética #############
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


# Ejericio POO rectangulo #############

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


# Ejercicio POO cubo #############

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


# ***********************************
# ******** ENCAPSULAMIENTO, GET y SET y DESTRUCTORES ##########
# ***********************************
#   Con _ indicamos que sólo desde la propia clase podemos acceder a la clase
#   Aunque no deberíamos el lenguaje si te deja, es una sugerencia
#   Con __ si que omite la modificación del valor (no falla) es menos comun y falla al sacarlo
# ************************
# ROBUSTECER METODO INIT ############

class Persona:

    def __init__(self, nombre, apellido, edad, *valores, **terminos):
        # *args si queremos pasar una tupla de elementos variable
        # **kwargs si queremos pasar diccionario

        self.__nombre = nombre
        self._apellido = apellido
        self._edad = edad
        self.valores = valores
        self.terminos = terminos

    # MÉTODOS #############
    def mostrar_detalle(self):  # en los métodos de instancia siempre vamos a meter la referencia self
        # AL encontrarnos dentro de la clase nos referimos con self al atributo
        print(
            f'Persona: {self.__nombre} {self.apellido}  que tiene {self.edad} años,  con {self.valores} y diccionario {self.terminos}')

    # SOBREESCRITURA **************
    def __str__(self):  # estamos sobrescribiendo la clase str de la clase padre

        return f'Persona: {self.__nombre} {self.apellido}  que tiene {self.edad} años'

    # get nos permite recuperar el valor y set modificarlos
    @property  # es un decorador, encapsula el atributo y lo hace accesible solo desde el metodo
    # entonces se hace accesible como un atributo (no tenemos que poner ())
    def nombre(self):
        return self.__nombre

    @nombre.setter  # con este decorador debemos indicar el nombre del atributo sin _
    # y .setter porque va a modificar el valor seguimos llámandolo si parentesis
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


# LLamada pasandole tupla y diccionario #############

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

# Si lo creamos en un archivo podemos acceder a este modulo desde otro archivo
# si hemos llamado el archivo Clases.py y queremos importar la clase persona
# ponemos: from Clases import Persona | si queremos todas : from Clases import *

# __name__ es una propiedad que indica nombre del modulo
# si lo ejecutamos desde el propio archivo donde aparece pondra main
# si no el nombre dle archivo
print(__name__)

if __name__ == '__main__':
    print('Me ejecuto solo si estoy dentro del modulo donde lo defino')
    # esto sirve para codigos de prueba dentro del modulo

# DESTRUCTORES #
#
print('Creación de objetos'.center(50, '-'))  # con .center(50, '-') centramos
# lo que imprimimos metiendo - hasta 50 caracteres
persona1 = Persona('Fran', 'Villa', 33)
persona1.mostrar_detalle()

print('Eliminación objetos'.center(50, '-'))
del persona1  # eliminción explicita


# Es raro en python por la existencia del concepto del reoclector de basura
# Esto es porque todos los objetos que no estén apuntados por una varable se van
# a destruir de forma automática y al finalizar el programa igual

# ***********************************
# ******** HERENCIA ##########
# ***********************************
# Todas las clases heredan de object
class Empleado(Persona):  # con (Padre) indicamos en la declaración que heredamos
    def __init__(self, nombre, apellido, edad, sueldo):
        # tenemos que inicializar los atributos del padre
        super().__init__(nombre, apellido, edad)
        # super metodo que nos permite acceder a los atributos del padre
        # con super().__init__(atributos padre) estamos inicializando los atr padre
        self.sueldo = sueldo

    # SOBREESCRITURA
    def __str__(self):  # estamos sobrescribiendo la clase str de la clase padre
        # no tenemos visibilidad sin sobrescribir sobre sueldo porque por defecto
        # estaríamos usando el __str__ de Persona por lo que sobreescribimos
        # con super podemos acceder al atributo o método de la clase padre
        return f'Empleado: {super().__str__()}, y con sueldo {self.sueldo}'


empleado1 = Empleado('Juan', 'garcia', 23, 5000)
print(empleado1.nombre)
print(empleado1.sueldo)
print(empleado1)


# Ejercicio Herencia en Python ***********
# ****************************
# Definir una clase padre llamada Vehiculo y dos clases hijas llamadas Coche y
# Bicicleta, las cuales heredan de la clase Padre Vehiculo

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
        super().__init__(color, 4)
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

# ***********************************
# ******** HERENCIA MULTIPLE y ABSTRACTA ##########
# ***********************************
# ABSTRACTA no se pueden crear instancias de ella ( figura = FiguraGeometrica() )
# Obliga a las clases hijas a realizar una implementación
# ABC = Abstract Base clase base para convertir una clase en abstracta
from abc import ABC, abstractmethod


class FiguraGeometrica(ABC):  # al extender de ABC es abstracta
    def __init__(self, ancho, alto):
        # añadimos comprobación de entrada con valor numerico positico
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

    # METODO ABSTRACTO
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
        # super().__init__(self, lado, lado)
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
# Los metodos por ejemplo
print(Cuadrado.mro())


# ***********************************
# ******** Variable de clase, Métodos estáticos y método de clase ##########
# ***********************************
# los atributos son independientes, corresponden a cada instancia
# las variables de clase se comparten
# porque se asocian con la clase en si misma y se comparte con todos los objetos.
# Esto es porque la clase se carga en memoria al pasar la parte del programa
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

    # un método de clase, sin embargo si que recibe un contexto de clase
    # recibe un parámetro cls que significa class (podría ser cualquiera, pero se recomienda)
    @classmethod
    # contexto estático !!
    def metodo_clase(cls):
        print(cls.variables_clase)  # recibe correctamente la referencia de nuestra clase

    def metodo_instancia(self):  # contexto dinamico !!
        self.metodo_clase()  # podemos acceder al contexto estatico


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
print(MiClase.variables_clase2)  # directamente no se puede (sin nombre clase)
print(objeto2.variables_clase2)
print(MiClase.metodo_estatico())
MiClase.metodo_clase()
objeto1.metodo_clase()  # se pasa cls auto
objeto1.metodo_instancia()

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


# ******** Ejercicio contador de clases ##########
#
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
# UML - Undefined modeling lenguaje, realizado con umlet (umletino.com)
# Simulación de venta de productos agregarla a una orden
# como si tuviéramos un ticket de venta en el cual vamos
# a vender varios productos y se van a agregar a una orden
# A partir de esa orden vamos a calcular el total de todos
# los productos que se han vendido utilizando el precio
# de producto para ello la clase de producto va a tener: id_producto mediante un contador
# nombre, precio ( nos permitirá obtener el total del ticket generado por producto
# agregado orden
# metodo str para imprimir los atributos
# por cada producto que creemos lo vamos a agregar a la clase de orden, como hemos comentado
# ****** RELACION AGREGACION ******* mediante una lista de objetos de tipo producto al que se agregará
# podremos tener varias ordenes y cada una de productos
# tendremos un contador de ordenes, ID , str
# la primera clase que se recomienda crear es la que no tiene relación con ninguna, en este
# caso producto ya que orden puede recibir un listado de productos

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
# Multiples formas en tiempo de ejecución
# una misma variable puede ejecutar varios métodos de distintos objetos dependiendo del objeto
# al cual esté apuntando en tiempo en ejecución
# Si tenemos una variable de una clase que tiene el método str, y otra que tiene gerente y ejecutarse el q sea
# Es decir ejecutar multiples métodos en tiempo de ejecución dependiendo del objeto al cual esté apuntando
# se ejecuta uno dependiendo de cuál apunte. En Python no tienen que tener relación

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
# ********  LABORATORIO Mundo PC ##########
# ***********************************
# diagrama uml en https://www.udemy.com/course/universidad-python-desde-cero-hasta-experto-django-flask-rest-web/learn/lecture/26667098#overview
# _ protegido (#)
# __privado
# orden que contiene computadoras, las cual tiene monitor telcado
# y raton ( los dos últimos pertenecen a dispositivo de entrada)

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

# ***********************************
# ********  Excepciones ##########
# ***********************************
# Manejo de errores
# en python Tenemos la clase base BaseException
# de ella hereda Exception
# de la cual cuelgan más:
#   AritmeticError ,
#   OSError (FileNotFoundError, PermissionError)
#   RuntimeError,
#   LookupError (IndexError, KeyError),
#   SyntaxError
while True:
    try:
        number = int(input("Enter an int number: "))
        print(5/number)
        break
    except (ValueError, ZeroDivisionError):
        print("Wrong value or No division by zero rule broken.")
    except:
        print("Sorry, something went wrong...")


# 0

try:
    '10' / 0
except Exception as e:  # ZeroDivisionError peta
    # Si capturamos un error con clase más específica, no cogerá los que se salgan
    # Solamente clases padre pueden procesar excepciones, incluyendo clases hijas
    print(f'Ocurrió un error: {e}')

# 1
resultado = None  # Al usarse fuera del try hay que declararla fuera si no falla
try:
    a = int(input('Primer numero: '))
    b = int(input('Segundo numero: '))
    resultado = a / b
except ZeroDivisionError as e:
    print(f'ZeroDivisionError Ocurrió un error: {e}, {type(e)}')
except TypeError as e:
    print(f'TypeError Ocurrió un error: {e}, {type(e)}')
except Exception as e:
    print(f'Exception Ocurrió un error: {e}, {type(e)}')
else:  # solo se ejecuta si no se lanza NINGUNA excepción
    print('No se arrojó ninguna excepción')  # 10 / 2 es ok, se arroja:  No se arrojó ninguna excepción
finally:  # Siempre se ejecuta incluso si se lanza una excepción
    print('Continuamos')
    print(f'Resultado: {resultado}')


# podemos manejas varias excepciones deben ir de más específico a más genérico
# Si ponemos la mas general al principio no se manejarán las mas específicas

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

# ***********************************
# ********  Manejo de Archivos ##########
# ***********************************
# podemos trabajar casi cualquier tipo de archivos desde textos hasta imágenes

# ********  Archivos  DE TEXTO ##########
# "r" leer, abre archivo para leer, falla si no existe
# "r+" leer o escribir información
# "a" añadir, crea el archivo si no existe
# "w" Escribir, abre el fichero para escritura, crea el fichero si no existe
# "w+" Escribir o leer información
# "x" Crear, crea un archivo específico, falla si existe
# Además puedes especificar si es manejado como binario o texto
# "t" por defecto, texto | "b" binario ( imágenes por ejemplo)
# Si estamos en otra carpeta podemos especificar la misma
# En windows, como \ es un carácter especial tenemos
# que poner otro \ para salvarlo e indicar que no es un carácter especial por eso es 'c:\\ ( en la 1º)
# en mac o linux como es / no hace falta
try:
    # OPEN **** puede abrir un archivo nuevo (si no existe) o existente y puede escribir en el o leer
    archivo = open('prueba.txt', 'w', encoding='utf8')  # encoding ='utf8' hace que permita acentos
    # WRITE **** escribir en un archivo ya abierto claro
    archivo.write('Agregamos información al archivo\n')
    archivo.write('EEEE')
    # lo está sobreescribiendo si lo hacemos varias veces
    # al no existir lo va a crear | Al no especificar ruta lo crea en la de por defecto
except Exception as e:
    print(e)
finally:
    archivo.close()  # siempre debe cerrarse, después de cerrar falla al escribir claro
    print('Fin del archivo')

try:
    archivo = open('prueba.txt', 'r', encoding='utf8')
    # print(archivo.read())   # al leerlo el recorremos el archivo,
    # con lo que se encontraría apuntando al final del mismo

    # leer algunos caracteres ***
    # print(archivo.read(5))
    # print(archivo.read(3))  # siguientes 3 sigue recorriendo ( en otra linea)

    # leer líneas completas **
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
        print('fin copia')

    except Exception as e:  # Excepción en archivo 2
        print(e)

    print('fin lectura')

except Exception as e:  # Excepción en archivo lectura
    print(e)

finally:

    archivo2.close()
    archivo.close()

# ********  Archivos  con with ##########
# existe una sintaxis simplificada que automáticamente va  abrir y a cerrar
# nuestro archivo automáticamente sin tener que cerrarlo después
# se ejecutan de forma dinámica con __enter__ ( para abrirlo )
# y con __exit__ para cerrarlo. Se pueden editar sus métodos

with open('prueba.txt', 'w+', encoding='utf8') as archivo:
    print(archivo.write('lolito'))

# Clase de manejo de archivos #######
# tiene que implementar __enter__ y __exit__
# para considerarse de manejo de archivos
# heredan de object no hay que heredar de nada más
# Aparte de abrir y cerrar archivos podríamos usarlo
# para abrir o cerrar otros recursos como conexiones a BBDD


class Manejo_archivos:
    def __init__(self, nombre):
        self.nombre = nombre

    # estamos encapsulando el codigo en el metodo enter
    # que se llamara automáticamente al momento de abrir el recurso
    # con utilizar with se manda a llamar enter
    # y al dejar de ejecutar se llama a exit
    def __enter__(self):  # obtener
        print('Obtenemos el recurso'.center(50, '#'))
        self.nombre = open(self.nombre, 'w', encoding='utf8')
        return self.nombre  # Devolvemos el objeto si no no irá

    # Recibe más parámetros
    # si ocurre una excepción podemos recibir el tipo, su valor
    # y la traza que es el texto del mismo no son obligatorios
    # pero si incluirlos
    def __exit__(self, tipo_excepción, valor_excepción, traza_error):  # cerrar
        print('cerramos el recurso'.center(50, '#'))
        # preguntamos si el atributo de nombre
        # está apuntando a algún objeto lo que querría decir que está abierto
        if self.nombre:
            # si está abierto lo cerramos
            self.nombre.close()


with Manejo_archivos('prueba.txt') as archivo:
    print(archivo.write('lolito'))



# ***********************************
# ********  Ejercicio catálogo películas ##########
# ***********************************
#
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


##############################
# desafío Edube 1: 3 en raya
##############################
"""
the computer (i.e., your program) should play the game using 'X's;
the user (e.g., you) should play the game using 'O's;
the first move belongs to the computer − it always puts its first 'X' in the middle of the board;
all the squares are numbered row by row starting with 1 (see the example session below for reference)
the user inputs their move by entering the number of the square they choose − the number must be valid, i.e.,
it must be an integer, it must be greater than 0 and less than 10,
 and it cannot point to a field which is already occupied;
the program checks if the game is over − there are four possible verdicts: the game should continue,
 the game ends with a tie, you win, or the computer wins;
the computer responds with its move and the check is repeated;
don't implement any form of artificial intelligence − a random field choice made by the computer is good enough for the game.
The example session with the program may look as follows:

+-------+-------+-------+
|       |       |       |
|   1   |   2   |   3   |
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

Requirements
Implement the following features:

the board should be stored as a three-element list, while each element is another three-element list
(the inner lists represent rows) so that all of the squares may be accessed using the following syntax:

board[row][column]
each of the inner list's elements can contain 'O', 'X', or a digit representing the square's number
 (such a square is considered free)
the board's appearance should be exactly the same as the one presented in the example.
implement the functions defined for you in the editor.

Drawing a random integer number can be done by utilizing a Python function called randrange().
The example program below shows how to use it (the program prints ten random numbers from 0 to 8).
Note: the from-import instruction provides access to the randrange function defined
within an external Python module callled random.

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



##############################
# DESAFIO DAMAVIS
##############################
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
