
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
    varia = 2
    print("Do I know that variable?", varia)  # 2


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
my_function()  # Do I know that variable? 2
print(var)  # 2


def mi_func(nombre1, apellido):  # parámetro la variable con la que se define
    if nombre1 == '':
        return
    print(nombre1, apellido)  # forma parte lo que está dentro de la indentación


mi_func('Jesus', 'Gomez')  # argumento valor que le paso


# si una función no devuelve un determinado valor mediante una cláusula de expresión return, se supone que
# implícitamente devuelve None.
def strange_function(num_v):
    if num_v % 2 == 0:
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


introduction(first_name="James", last_name="Bond")
introduction(last_name="Skywalker", first_nam="Luke")


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
    for i in range(2, num):
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
    string = string.upper().replace(' ', '')
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


def palindromo(string, string2):
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


palindromo('Listen', 'Silent')


# ++++   ejemplo funciones variables  +++++++
def printer(*args, **dics):
    for arg in args:
        print(arg)

    for clave, valor in dics.items():
        print(f'{clave} -->{valor} ')


printer(1, 2, 3, 'll', a='lol', b='lal')

# ******** Problema listas en Funciones ####


def my_function(my_list_1):

    print("Print #1:", my_list_1)  # Print #1: [2, 3]
    print("Print #2:", my_list_2)  # Print #2: [2, 3]
    my_list_1 = [0, 1]
    print("Print #3:", my_list_1)  # Print #3: [0, 1], Estamos imprimiendo la lista recién modificada
    print("Print #4:", my_list_2)  # Print #4: [2, 3], no se modifica la lista original


my_list_2 = [2, 3]
my_function(my_list_2)
print("Print #5:", my_list_2)  # Print #5: [2, 3]
""" sin embargo """


def my_function(my_list_1):
    print("Print #1:", my_list_1)  # Print #1: [2, 3]
    print("Print #2:", my_list_2)  # Print #2: [2, 3]
    del my_list_1[0]  # OJO AQUI.
    print("Print #3:", my_list_1)  # Print #3: [3]
    print("Print #4:", my_list_2)  # Print #3: [3]


my_list_2 = [2, 3]
my_function(my_list_2)
print("Print #5:", my_list_2)   # Print #3: [3]
"""
No cambiamos el valor del parámetro my_list_1 (ya sabemos que no afectará al argumento), sino que modificamos la lista 
identificada por él. si el argumento es una lista, cambiar el valor del parámetro correspondiente no afecta a la lista 
(recuerde: las variables que contienen listas se almacenan de una manera diferente a los escalares), Pero si cambia una 
lista identificada por el parámetro (nota: ¡la lista, no el parámetro!), La lista reflejará el cambio.
la solución podría ser, copiar la lista, como se vío anteriormente:
list_2 = list_1  # no estás copiando el contenido, sino la dirección de memoria donde está alojado
list_1 = [1]
list_2 = list_1[:]
list_1[0] = 2
print(list_2)  # [1]
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
    if year % 4 == 0:  # divisible entre 4
        # salvo que sea año secular -último de cada siglo, terminado en «00»-, en cuyo caso también ha de ser divisible
        # entre 400.
        if str(year)[-2:] != '00' or (str(year)[-2:] == '00' and year % 400 == 0):
            r = True
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

# Versión con menos código pero mas complejidad (n*n vs MIA 5*n)
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
