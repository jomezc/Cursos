
# ***********************************
# ********  Excepciones ##########
# ***********************************
'''Manejo de errores
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

Si desea controlar dos o más excepciones de la misma manera, puede usar la siguiente sintaxis:'''
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
# What happened? An exception was raised!
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
        raise  # !!!!! solo dentro de except, genera la misma excepción


try:
    bad_fun(0)
except ArithmeticError:
    print("I see!")

# salida de ejecución:
# I did it again!
# I see!
'''
La instrucción de aumento también se puede utilizar de la siguiente manera (tenga en cuenta la ausencia del nombre de 
la excepción, solo raise): este tipo de instrucción de elevación puede usarse SOLO dentro  de la rama except; Usarlo en
cualquier otro contexto causa un error. 
La instrucción volverá a generar inmediatamente la misma excepción que se maneja actualmente.
Gracias a esto, puede distribuir el manejo de excepciones entre diferentes partes del código. en el ejemplo anterior 
ZeroDivisionError se genera dos veces: 
Primero, dentro de la parte Try del código (esto es causado por la división cero real) 
Segundo, dentro de la parte excepto por la instrucción de elevación.
"""

"""
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
            return v  # es lo que impide un bucle eterno devuelve el número cuando es correcto
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
    print_args(e.args)  #

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

class NewValueError(ValueError):
    def __init__(self, name, color, state):
        self.data = (name, color, state)

try:
    raise NewValueError("Enemy warning", "Red alert", "High readiness")
except NewValueError as nve:
    for arg in nve.args:
        print(arg, end='! ')  # Enemy warning! Red alert! High readiness!
