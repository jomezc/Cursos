
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
'''Multiples formas en tiempo de ejecución una misma variable puede ejecutar varios métodos de distintos objetos 
dependiendo del objeto al cual esté apuntando en tiempo en ejecución. Si tenemos una variable de una clase que tiene el 
método str, y otra que tiene gerente y ejecutarse el que sea Es decir ejecutar multiples métodos en tiempo de ejecución 
dependiendo del objeto al cual esté apuntando se ejecuta uno dependiendo de cuál apunte. En Python no tienen que tener 
relación.
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
    def sueldo(self, sueldo):
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
    # mismo resultado se ejecuta el método del padre, pero el str de la que está apuntando es el del hijo (polimorfismo)
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


class Class:
    def __init__(self, n):
        self.__iter = Fib(n)

    def __iter__(self):
        print("Class iter")
        return self.__iter


object1 = Class(8)  # se inicializa class que asu vez fib

for i in object1:  # es este for el que llama a iter() que a su vez llama a next
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
print(isinstance(a, Iterable))  # True
print(isinstance(a, Iterator))  # True

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
una compresión de listas, una lista! e incluso el operador in (como for i in range(x)). La invocación devolverá el 
identificador del objeto, no la serie que esperamos del generador.
'''


# ++++++++ ejemplo yield +++++
def powers_of_2(n):
    power = 1
    for i in range(n):
        yield power  # se guarda y luego se hace la potencia si no el 1º elemento no aparecería
        power *= 2


t = [x for x in powers_of_2(5)]
ll = list(powers_of_2(3))
for i in range(20):
    if i in powers_of_2(8):
        print(i, end=' ')
print(f'\n{t}')  # [1, 2, 4, 8, 16]
print(ll)  # [1, 2, 4]


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
def outer(par):
    loc = par


var = 1
outer(var)
print(var)
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
print(fun())  # 1
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
        Exception.__init__(self, msg+msg)  # llama al init del padre con 2 veces el mensaje
        self.args = (msg,)  # aquí al pasarle los argumentos sobrescribe la inicialización por el mensaje simple


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
    print(e.args)  # (1, 2, 3), una tupla


# ++++++ otro
class A:
    def __init__(self):
        pass


a = A(1)
# a = A() así si funcionaría y daría False hasattr
print(hasattr(a, 'A'))  # TypeError, 2 argumentos y solo hemos definido uno!!!


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
# map(lambda n: n | 1, any_list) logica, list porque es un generador
print(list(map(lambda n: n | 1, any_list)))  # [1, 3, 3, 5] recuerda, | es como un or


# +++++++++ otro
# ¿cual es la salida esperada del siguiente código?
def replace_spaces(replacement='*'):
    def new_replacement(text):
        return text.replace(' ', replacement)
    return new_replacement


stars = replace_spaces()
print(stars("And Now for Something Completely Different"))  # And*Now*for*Something*Completely*Different
