
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
#   Con _ indicamos que solo desde la propia clase podemos acceder a la clase
#   Aunque no deberíamos el lenguaje si te deja, es una sugerencia
#   Con __ si que omite la modificación del valor (no falla) es menos comun y falla al sacarlo, menos con
# objeto._Clase__variable/método()
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
# si no el nombre del archivo
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
        super().__init__(nombre, apellido, edad)  # no necesitamos saber el nombre ni hacer referencia a self
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
# Bicicleta, las cuales heredan de la clase Padre Vehículo

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
        super().__init__(color, 4)  # en curso cert no habla de super()
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
    def __init__(self, mensaje):
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
            self._alto = 0
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
# Si cambiamos el orden en que se hereda cambiaría. Nos indica el orden en que ira buscando los métodos, por ejemplo
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

