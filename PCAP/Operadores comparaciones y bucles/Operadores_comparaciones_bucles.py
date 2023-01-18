import time

# ***********************************
# ********  IF  ###############
# ***********************************
miVariable2 = True
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
i = int(s1)  # valueError aquí porque no podemos pasar el string 12.8 a entero a float si podríamos

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
else:  # Ojo en cada iteración, si no cumple, en este caso fin del contador último número: 11
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
for i in range(1, 6):  # cuenta de 1 inlcuido a 5 ( 6 no incluido)
    print(f" Mississippi {i}")
# Write a for loop that counts to five.
    # Body of the loop - print the loop iteration number and the word "Mississippi".
    # Body of the loop - use:
    time.sleep(1)  # suspende la ejecución durante los segundos indicados

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
while c0 > 1:
    if c0 % 2 == 0:
        c0 /= 2
        c0 = int(c0)
    else:
        c0 = 3 * c0 + 1
    step += 1
    print(c0)
print(f'steps = {step}')  # par || impar

# ++++  Ejercicio ++++++
''' Su tarea es escribir su propia función, que se comporta casi exactamente como el método split() original, es decir:
debe aceptar exactamente un argumento: una cadena;debe devolver una lista de palabras creadas a partir de la cadena,
dividida en los lugares donde la cadena contiene espacios en blanco; si la cadena está vacía, la función debería
devolver una lista vacía; su nombre debería ser mysplit() Utilice la plantilla en el editor. Prueba tu código
cuidadosamente '''


def my_split(cadena1):
    try:
        if type(cadena1) != str:
            raise Exception('only accept strings')
        if cadena1.isspace():
            return []
        # return string.split()
        # like is not use split
        cadena1 = cadena1.strip()
        lista1 = []
        auxiliar = 0
        for stx1 in range(len(cadena1)):
            if cadena1[stx1].isspace() or stx1 == (len(cadena1) - 1):
                lista1.append(cadena1[auxiliar:stx1+1].strip())
                auxiliar = stx1
        return lista1
    except Exception as err:
        print('raise an error:', err)


print(my_split("To be or not to be, that is the question"))
# ['To', 'be', 'or', 'not', 'to', 'be,', 'that', 'is', 'the', 'question']
print(my_split("To be or not to be,that is the question"))
# ['To', 'be', 'or', 'not', 'to', 'be,that', 'is', 'the', 'question']
print(my_split("   "))   # []
print(my_split(" abc "))     # ['abc']
print(my_split(""))  # []

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
    for sub in strings:
        total += float(sub)
    print("The total is:", total)

except:
    print(sub, "is not a number.")  # 21.0

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
iban = iban.replace(' ', '')

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
están ocultos dentro de la segunda cadena? si la segunda cadena se da como "vcxzxduybfdsobywuefgas", la respuesta es sí;
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
print(response)  # Yes || con donut como st1 --> No

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
