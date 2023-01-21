
# ***********************************
# ********  Manejo de Archivos ##########
# ***********************************
# podemos trabajar casi cualquier tipo de archivos desde textos hasta imágenes

# ********  Archivos  DE TEXTO ##########
'''Cualquier programa escrito en Python (y no solo en Python, porque esa convención se aplica a prácticamente todos los
lenguajes de programación) no se comunica con los archivos directamente, sino a través de algunas entidades abstractas
que se nombran de manera diferente en diferentes lenguajes o entornos: los términos más utilizados. Son identificadores
o flujos (aquí los usaremos como sinónimos). El programador, que tiene un conjunto más o menos rico de funciones/método.
Puede realizar ciertas operaciones en el flujo, que afectan a los archivos reales, mediante mecanismos contenidos en
el kernel del sistema operativo. De esta forma, puede implementar el proceso de acceso a cualquier archivo, incluso
cuando se desconoce el nombre del archivo en el momento de escribir el programa.

-Acceso a archivos: un concepto de estructura de árbol-
Para conectar (vincular) la transmisión con el archivo, es necesario el uso de una operación explícita. La operación de
conectar la secuencia con un archivo se denomina abrir (open()) el archivo, mientras que desconectar este enlace se
denomina cerrar el archivo close(). Por lo tanto, la conclusión es que la primera operación realizada en la corriente
siempre está abierta y la última está cerrada. El programa, en efecto, es libre de manipular el flujo entre estos
dos eventos y manejar el archivo asociado. Esta libertad está limitada, por supuesto, por las características físicas
del expediente y la forma en que se ha abierto el expediente.

La diferencia principal y más llamativa entre Sistemas Operativos es que debe usar dos separadores diferentes para
los nombres de los directorios: '\' en Windows y '/ 'en Unix/Linux. Si estamos en otra carpeta ( si no solo necesitamos
el nombre y extensión del archivo) podemos especificar la misma. En windows, como en python '\' es un carácter especial
tenemos que poner otro '\' para salvarlo e indicar que no es un carácter especial por eso es 'c:\\ (en la 1º) en mac o
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

La apertura de la transmisión no sólo está asociada con el archivo, sino que también debe declarar la forma en que se
procesará la transmisión. Esta declaración se denomina modo abierto. Si la apertura es exitosa, el programa podrá
realizar solo las operaciones que sean consistentes con el modo de apertura declarado.

Hay dos operaciones básicas realizadas en el flujo:
- leer del flujo: las partes de los datos se recuperan del archivo y se colocan en un área de memoria administrada
por el programa (por ejemplo, una variable);
- escribir en el flujo: las porciones de los datos de la memoria (por ejemplo, una variable) se transfieren al archivo.

Hay tres modos básicos que se utilizan para abrir la secuencia:
+modo de lectura: una secuencia abierta en este modo solo permite operaciones de lectura; intentar escribir en la
                  transmisión causará una excepción (la excepción se llama UnsupportedOperation, que hereda OSError y
                  ValueError, y proviene del módulo io);
+modo de escritura: una secuencia abierta en este modo solo permite operaciones de escritura; intentar leer la
                    transmisión causará la excepción mencionada anteriormente;
+modo de actualización: una secuencia abierta en este modo permite escrituras y lecturas.

Identificadores de archivos
Python asume que cada archivo está oculto detrás de un objeto de una clase adecuada. Diferentes archivos pueden requerir
diferentes conjuntos de operaciones y comportarse de diferentes maneras. Un objeto de una clase adecuada se crea cuando
abre el archivo y lo aniquila en el momento de cerrarlo. Entre estos dos eventos, puede usar el objeto para
especificar qué operaciones se deben realizar en una transmisión en particular. Las operaciones que puede usar están
impuestas por la forma en que abrió el archivo.

En general, el objeto proviene de
IOBase
    - RawIOBase
    - BufferIOBase
    - TextoIOBase

Nunca uses constructores para dar vida a estos objetos. La única forma de obtenerlos es invocando la función llamada
open(). La función analiza los argumentos que ha proporcionado y crea automáticamente el objeto requerido. Si desea
deshacerse del objeto, invoque el método llamado close(). La invocación cortará la conexión con el objeto y el archivo
y eliminará el objeto.

Como tipos de identificadores de archivos, tenemos de texto ( secuencia de caracteres) y binario (secuencias de bits
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

Durante la lectura/escritura de líneas desde/hacia el archivo asociado, no ocurre nada especial en el entorno Unix,
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

- errno.EACCES → Permiso denegado , intenta, por ejemplo, abrir un archivo con el atributo de solo lectura para escribir
- errno.EBADF → Número de archivo erróneo, intenta, por ejemplo, operar con una transmisión sin abrir.
- errno.EEXIST → El archivo existe,  intentas, por ejemplo, renombrar un archivo con su nombre anterior.
- errno.EFBIG → Archivo demasiado grande ( max permitido SO)
- errno.EISDIR → Es un directorio, intenta tratar un nombre de directorio como el nombre de un archivo ordinario.
- errno.EMFILE → Demasiados archivos abiertos (simultáneamente intenta abrir max permitido SO)
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
    while readin > 0:  # mientras recibas bytes en el buffer
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
existe una sintaxis simplificada que automáticamente va a abrir y a cerrar nuestro archivo sin tener que cerrarlo,
después se ejecutan de forma dinámica con __enter__ ( para abrirlo ) y con __exit__ para cerrarlo. Se pueden 
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
